from transformers import EsmPreTrainedModel, EsmModel
import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput


class AutoCNNLayer(nn.Module):
    def __init__(self, hidden_size, filter_size, window_sizes):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_size, filter_size, kernel_size=ws),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1)
            )
            for ws in window_sizes
        ])
        self.attention = nn.Sequential(
            nn.Linear(filter_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):  # x: (B, H, L)
        conv_outputs = [conv(x).squeeze(-1) for conv in self.convs]  # (B, filter_size)
        conv_stack = torch.stack(conv_outputs, dim=1)  # (B, num_windows, filter_size)
        attn_scores = self.attention(conv_stack)       # (B, num_windows, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        weighted_sum = (conv_stack * attn_weights).sum(dim=1)  # (B, filter_size)
        return self.dropout(weighted_sum)


class ExpertLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.ReLU()
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        return torch.stack([expert(x) for expert in self.experts], dim=1)  # (B, num_experts, output_dim)


class GatingMechanism(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, shared_feats, expert_outs):
        weights = torch.softmax(self.gate(shared_feats), dim=-1)  # (B, num_experts)
        weighted_experts = (weights.unsqueeze(-1) * expert_outs).sum(dim=1)  # (B, output_dim)
        return weighted_experts


class GCM_MultiLabelModel(EsmPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_resistance_labels = 1
        self.num_mechanism_labels = config.num_mechanism_labels
        self.num_antibiotic_labels = config.num_antibiotic_labels
        self.num_remove_labels = 1

        self.esm = EsmModel.from_pretrained(config._name_or_path)
        hidden_size = config.hidden_size
        filter_size = 256  # AutoCNNLayer 输出维度
        combined_hidden_size = hidden_size + filter_size
        expert_dim = 256
        num_shared_experts = 3
        num_task_experts = 3

        # AutoCNN特征增强
        self.cnn_feature = AutoCNNLayer(hidden_size=hidden_size, filter_size=filter_size, window_sizes=[2, 3, 4])

        # Shared Expert
        self.shared_experts = ExpertLayer(combined_hidden_size, expert_dim, num_shared_experts)

        # Task-specific Experts
        self.experts_arg      = ExpertLayer(combined_hidden_size, expert_dim, num_task_experts)
        self.experts_mech     = ExpertLayer(combined_hidden_size, expert_dim, num_task_experts)
        self.experts_abc      = ExpertLayer(combined_hidden_size, expert_dim, num_task_experts)
        self.experts_remove   = ExpertLayer(combined_hidden_size, expert_dim, num_task_experts)

        # Gating Mechanism
        self.gate_arg     = GatingMechanism(combined_hidden_size, num_shared_experts + num_task_experts)
        self.gate_mech    = GatingMechanism(combined_hidden_size, num_shared_experts + num_task_experts)
        self.gate_abc     = GatingMechanism(combined_hidden_size, num_shared_experts + num_task_experts)
        self.gate_remove  = GatingMechanism(combined_hidden_size, num_shared_experts + num_task_experts)

        # Classification heads
        total_feat_dim = expert_dim
        self.resistance_head = nn.Linear(total_feat_dim, self.num_resistance_labels)
        self.mechanism_head  = nn.Linear(total_feat_dim, self.num_mechanism_labels)
        self.antibiotic_head = nn.Linear(total_feat_dim, self.num_antibiotic_labels)
        self.remove_head     = nn.Linear(total_feat_dim, self.num_remove_labels)

        # Loss
        self.loss_resistance = nn.BCEWithLogitsLoss()
        self.loss_mechanism  = nn.BCEWithLogitsLoss(pos_weight=torch.load('processed_data/pos_weight_mech.pt'))
        self.loss_antibiotic = nn.BCEWithLogitsLoss(pos_weight=torch.load('processed_data/pos_weight_anti.pt'))
        self.loss_remove     = nn.BCEWithLogitsLoss()

        self.post_init()

    def forward(
        self, input_ids=None, attention_mask=None,
        resistance_labels=None, mechanism_labels=None,
        antibiotic_labels=None, remove_labels=None
    ):
        outputs = self.esm(input_ids, attention_mask)

        # 获取 [CLS] token 向量
        cls_token = outputs.last_hidden_state[:, 0, :]  # (B, H)

        # 提取卷积特征并拼接
        token_embeddings = outputs.last_hidden_state.transpose(1, 2)  # (B, H, L)
        cnn_feats = self.cnn_feature(token_embeddings)  # (B, 256)
        cls = torch.cat([cls_token, cnn_feats], dim=-1)  # (B, H+256)

        # Expert outputs
        shared_out = self.shared_experts(cls)
        arg_out    = self.experts_arg(cls)
        mech_out   = self.experts_mech(cls)
        abc_out    = self.experts_abc(cls)
        rm_out     = self.experts_remove(cls)

        # Task-specific gated fusion
        task_input = lambda task_out, gate: gate(cls, torch.cat([task_out, shared_out], dim=1))

        f_arg    = task_input(arg_out, self.gate_arg)
        f_mech   = task_input(mech_out, self.gate_mech)
        f_abc    = task_input(abc_out, self.gate_abc)
        f_remove = task_input(rm_out, self.gate_remove)

        r_logits = self.resistance_head(f_arg).squeeze(-1)
        m_logits = self.mechanism_head(f_mech)
        a_logits = self.antibiotic_head(f_abc)
        v_logits = self.remove_head(f_remove).squeeze(-1)

        loss = None
        if resistance_labels is not None:
            loss = 0.4 * self.loss_resistance(r_logits, resistance_labels.float())
            pos_mask = resistance_labels == 1
            if pos_mask.sum() > 0:
                loss += 0.2 * self.loss_mechanism(m_logits[pos_mask], mechanism_labels[pos_mask].float())
                loss += 0.2 * self.loss_antibiotic(a_logits[pos_mask], antibiotic_labels[pos_mask].float())
                loss += 0.2 * self.loss_remove(v_logits[pos_mask], remove_labels[pos_mask].float())

        return SequenceClassifierOutput(
            loss=loss,
            logits=(r_logits, m_logits, a_logits, v_logits),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
