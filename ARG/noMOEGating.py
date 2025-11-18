from transformers import EsmPreTrainedModel, EsmModel
import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput
from asym import AsymmetricLoss


class AutoCNNLayer(nn.Module):
    def __init__(self, hidden_size, filter_size, window_sizes):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_size, filter_size, kernel_size=ws),
                nn.GELU(),
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
        self.norm = nn.LayerNorm(filter_size)

    def forward(self, x):  # x: (B, H, L)
        conv_outputs = []
        for conv in self.convs:
            out = conv(x)  # (B, filter_size, 1)
            out = out.squeeze(-1)  # (B, filter_size)
            out = self.norm(out)
            conv_outputs.append(out)

        conv_stack = torch.stack(conv_outputs, dim=1)  # (B, num_windows, filter_size)
        attn_scores = self.attention(conv_stack)  # (B, num_windows, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        weighted_sum = (conv_stack * attn_weights).sum(dim=1)  # (B, filter_size)
        return self.dropout(weighted_sum)


class GCM_MultiLabelModel(EsmPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_resistance_labels = 1
        self.num_mechanism_labels = config.num_mechanism_labels
        self.num_antibiotic_labels = config.num_antibiotic_labels
        self.num_remove_labels = 1

        self.esm = EsmModel.from_pretrained(config._name_or_path)
        hidden_size = config.hidden_size
        filter_size = 256
        combined_hidden_size = hidden_size + filter_size
        task_dim = 256  # 任务特定特征维度

        # 特征归一化层
        self.feature_norm = nn.LayerNorm(combined_hidden_size)

        # AutoCNN特征增强
        self.cnn_feature = AutoCNNLayer(hidden_size=hidden_size, filter_size=filter_size, window_sizes=[2, 3, 4])

        # 任务特定投影层（代替专家结构）
        self.task_proj_arg = nn.Sequential(
            nn.Linear(combined_hidden_size, task_dim),
            nn.LayerNorm(task_dim),
            nn.GELU()
        )
        self.task_proj_mech = nn.Sequential(
            nn.Linear(combined_hidden_size, task_dim),
            nn.LayerNorm(task_dim),
            nn.GELU()
        )
        self.task_proj_abc = nn.Sequential(
            nn.Linear(combined_hidden_size, task_dim),
            nn.LayerNorm(task_dim),
            nn.GELU()
        )
        self.task_proj_remove = nn.Sequential(
            nn.Linear(combined_hidden_size, task_dim),
            nn.LayerNorm(task_dim),
            nn.GELU()
        )

        # 分类头
        self.resistance_head = nn.Linear(task_dim, self.num_resistance_labels)
        self.mechanism_head = nn.Linear(task_dim, self.num_mechanism_labels)
        self.antibiotic_head = nn.Linear(task_dim, self.num_antibiotic_labels)
        self.remove_head = nn.Linear(task_dim, self.num_remove_labels)

        # Loss
        self.loss_resistance = AsymmetricLoss(gamma_neg=1, gamma_pos=0, clip=0.05, eps=1e-8)
        self.loss_mechanism = AsymmetricLoss(gamma_neg=2, gamma_pos=0, clip=0.05, eps=1e-8)
        self.loss_antibiotic = AsymmetricLoss(gamma_neg=2, gamma_pos=0, clip=0.05, eps=1e-8)
        self.loss_remove = AsymmetricLoss(gamma_neg=1, gamma_pos=0, clip=0.05, eps=1e-8)

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
        combined_feats = torch.cat([cls_token, cnn_feats], dim=-1)  # (B, H+256)
        # 归一化融合特征
        combined_feats = self.feature_norm(combined_feats)

        # 任务特定投影（直接使用投影层，无MOE）
        f_arg = self.task_proj_arg(combined_feats)
        f_mech = self.task_proj_mech(combined_feats)
        f_abc = self.task_proj_abc(combined_feats)
        f_remove = self.task_proj_remove(combined_feats)

        # 计算logits
        r_logits = self.resistance_head(f_arg).squeeze(-1)
        m_logits = self.mechanism_head(f_mech)
        a_logits = self.antibiotic_head(f_abc)
        v_logits = self.remove_head(f_remove).squeeze(-1)

        # 限制logits范围防止数值问题
        r_logits = torch.clamp(r_logits, min=-10.0, max=10.0)
        m_logits = torch.clamp(m_logits, min=-10.0, max=10.0)
        a_logits = torch.clamp(a_logits, min=-10.0, max=10.0)
        v_logits = torch.clamp(v_logits, min=-10.0, max=10.0)

        loss = None
        if resistance_labels is not None:
            # 计算损失
            loss_resistance = self.loss_resistance(r_logits, resistance_labels.float())
            loss = 0.4 * loss_resistance
            pos_mask = resistance_labels == 1

            if pos_mask.sum() > 0:
                loss_mechanism = self.loss_mechanism(m_logits[pos_mask], mechanism_labels[pos_mask].float())
                loss_antibiotic = self.loss_antibiotic(a_logits[pos_mask], antibiotic_labels[pos_mask].float())
                loss_remove = self.loss_remove(v_logits[pos_mask], remove_labels[pos_mask].float())

                loss += 0.2 * loss_mechanism
                loss += 0.2 * loss_antibiotic
                loss += 0.2 * loss_remove

        # 如果损失是NaN，设置为零
        if loss is not None and (torch.isnan(loss) or torch.isinf(loss)):
            loss = torch.tensor(0.0, device=r_logits.device, requires_grad=True)

        return SequenceClassifierOutput(
            loss=loss,
            logits=(r_logits, m_logits, a_logits, v_logits),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )