from transformers import EsmPreTrainedModel, EsmModel
import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput


class MoELayer(nn.Module):
    def __init__(self, hidden_size, expert_size, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, expert_size),
                nn.GELU(),
                nn.Linear(expert_size, hidden_size)
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(hidden_size, num_experts)
        self.dropout = nn.Dropout(0.2)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):  # x: (B, H)
        gate_logits = self.gate(x)
        gate_weights = torch.softmax(gate_logits, dim=-1)  # (B, num_experts)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # (B, num_experts, H)
        gated_output = (expert_outputs * gate_weights.unsqueeze(-1)).sum(dim=1)  # (B, H)
        return self.dropout(self.norm(gated_output))


class ESM2MultiLabel(EsmPreTrainedModel):
    def __init__(self, config, filter_size=64):
        super().__init__(config)

        # 加载权重
        pos_weight_mech = torch.load('processed_data/pos_weight_mech.pt')
        pos_weight_anti = torch.load('processed_data/pos_weight_anti.pt')

        # 类别数
        self.num_resistance_labels = 1
        self.num_mechanism_labels = config.num_mechanism_labels
        self.num_antibiotic_labels = config.num_antibiotic_labels
        self.num_remove_labels = 1

        # Backbone: ESM
        self.esm = EsmModel.from_pretrained(config._name_or_path)
        self.dropout = nn.Dropout(0.1)

        # MoE 分支
        self.moe_branch_cls = MoELayer(config.hidden_size, expert_size=256, num_experts=4)
        self.moe_branch_seq = MoELayer(config.hidden_size, expert_size=256, num_experts=4)

        # 分类头
        total_feats = config.hidden_size * 2
        self.dropout2 = nn.Dropout(0.2)
        self.resistance_head = nn.Linear(total_feats, self.num_resistance_labels)
        self.mechanism_head = nn.Linear(total_feats, self.num_mechanism_labels)
        self.antibiotic_head = nn.Linear(total_feats, self.num_antibiotic_labels)
        self.remove_head = nn.Linear(total_feats, self.num_remove_labels)

        # 损失函数
        self.loss_resistance = nn.BCEWithLogitsLoss()
        self.loss_mechanism = nn.BCEWithLogitsLoss(pos_weight=pos_weight_mech)
        self.loss_antibiotic = nn.BCEWithLogitsLoss(pos_weight=pos_weight_anti)
        self.loss_remove = nn.BCEWithLogitsLoss()

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        resistance_labels=None,
        mechanism_labels=None,
        antibiotic_labels=None,
        remove_labels=None
    ):
        outputs = self.esm(input_ids, attention_mask)
        seq_out = self.dropout(outputs.last_hidden_state)  # (B, L, H)

        # MoE 表达提取
        cls_feats = seq_out[:, 0, :]
        cls_moe_out = self.moe_branch_cls(cls_feats)

        seq_avg = seq_out.mean(dim=1)
        seq_moe_out = self.moe_branch_seq(seq_avg)

        feat = torch.cat([cls_moe_out, seq_moe_out], dim=1)

        # 分类 logits
        r_logits = self.resistance_head(self.dropout2(feat)).squeeze(-1)
        m_logits = self.mechanism_head(self.dropout2(feat))
        a_logits = self.antibiotic_head(self.dropout2(feat))
        v_logits = self.remove_head(self.dropout2(feat)).squeeze(-1)

        total_loss = None
        if resistance_labels is not None:
            total_loss = 0.4 * self.loss_resistance(r_logits, resistance_labels.float())

            pos_mask = resistance_labels == 1
            neg_mask = resistance_labels == 0

            # 子任务损失（正样本参与主学习，负样本低权重辅助）
            if pos_mask.sum() > 0:
                total_loss += 0.25 * self.loss_mechanism(
                    m_logits[pos_mask], mechanism_labels[pos_mask].float()
                )
                total_loss += 0.25 * self.loss_antibiotic(
                    a_logits[pos_mask], antibiotic_labels[pos_mask].float()
                )
                total_loss += 0.1 * self.loss_remove(
                    v_logits[pos_mask], remove_labels[pos_mask].float()
                )

            if neg_mask.sum() > 0:
                total_loss += 0.05 * self.loss_mechanism(
                    m_logits[neg_mask], mechanism_labels[neg_mask].float()
                )
                total_loss += 0.05 * self.loss_antibiotic(
                    a_logits[neg_mask], antibiotic_labels[neg_mask].float()
                )
                total_loss += 0.05 * self.loss_remove(
                    v_logits[neg_mask], remove_labels[neg_mask].float()
                )

        return SequenceClassifierOutput(
            loss=total_loss,
            logits=(r_logits, m_logits, a_logits, v_logits),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
