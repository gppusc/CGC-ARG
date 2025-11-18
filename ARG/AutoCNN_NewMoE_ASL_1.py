from transformers import EsmPreTrainedModel, EsmModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import SequenceClassifierOutput
from asym import AsymmetricLoss


class AutoCNNLayer(nn.Module):
    def __init__(self, hidden_size, filter_size, window_sizes):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_size, filter_size, kernel_size=ws),
                nn.GELU(),
                nn.AdaptiveMaxPool1d(1),
                nn.Dropout(0.4)
            )
            for ws in window_sizes
        ])
        self.attention = nn.Sequential(
            nn.Linear(filter_size, 64),
            nn.Dropout(0.3),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Dropout(0.3)
        )
        self.norm = nn.LayerNorm(filter_size)

    def forward(self, x):
        conv_outputs = []
        for conv in self.convs:
            out = conv(x).squeeze(-1)
            out = self.norm(out)
            conv_outputs.append(out)

        conv_stack = torch.stack(conv_outputs, dim=1)
        attn_scores = self.attention(conv_stack)
        attn_weights = torch.softmax(attn_scores, dim=1)
        weighted_sum = (conv_stack * attn_weights).sum(dim=1)
        return weighted_sum


class FeatureFusion(nn.Module):
    """修复维度问题的门控特征融合"""

    def __init__(self, hidden_size, cnn_size):
        super().__init__()
        # 添加投影层使维度匹配
        self.cnn_proj = nn.Linear(cnn_size, hidden_size)
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, cls_token, cnn_feats):
        # 将CNN特征投影到与CLS token相同的维度
        cnn_projected = self.cnn_proj(cnn_feats)
        # 生成门控信号
        gate = self.gate(cls_token)
        # 融合特征
        fused = gate * cls_token + (1 - gate) * cnn_projected
        return self.norm(fused)


class ExpertLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU()
            ) for _ in range(num_experts)
        ])
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        expert_outputs = [self.dropout(expert(x)) for expert in self.experts]
        return torch.stack(expert_outputs, dim=1)

class GatingMechanism(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, shared_feats, expert_outs):
        weights = torch.softmax(self.gate(shared_feats), dim=-1)  # (B, num_experts)
        weighted_experts = (weights.unsqueeze(-1) * expert_outs).sum(dim=1)  # (B, output_dim)
        return weighted_experts

class DiversityGating(GatingMechanism):
    """带多样性正则的门控机制"""

    def __init__(self, input_dim, num_experts):
        super().__init__(input_dim, num_experts)
        self.dropout = nn.Dropout(0.2)

    def forward(self, shared_feats, expert_outs):
        weights = torch.softmax(self.gate(shared_feats), dim=-1)
        weights = self.dropout(weights)

        # 计算专家多样性正则
        expert_flat = expert_outs.flatten(start_dim=2)
        expert_corr = torch.matmul(expert_flat, expert_flat.transpose(1, 2))
        eye_mask = torch.eye(expert_corr.size(1), device=expert_corr.device).bool()
        expert_corr = expert_corr.masked_fill(eye_mask, 0)
        div_loss = torch.mean(torch.triu(expert_corr, diagonal=1) ** 2)

        weighted_experts = (weights.unsqueeze(-1) * expert_outs).sum(dim=1)
        return weighted_experts, div_loss


class AntibioticAdapter(nn.Module):
    """抗生素任务专用适配器"""

    def __init__(self, input_dim, bottleneck=128):
        super().__init__()
        self.down = nn.Linear(input_dim, bottleneck)
        self.up = nn.Linear(bottleneck, input_dim)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        residual = x
        x = F.gelu(self.down(x))
        x = self.up(x)
        return self.norm(residual + x)


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
        expert_dim = 256
        num_shared_experts = 3
        num_task_experts = 3

        # 特征提取增强
        self.cnn_feature = AutoCNNLayer(
            hidden_size=hidden_size,
            filter_size=filter_size,
            window_sizes=[2, 3, 4]
        )
        self.feature_fusion = FeatureFusion(hidden_size, filter_size)

        # Shared Expert
        self.shared_experts = ExpertLayer(hidden_size, expert_dim, num_shared_experts)

        # Task-specific Experts
        self.experts_arg = ExpertLayer(hidden_size, expert_dim, num_task_experts)
        self.experts_mech = ExpertLayer(hidden_size, expert_dim, num_task_experts)
        self.experts_abc = ExpertLayer(hidden_size, expert_dim, num_task_experts)
        self.experts_remove = ExpertLayer(hidden_size, expert_dim, num_task_experts)

        # 抗生素适配器
        self.antibiotic_adapter = AntibioticAdapter(expert_dim)

        # Gating Mechanism
        self.gate_arg = DiversityGating(hidden_size, num_shared_experts + num_task_experts)
        self.gate_mech = DiversityGating(hidden_size, num_shared_experts + num_task_experts)
        self.gate_abc = DiversityGating(hidden_size, num_shared_experts + num_task_experts)
        self.gate_remove = DiversityGating(hidden_size, num_shared_experts + num_task_experts)

        # Classification heads
        total_feat_dim = expert_dim

        # 增强任务归一化
        self.task_norm_arg = nn.Sequential(
            nn.LayerNorm(total_feat_dim),
            nn.Dropout(0.3)
        )
        self.task_norm_mech = nn.Sequential(
            nn.LayerNorm(total_feat_dim),
            nn.Dropout(0.3)
        )
        self.task_norm_abc = nn.Sequential(
            nn.LayerNorm(total_feat_dim),
            nn.Dropout(0.4)
        )
        self.task_norm_remove = nn.Sequential(
            nn.LayerNorm(total_feat_dim),
            nn.Dropout(0.3)
        )

        self.resistance_head = nn.Linear(total_feat_dim, self.num_resistance_labels)
        self.mechanism_head = nn.Linear(total_feat_dim, self.num_mechanism_labels)
        self.antibiotic_head = nn.Linear(total_feat_dim, self.num_antibiotic_labels)
        self.remove_head = nn.Linear(total_feat_dim, self.num_remove_labels)

        # 调整损失函数参数
        self.loss_resistance = AsymmetricLoss(gamma_neg=1, gamma_pos=0, clip=0.05, eps=1e-8)
        self.loss_mechanism = AsymmetricLoss(gamma_neg=2, gamma_pos=0, clip=0.05, eps=1e-8)
        self.loss_antibiotic = AsymmetricLoss(
            gamma_neg=4,
            gamma_pos=1,
            clip=0.1,
            eps=1e-8
        )
        self.loss_remove = AsymmetricLoss(gamma_neg=1, gamma_pos=0, clip=0.05, eps=1e-8)

        self.post_init()

    def forward(
            self, input_ids=None, attention_mask=None,
            resistance_labels=None, mechanism_labels=None,
            antibiotic_labels=None, remove_labels=None
    ):
        outputs = self.esm(input_ids, attention_mask)

        # 特征提取与融合
        cls_token = outputs.last_hidden_state[:, 0, :]
        token_embeddings = outputs.last_hidden_state.transpose(1, 2)
        cnn_feats = self.cnn_feature(token_embeddings)
        fused_feats = self.feature_fusion(cls_token, cnn_feats)

        # Expert outputs
        shared_out = self.shared_experts(fused_feats)
        arg_out = self.experts_arg(fused_feats)
        mech_out = self.experts_mech(fused_feats)
        abc_out = self.experts_abc(fused_feats)
        rm_out = self.experts_remove(fused_feats)

        # 抗生素特征增强
        abc_out = self.antibiotic_adapter(abc_out)

        # Task-specific gated fusion
        def task_input(task_out, gate):
            expert_combined = torch.cat([task_out, shared_out], dim=1)
            weighted, div_loss = gate(fused_feats, expert_combined)
            return weighted, div_loss

        f_arg, div_loss_arg = task_input(arg_out, self.gate_arg)
        f_mech, div_loss_mech = task_input(mech_out, self.gate_mech)
        f_abc, div_loss_abc = task_input(abc_out, self.gate_abc)
        f_remove, div_loss_remove = task_input(rm_out, self.gate_remove)

        # 合并多样性损失
        div_loss = 0.25 * (div_loss_arg + div_loss_mech + div_loss_abc + div_loss_remove)

        # 任务特征归一化
        f_arg = self.task_norm_arg(f_arg)
        f_mech = self.task_norm_mech(f_mech)
        f_abc = self.task_norm_abc(f_abc)
        f_remove = self.task_norm_remove(f_remove)

        # 计算logits
        r_logits = torch.clamp(self.resistance_head(f_arg).squeeze(-1), -10.0, 10.0)
        m_logits = torch.clamp(self.mechanism_head(f_mech), -10.0, 10.0)
        a_logits = torch.clamp(self.antibiotic_head(f_abc), -10.0, 10.0)
        v_logits = torch.clamp(self.remove_head(f_remove).squeeze(-1), -10.0, 10.0)

        loss = None
        if resistance_labels is not None:
            # 基础损失计算
            loss_resistance = self.loss_resistance(r_logits, resistance_labels.float())
            loss = 0.4 * loss_resistance

            # 动态抗生素损失权重
            antibiotic_pos_rate = antibiotic_labels.float().mean().clamp(min=0.1, max=0.9)
            abc_weight = 0.25 + (0.15 * antibiotic_pos_rate)

            pos_mask = resistance_labels == 1
            if pos_mask.sum() > 0:
                loss_mechanism = self.loss_mechanism(m_logits[pos_mask], mechanism_labels[pos_mask].float())
                loss_antibiotic = self.loss_antibiotic(a_logits[pos_mask], antibiotic_labels[pos_mask].float())
                loss_remove = self.loss_remove(v_logits[pos_mask], remove_labels[pos_mask].float())

                loss += 0.2 * loss_mechanism
                loss += abc_weight * loss_antibiotic
                loss += 0.2 * loss_remove

            # 添加多样性正则
            loss += 0.1 * div_loss

        # NaN/Inf处理
        if loss is not None:
            if torch.isnan(loss) or torch.isinf(loss):
                loss = torch.tensor(0.0, device=r_logits.device, requires_grad=True)

        return SequenceClassifierOutput(
            loss=loss,
            logits=(r_logits, m_logits, a_logits, v_logits),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )