import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import EsmModel, EsmConfig




class MultiTaskResistanceModel(nn.Module):
    def __init__(self, config):
        """
        多任务蛋白质抗性预测模型

        参数:
        config: 配置对象，需包含:
          - esm_model_name: ESM预训练模型名称或路径
          - num_mechanism_labels: 抗性机制类别数量
          - num_antibiotic_labels: 抗生素类别数量
          - freeze_esm_layers: 冻结的ESM层数
          - use_remove: 是否使用基因来源预测任务
        """
        super().__init__()

        # 保存配置参数
        self.config = config
        self.use_remove = config.use_remove if hasattr(config, 'use_remove') else True

        # ================= 共享特征提取层 =================
        # 加载ESM模型
        self.esm = EsmModel.from_pretrained(config.esm_model_name)
        esm_hidden_size = self.esm.config.hidden_size

        # 冻结指定层数
        self.freeze_esm_layers(config.freeze_esm_layers)

        # 特征投影层
        self.feature_projector = nn.Sequential(
            nn.Linear(esm_hidden_size, 1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(0.3)
        )

        # ================= 多任务预测头 =================
        # 1. 抗性判断 (二分类)
        self.resistance_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

        # 2. 抗生素分类 (多标签)
        self.antibiotic_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, config.num_antibiotic_labels)
        )

        # 3. 抗性机制 (多标签)
        self.mechanism_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, config.num_mechanism_labels)
        )

        # 4. 基因来源 (二分类) - 可选任务
        if self.use_remove:
            self.remove_head = nn.Sequential(
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(256, 1)
            )

        # 参数初始化
        self.init_weights()

    def freeze_esm_layers(self, num_layers):
        """冻结ESM前N层"""
        if num_layers <= 0:
            return

        frozen = 0

        # 冻结嵌入层
        if hasattr(self.esm, 'embeddings'):
            for param in self.esm.embeddings.parameters():
                param.requires_grad = False
            frozen += 1

        # 冻结编码器层
        if hasattr(self.esm, 'encoder'):
            total_layers = len(self.esm.encoder.layer)
            for i in range(min(num_layers - 1, total_layers)):
                for param in self.esm.encoder.layer[i].parameters():
                    param.requires_grad = False
                frozen += 1



    def init_weights(self):
        """权重初始化"""
        # 特征投影层
        for module in self.feature_projector:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # 任务头
        for head in [self.resistance_head, self.antibiotic_head,
                     self.mechanism_head, self.remove_head if self.use_remove else None]:
            if head is None:
                continue
            for module in head:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_normal_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)

    def forward(self, input_ids, attention_mask=None):
        """
        前向传播

        参数:
        input_ids: token ID张量 [batch_size, seq_len]
        attention_mask: 注意力掩码张量 [batch_size, seq_len]

        返回:
        dict: 包含四个任务输出的字典
        """
        # 通过ESM获取序列表示
        outputs = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # 取[CLS]位置的表示作为序列整体特征
        cls_features = outputs.last_hidden_state[:, 0]

        # 特征投影
        features = self.feature_projector(cls_features)

        # 任务预测
        resistance = self.resistance_head(features)
        antibiotics = self.antibiotic_head(features)
        mechanisms = self.mechanism_head(features)
        remove = self.remove_head(features) if self.use_remove else None

        return {
            "resistance": resistance,  # [batch_size, 1]
            "antibiotics": antibiotics,  # [batch_size, num_antibiotic_labels]
            "mechanisms": mechanisms,  # [batch_size, num_mechanism_labels]
            "remove": remove  # [batch_size, 1] 或 None
        }

    def predict(self, input_ids, attention_mask=None, threshold=0.5):
        """
        完整预测流程

        返回包含概率和类别预测的字典
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)

            # 计算概率
            resistance_prob = torch.sigmoid(outputs["resistance"])
            antibiotics_prob = torch.sigmoid(outputs["antibiotics"])
            mechanisms_prob = torch.sigmoid(outputs["mechanisms"])

            # 转换预测类别
            resistance_pred = (resistance_prob > threshold).float()
            antibiotics_pred = (antibiotics_prob > threshold).float()
            mechanisms_pred = (mechanisms_prob > threshold).float()

            # 如果有remove头
            if self.use_remove and outputs["remove"] is not None:
                remove_prob = torch.sigmoid(outputs["remove"])
                remove_pred = (remove_prob > threshold).float()
            else:
                remove_prob = None
                remove_pred = None

            return {
                "resistance": {
                    "prob": resistance_prob.squeeze(1),
                    "pred": resistance_pred.squeeze(1)
                },
                "antibiotics": {
                    "prob": antibiotics_prob,
                    "pred": antibiotics_pred
                },
                "mechanisms": {
                    "prob": mechanisms_prob,
                    "pred": mechanisms_pred
                },
                "remove": {
                    "prob": remove_prob.squeeze(1) if remove_prob is not None else None,
                    "pred": remove_pred.squeeze(1) if remove_pred is not None else None
                }
            }

    def get_feature_vector(self, input_ids, attention_mask=None):
        """获取特征向量表示（用于迁移学习）"""
        self.eval()
        with torch.no_grad():
            outputs = self.esm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            cls_features = outputs.last_hidden_state[:, 0]
            return self.feature_projector(cls_features)