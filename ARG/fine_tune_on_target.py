import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import AutoConfig
import numpy as np
from tqdm import tqdm
from AutoCNN_NewMoE_ASL import GCM_MultiLabelModel
from asym import AsymmetricLoss
from collections import Counter

sys.path.append('data')
from Dataset import ProteinDataset

# -------------------------------
# 🔧 配置
# -------------------------------
model_dir = "outputs/AutoCNN_NewMoE_ASL_outputs_6/best_model"
target_data_path = "processed_data/my_test_encoded_dataset.pt"
batch_size = 4
num_epochs = 50
patience = 5
lr = 1e-5
save_dir = "finetune_on_target_outputs"
os.makedirs(save_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------
# ✅ 加载模型
# -------------------------------
def load_model(model_class, model_dir, device):
    config = AutoConfig.from_pretrained(model_dir)
    model = model_class(config=config)
    state_dict = torch.load(os.path.join(model_dir, "pytorch_model.bin"), map_location=device)
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "") if key.startswith("module.") else key
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)
    return model.to(device)


print("************加载原始模型**************")
model = load_model(GCM_MultiLabelModel, model_dir, device)

# -------------------------------
# ✅ 数据准备 (专注于二分类任务)
# -------------------------------
print("************加载外部数据集**************")
target_dataset = torch.load(target_data_path)


# 统计二分类标签分布
def get_binary_class_distribution(dataset):
    """获取二分类标签的分布"""
    class_counts = Counter()
    for item in dataset:
        label = item['is_arg']
        class_counts[label] += 1

    total_samples = len(dataset)
    print("\n🔬 二分类标签分布统计:")
    print(f"非耐药基因 (0): {class_counts[0]} 个样本 ({class_counts[0] / total_samples * 100:.2f}%)")
    print(f"耐药基因 (1): {class_counts[1]} 个样本 ({class_counts[1] / total_samples * 100:.2f}%)")
    print(f"总样本数: {total_samples}")

    return class_counts


class_counts = get_binary_class_distribution(target_dataset)

# 创建样本权重以平衡类别
sample_weights = []
for item in target_dataset:
    # 对于少数类别样本，增加权重
    weight = 1.0
    if item['is_arg'] == 1:  # 耐药基因样本
        weight = max(weight, class_counts[0] / class_counts[1])  # 反比于类别频率

    sample_weights.append(weight)

# 使用分层抽样确保类别分布均衡
binary_labels = [item['is_arg'] for item in target_dataset]
indices = list(range(len(target_dataset)))
train_indices, val_indices = train_test_split(
    indices,
    test_size=0.1,
    stratify=binary_labels,  # 使用二分类标签进行分层
    random_state=42
)

# 创建子集
target_train = Subset(target_dataset, train_indices)
target_val = Subset(target_dataset, val_indices)

# 验证分层抽样效果
print("\n🔬 分层抽样后二分类标签分布:")
print("训练集分布:")
get_binary_class_distribution(target_train)
print("\n验证集分布:")
get_binary_class_distribution(target_val)

# 创建带权重的随机采样器以平衡训练集
train_weights = [sample_weights[i] for i in train_indices]
train_sampler = WeightedRandomSampler(
    weights=train_weights,
    num_samples=len(train_weights),
    replacement=True
)

# 创建数据加载器
train_loader = DataLoader(
    target_train,
    batch_size=batch_size,
    sampler=train_sampler,  # 使用加权采样器
    pin_memory=True
)
val_loader = DataLoader(
    target_val,
    batch_size=batch_size,
    pin_memory=True
)


# -------------------------------
# 🧪 二分类评估函数
# -------------------------------
def evaluate_binary(model, dataloader):
    model.eval()
    all_preds = []
    all_probs = []
    all_true = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="🔍 Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            resistance_logits, _, _, _ = outputs.logits  # 只关注二分类输出

            # 获取预测概率和类别
            probs = resistance_logits.sigmoid().cpu().numpy()
            preds = (probs > 0.5).astype(int)

            true_labels = batch['is_arg'].cpu().numpy()

            all_preds.append(preds)
            all_probs.append(probs)
            all_true.append(true_labels)

    preds = np.concatenate(all_preds)
    probs = np.concatenate(all_probs)
    true = np.concatenate(all_true)

    # 计算分类报告
    report = classification_report(
        true, preds,
        target_names=["非耐药基因", "耐药基因"],
        zero_division=0
    )

    # 计算AUROC
    if len(np.unique(true)) >= 2:  # 确保有正负样本
        auroc = roc_auc_score(true, probs)
    else:
        auroc = 0.0
        print("⚠️ 无法计算AUROC - 验证集中缺少正样本或负样本")

    return report, auroc


# -------------------------------
# 🔁 二分类微调训练
# -------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=2,
    verbose=True
)


# 二分类损失函数
def binary_loss(outputs, resistance_labels):
    resistance_logits, _, _, _ = outputs.logits
    return nn.BCEWithLogitsLoss()(resistance_logits, resistance_labels.float())


best_auroc = 0
no_improve = 0

for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0
    print(f"\n🔁 Epoch {epoch}/{num_epochs}")
    for batch in tqdm(train_loader, desc="🧪 Training", leave=False):
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        resistance_labels = batch['is_arg'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = binary_loss(outputs, resistance_labels)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    report, auroc = evaluate_binary(model, val_loader)
    print(f"\n📉 Epoch {epoch} | Loss: {total_loss:.4f} | Val AUROC: {auroc:.4f}")
    print("🧬 二分类报告:")
    print(report)

    # 更新学习率
    scheduler.step(auroc)
    current_lr = optimizer.param_groups[0]['lr']
    print(f"当前学习率: {current_lr:.2e}")

    if auroc > best_auroc:
        best_auroc = auroc
        no_improve = 0
        model_save_path = os.path.join(save_dir, "best_model")
        os.makedirs(model_save_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(model_save_path, "pytorch_model.bin"))
        print("✅ Best model_1 saved.")
    else:
        no_improve += 1
        if no_improve >= patience:
            print("⏹️ Early stopping triggered.")
            break

# -------------------------------
# 🧪 Final 测试
# -------------------------------
print("\n🔍 加载最佳模型进行最终评估...")
best_model_path = os.path.join(save_dir, "best_model")
model = load_model(GCM_MultiLabelModel, best_model_path, device)
report, final_auroc = evaluate_binary(model, val_loader)

print("\n📊 目标验证集上的最终评估")
print("🧬 二分类报告:")
print(report)
print(f"🎯 最终AUROC: {final_auroc:.4f}")