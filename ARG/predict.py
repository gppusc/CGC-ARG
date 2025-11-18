import sys
import os
import argparse
import yaml

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from ARG_Cleaned_1.AutoCNN_NewMoE_ASL import GCM_MultiLabelModel

sys.path.append('data')
from labels import mechanism_labels, antibiotic_labels

MECHANISM_NAMES = mechanism_labels
ANTIBIOTIC_NAMES = antibiotic_labels


class ProteinPredictionDataset(Dataset):
    def __init__(self, sequences, ids, classes, real_labels, tokenizer, max_length=1024, mask_ratio=0.05,
                 training=False):
        self.sequences = sequences
        self.ids = ids
        self.classes = classes
        self.real_labels = real_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_ratio = mask_ratio
        self.training = training
        self.amino_acids = list("ACDEFGHIKLMNPQRSTVWYXZUBO")

        self.is_arg = [0] * len(sequences)
        self.mechanism_labels = [[0.0] * len(MECHANISM_NAMES)] * len(sequences)
        self.antibiotic_labels = [[0.0] * len(ANTIBIOTIC_NAMES)] * len(sequences)
        self.remove_labels = [0] * len(sequences)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = str(self.sequences[idx])
        sample_id = self.ids[idx]
        sample_class = self.classes[idx]
        real_label = self.real_labels[idx]

        if len(seq) > self.max_length:
            start = np.random.randint(0, len(seq) - self.max_length + 1)
            seq = seq[start:start + self.max_length]

        if self.training:
            seq = self.random_mask(seq, self.mask_ratio)

        inputs = self.tokenizer(
            seq,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "id": sample_id,
            "class": sample_class,
            "real_label": real_label,
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "is_arg": torch.tensor(self.is_arg[idx], dtype=torch.long),
            "mechanism_labels": torch.tensor(self.mechanism_labels[idx], dtype=torch.float),
            "antibiotic_labels": torch.tensor(self.antibiotic_labels[idx], dtype=torch.float),
            "remove_label": torch.tensor(self.remove_labels[idx], dtype=torch.long)
        }

    def random_mask(self, seq, ratio):
        seq = list(seq)
        for i in range(len(seq)):
            if seq[i] in self.amino_acids and np.random.rand() < ratio:
                seq[i] = 'X'
        return ''.join(seq)


def predict(test_loader, model, device, threshold=0.5):
    model.eval()

    all_ids = []
    all_classes = []
    all_real_labels = []
    all_resistance_probs = []
    all_mechanism_probs = []
    all_antibiotic_probs = []
    all_remove_probs = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "resistance_labels": batch["is_arg"].to(device),
                "mechanism_labels": batch["mechanism_labels"].to(device),
                "antibiotic_labels": batch["antibiotic_labels"].to(device),
                "remove_labels": batch["remove_label"].to(device),
            }

            sample_ids = batch["id"]
            all_ids.extend(sample_ids)

            all_classes.extend(batch["class"])
            all_real_labels.extend(batch["real_label"].tolist())

            outputs = model(**inputs)

            resistance_probs = torch.sigmoid(outputs.logits[0]).squeeze().detach().cpu().numpy()
            mechanism_probs = torch.sigmoid(outputs.logits[1]).detach().cpu().numpy()
            antibiotic_probs = torch.sigmoid(outputs.logits[2]).detach().cpu().numpy()
            remove_probs = torch.sigmoid(outputs.logits[3]).squeeze().detach().cpu().numpy()

            resistance_probs = resistance_probs.tolist()
            remove_probs = remove_probs.tolist()

            if isinstance(resistance_probs, float):
                resistance_probs = [resistance_probs]
                remove_probs = [remove_probs]

            all_resistance_probs.extend(resistance_probs)
            all_mechanism_probs.extend(mechanism_probs)
            all_antibiotic_probs.extend(antibiotic_probs)
            all_remove_probs.extend(remove_probs)

    results = pd.DataFrame({
        "ID": all_ids,
        "classes": all_classes,
        "real_label": all_real_labels,
        "Resistance_Probability": all_resistance_probs,
        "Resistance_Prediction": [1 if p >= threshold else 0 for p in all_resistance_probs],
        "Remove_Probability": all_remove_probs,
        "Remove_Prediction": [1 if p >= threshold else 0 for p in all_remove_probs]
    })

    for i, name in enumerate(MECHANISM_NAMES):
        results[f"Mechanism_{name}_Probability"] = [p[i] for p in all_mechanism_probs]
        results[f"Mechanism_{name}_Prediction"] = [1 if p[i] >= threshold else 0 for p in all_mechanism_probs]

    for i, name in enumerate(ANTIBIOTIC_NAMES):
        results[f"Antibiotic_{name}_Probability"] = [p[i] for p in all_antibiotic_probs]
        results[f"Antibiotic_{name}_Prediction"] = [1 if p[i] >= threshold else 0 for p in all_antibiotic_probs]

    return results


def calculate_metrics(y_true, y_pred, y_probs):
    """计算并打印分类评估指标，包括AUC"""
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # 计算AUC
    try:
        auc = roc_auc_score(y_true, y_probs)
    except ValueError as e:
        print(f"⚠️ 无法计算AUC: {e}")
        auc = float('nan')  # 如果无法计算，设为NaN

    print(f"📊 模型评估指标:")
    print(f"  - 准确率 (Accuracy): {acc:.4f}")
    print(f"  - 精确率 (Precision): {precision:.4f}")
    print(f"  - 召回率 (Recall): {recall:.4f}")
    print(f"  - F1值 (F1-Score): {f1:.4f}")
    print(f"  - AUC值 (AUC): {auc:.4f}")

    # 计算并显示类别分布
    unique, counts = np.unique(y_true, return_counts=True)
    class_dist = dict(zip(unique, counts))
    print(f"📈 类别分布 (真实标签):")
    for cls, count in class_dist.items():
        print(f"  - 类别 {cls}: {count} 样本 ({count / len(y_true) * 100:.2f}%)")

    # 返回指标值
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "auc": auc}



def few_shot_finetune(model, tokenizer, csv_path, device, max_length, batch_size, lr, epochs):
    print(f"🔧 Starting few-shot fine-tuning with data: {csv_path}")
    df = pd.read_csv(csv_path)
    if 'Sequence' not in df.columns or 'real_label' not in df.columns:
        raise ValueError("Few-shot CSV must contain 'Sequence' and 'real_label' columns")

    sequences = df['Sequence'].tolist()
    ids = df['ID'].tolist() if 'ID' in df.columns else [f"sample_{i}" for i in range(len(sequences))]
    classes = df['classes'].tolist() if 'classes' in df.columns else [""] * len(sequences)
    real_labels = df['real_label'].tolist()

    dataset = ProteinPredictionDataset(sequences, ids, classes, real_labels, tokenizer, max_length, training=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "resistance_labels": batch["is_arg"].to(device),
                "mechanism_labels": batch["mechanism_labels"].to(device),
                "antibiotic_labels": batch["antibiotic_labels"].to(device),
                "remove_labels": batch["remove_label"].to(device),
            }
            outputs = model(**inputs)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

    model.eval()
    print("✅ Few-shot fine-tuning completed.")

# predict, calculate_metrics as-is from your original code
# main function with few-shot support

def main():
    parser = argparse.ArgumentParser(description="Predict antibiotic resistance with optional few-shot fine-tuning")
    parser.add_argument("--model_dir", type=str,
                        default="/liymai24/hjh/codes/kkkk/ARG_Cleaned_1/outputs/AutoCNN_NewMoE_ASL_outputs_6/best_model")
    parser.add_argument("--test_csv", type=str, default="predict_data_1/data_aa.csv")
    parser.add_argument("--output", type=str, default="predictions_data_aa.csv")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tokenizer", type=str, default="/liymai24/hjh/codes/kkkk/ESM2_t30_150M_UR50D")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--mask_ratio", type=float, default=0)
    parser.add_argument("--few_shot_csv", type=str, default=None)
    parser.add_argument("--few_shot_epochs", type=int, default=3)
    parser.add_argument("--few_shot_lr", type=float, default=2e-5)
    args = parser.parse_args()

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    test_df = pd.read_csv(args.test_csv)
    if 'ID' not in test_df.columns or 'Sequence' not in test_df.columns:
        raise ValueError("Test CSV must contain 'ID' and 'Sequence' columns")

    if 'classes' not in test_df.columns:
        test_df['classes'] = ""
    if 'real_label' not in test_df.columns:
        test_df['real_label'] = 0

    test_dataset = ProteinPredictionDataset(
        sequences=test_df['Sequence'].tolist(),
        ids=test_df['ID'].tolist(),
        classes=test_df['classes'].tolist(),
        real_labels=test_df['real_label'].tolist(),
        tokenizer=tokenizer,
        max_length=args.max_length,
        training=False
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    config = AutoConfig.from_pretrained(args.model_dir)
    config.rare_mechanism_indices = getattr(config, 'rare_mechanism_indices', [])
    config.rare_antibiotic_indices = getattr(config, 'rare_antibiotic_indices', [])

    model = GCM_MultiLabelModel(config).to(device)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, "pytorch_model.bin"), map_location=device))
    model.eval()

    if args.few_shot_csv:
        few_shot_finetune(model, tokenizer, args.few_shot_csv, device, args.max_length, args.batch_size, args.few_shot_lr, args.few_shot_epochs)

    predictions = predict(test_loader, model, device, args.threshold)

    y_true = [int(x) for x in predictions["real_label"].tolist()]
    y_pred = [int(x) for x in predictions["Resistance_Prediction"].tolist()]
    y_probs = predictions["Resistance_Probability"].tolist()

    metrics = calculate_metrics(y_true, y_pred, y_probs)
    metrics_output = os.path.splitext(args.output)[0] + "_metrics.txt"
    with open(metrics_output, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")

    predictions.to_csv(args.output, index=False)
    print(f"✅ Predictions saved to {args.output}")

if __name__ == "__main__":
    main()













