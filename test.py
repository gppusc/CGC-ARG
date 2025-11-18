import torch
from torch.utils.data import DataLoader
from  MOE import ESM2MultiLabel
from transformers import AutoConfig
from utils import compute_metrics
import numpy as np
from Dataset import  ProteinDataset

def test():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # 加载测试数据
    test_dataset = torch.load("processed_data/test_dataset.pt")
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # 加载最佳模型
    config = AutoConfig.from_pretrained("AutoCNN_Transformer_outputs_5/best_model")
    model = ESM2MultiLabel.from_pretrained("AutoCNN_Transformer_outputs_5/best_model", config=config).to(device)
    model.eval()

    # 收集所有预测结果
    all_preds = {
        "resistance": [],
        "mechanism": [],
        "antibiotic": [],
        "remove":[]
    }
    all_labels = {
        "resistance": [],
        "mechanism": [],
        "antibiotic": [],
        "remove":[]
    }

    with torch.no_grad():
        for batch in test_loader:
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device)
            }
            outputs = model(**inputs)

            # 抗性基因预测
            resistance_probs = torch.sigmoid(outputs.logits[0]).cpu().numpy()
            all_preds["resistance"].extend(resistance_probs)
            all_labels["resistance"].extend(batch["is_arg"].numpy())

            # 抗性机制预测（仅对抗性阳性样本）
            mask = batch["is_arg"] == 1
            if mask.sum() > 0:
                mechanism_probs = torch.sigmoid(outputs.logits[1][mask]).cpu().numpy()
                all_preds["mechanism"].extend(mechanism_probs)
                all_labels["mechanism"].extend(batch["mechanism_labels"][mask].numpy())

                antibiotic_probs = torch.sigmoid(outputs.logits[2][mask]).cpu().numpy()
                all_preds["antibiotic"].extend(antibiotic_probs)
                all_labels["antibiotic"].extend(batch["antibiotic_labels"][mask].numpy())

                remove_probs = torch.sigmoid(outputs.logits[3][mask]).cpu().numpy()
                all_preds["remove"].extend(remove_probs)
                all_labels["remove"].extend(batch["remove_label"][mask].numpy())

    # 计算指标
    resistance_metrics = compute_metrics(
        np.array(all_preds["resistance"]),
        np.array(all_labels["resistance"]),
        task_type="binary"
    )

    mechanism_metrics = compute_metrics(
        np.array(all_preds["mechanism"]),
        np.array(all_labels["mechanism"]),
        task_type="multilabel"
    )

    antibiotic_metrics = compute_metrics(
        np.array(all_preds["antibiotic"]),
        np.array(all_labels["antibiotic"]),
        task_type="multilabel"
    )
    remove_metrics = compute_metrics(
        np.array(all_preds["remove"]),
        np.array(all_labels["remove"]),
        task_type="binary"
    )

    print("\n=== Test Results ===")
    print("=== Resistance (Binary) ===")
    print(f"AUC: {resistance_metrics['roc_auc']:.4f}")
    print(f"Precision: {resistance_metrics['precision']:.4f}")
    print(f"Recall: {resistance_metrics['recall']:.4f}")
    print(f"F1: {resistance_metrics['f1']:.4f}")

    print("\n=== Mechanism (Multilabel) ===")
    print(f"Macro AUC: {mechanism_metrics['roc_auc_macro']:.4f}")
    print(f"Macro Precision: {mechanism_metrics['precision_macro']:.4f}")
    print(f"Macro Recall: {mechanism_metrics['recall_macro']:.4f}")
    print(f"Macro F1: {mechanism_metrics['f1_macro']:.4f}")
    print(f"Micro Precision: {mechanism_metrics['precision_micro']:.4f}")
    print(f"Micro Recall: {mechanism_metrics['recall_micro']:.4f}")
    print(f"Micro F1: {mechanism_metrics['f1_micro']:.4f}")

    print("\n=== Antibiotic (Multilabel) ===")
    print(f"Macro AUC: {antibiotic_metrics['roc_auc_macro']:.4f}")
    print(f"Macro Precision: {antibiotic_metrics['precision_macro']:.4f}")
    print(f"Macro Recall: {antibiotic_metrics['recall_macro']:.4f}")
    print(f"Macro F1: {antibiotic_metrics['f1_macro']:.4f}")
    print(f"Micro Precision: {antibiotic_metrics['precision_micro']:.4f}")
    print(f"Micro Recall: {antibiotic_metrics['recall_micro']:.4f}")
    print(f"Micro F1: {antibiotic_metrics['f1_micro']:.4f}")

    print("=== Remove (Binary) ===")
    print(f"AUC: {remove_metrics['roc_auc']:.4f}")
    print(f"Precision: {remove_metrics['precision']:.4f}")
    print(f"Recall: {remove_metrics['recall']:.4f}")
    print(f"F1: {remove_metrics['f1']:.4f}")


if __name__ == "__main__":
    test()