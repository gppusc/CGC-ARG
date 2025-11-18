import sys

import torch
from torch.utils.data import DataLoader
from AutoCNN_NewMoE_ASL import GCM_MultiLabelModel
from transformers import AutoConfig
from utils_1 import compute_metrics, compute_class_metrics  # 新增compute_class_metrics函数
import numpy as np

sys.path.append('data')
from labels import mechanism_labels, antibiotic_labels
from Dataset import ProteinDataset


def test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载测试数据
    test_dataset = torch.load("processed_data/my_test_encoded_dataset.pt")
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # 加载最佳模型
    config = AutoConfig.from_pretrained(
        "/liymai24/hjh/codes/kkkk/ARG_Cleaned_1/outputs/AutoCNN_NewMoE_ASL_outputs_6/best_model")
    model = GCM_MultiLabelModel.from_pretrained(
        "/liymai24/hjh/codes/kkkk/ARG_Cleaned_1/outputs/AutoCNN_NewMoE_ASL_outputs_6/best_model",
        config=config).to(device)
    model.eval()

    # 收集所有预测结果
    all_preds = {
        "resistance": [],
        "mechanism": [],
        "antibiotic": [],
        "remove": []
    }
    all_labels = {
        "resistance": [],
        "mechanism": [],
        "antibiotic": [],
        "remove": []
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

    # 计算正负样本指标 - Resistance
    resistance_preds = np.array(all_preds["resistance"])
    resistance_labels = np.array(all_labels["resistance"])

    # 正样本指标 (抗性基因阳性)
    pos_resistance_metrics = compute_class_metrics(
        resistance_preds,
        resistance_labels,
        target_class=1
    )

    # 负样本指标 (抗性基因阴性)
    neg_resistance_metrics = compute_class_metrics(
        resistance_preds,
        resistance_labels,
        target_class=0
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

    # 计算正负样本指标 - Remove
    remove_preds = np.array(all_preds["remove"])
    remove_labels = np.array(all_labels["remove"])

    # 正样本指标 (需要移除)
    pos_remove_metrics = compute_class_metrics(
        remove_preds,
        remove_labels,
        target_class=1
    )

    # 负样本指标 (不需要移除)
    neg_remove_metrics = compute_class_metrics(
        remove_preds,
        remove_labels,
        target_class=0
    )

    print("\n=== Test Results ===")
    print("=== Resistance (Binary) ===")
    print(f"AUC: {resistance_metrics['roc_auc']:.4f}")
    print(f"Precision: {resistance_metrics['precision']:.4f}")
    print(f"Recall: {resistance_metrics['recall']:.4f}")
    print(f"F1: {resistance_metrics['f1']:.4f}")

    # 打印正负样本指标
    print("\nResistance Class Metrics:")
    print("  Positive Class (抗性基因阳性):")
    print(f"    Precision: {pos_resistance_metrics['precision']:.4f}")
    print(f"    Recall: {pos_resistance_metrics['recall']:.4f}")
    print(f"    F1: {pos_resistance_metrics['f1']:.4f}")
    print(f"    Support: {pos_resistance_metrics['support']}")

    print("  Negative Class (抗性基因阴性):")
    print(f"    Precision: {neg_resistance_metrics['precision']:.4f}")
    print(f"    Recall: {neg_resistance_metrics['recall']:.4f}")
    print(f"    F1: {neg_resistance_metrics['f1']:.4f}")
    print(f"    Support: {neg_resistance_metrics['support']}")

    print("\n=== Mechanism (Multilabel) ===")
    print(f"Macro AUC: {mechanism_metrics['roc_auc_macro']:.4f}")
    print(f"Macro Precision: {mechanism_metrics['precision_macro']:.4f}")
    print(f"Macro Recall: {mechanism_metrics['recall_macro']:.4f}")
    print(f"Macro F1: {mechanism_metrics['f1_macro']:.4f}")
    print(f"Micro Precision: {mechanism_metrics['precision_micro']:.4f}")
    print(f"Micro Recall: {mechanism_metrics['recall_micro']:.4f}")
    print(f"Micro F1: {mechanism_metrics['f1_micro']:.4f}")

    # 打印每个机制类别的指标
    print("\nPer Mechanism Class Metrics:")
    mechanism_preds = np.array(all_preds["mechanism"])
    mechanism_labels_arr = np.array(all_labels["mechanism"])

    for i, class_name in enumerate(mechanism_labels):
        class_preds = mechanism_preds[:, i]
        class_labels = mechanism_labels_arr[:, i]
        class_metrics = compute_metrics(
            class_preds,
            class_labels,
            task_type="binary"
        )
        print(f"  {class_name}:")
        print(f"    AUC: {class_metrics['roc_auc']:.4f}")
        print(f"    Precision: {class_metrics['precision']:.4f}")
        print(f"    Recall: {class_metrics['recall']:.4f}")
        print(f"    F1: {class_metrics['f1']:.4f}")

    print("\n=== Antibiotic (Multilabel) ===")
    print(f"Macro AUC: {antibiotic_metrics['roc_auc_macro']:.4f}")
    print(f"Macro Precision: {antibiotic_metrics['precision_macro']:.4f}")
    print(f"Macro Recall: {antibiotic_metrics['recall_macro']:.4f}")
    print(f"Macro F1: {antibiotic_metrics['f1_macro']:.4f}")
    print(f"Micro Precision: {antibiotic_metrics['precision_micro']:.4f}")
    print(f"Micro Recall: {antibiotic_metrics['recall_micro']:.4f}")
    print(f"Micro F1: {antibiotic_metrics['f1_micro']:.4f}")

    # 打印每个抗生素类别的指标
    print("\nPer Antibiotic Class Metrics:")
    antibiotic_preds = np.array(all_preds["antibiotic"])
    antibiotic_labels_arr = np.array(all_labels["antibiotic"])

    for i, class_name in enumerate(antibiotic_labels):
        class_preds = antibiotic_preds[:, i]
        class_labels = antibiotic_labels_arr[:, i]
        class_metrics = compute_metrics(
            class_preds,
            class_labels,
            task_type="binary"
        )
        print(f"  {class_name}:")
        print(f"    AUC: {class_metrics['roc_auc']:.4f}")
        print(f"    Precision: {class_metrics['precision']:.4f}")
        print(f"    Recall: {class_metrics['recall']:.4f}")
        print(f"    F1: {class_metrics['f1']:.4f}")

    print("\n=== Remove (Binary) ===")
    print(f"AUC: {remove_metrics['roc_auc']:.4f}")
    print(f"Precision: {remove_metrics['precision']:.4f}")
    print(f"Recall: {remove_metrics['recall']:.4f}")
    print(f"F1: {remove_metrics['f1']:.4f}")

    # 打印正负样本指标
    print("\nRemove Class Metrics:")
    print("  Positive Class (需要移除):")
    print(f"    Precision: {pos_remove_metrics['precision']:.4f}")
    print(f"    Recall: {pos_remove_metrics['recall']:.4f}")
    print(f"    F1: {pos_remove_metrics['f1']:.4f}")
    print(f"    Support: {pos_remove_metrics['support']}")

    print("  Negative Class (不需要移除):")
    print(f"    Precision: {neg_remove_metrics['precision']:.4f}")
    print(f"    Recall: {neg_remove_metrics['recall']:.4f}")
    print(f"    F1: {neg_remove_metrics['f1']:.4f}")
    print(f"    Support: {neg_remove_metrics['support']}")


if __name__ == "__main__":
    test()