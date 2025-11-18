import sys
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from noESM import GCM_MultiLabelModel
from transformers import AutoConfig
from utils_1 import compute_metrics, compute_class_metrics
import numpy as np
from sklearn.metrics import accuracy_score  # 导入准确率计算函数

sys.path.append('data')
from labels import mechanism_labels, antibiotic_labels
from Dataset import ProteinDataset

BASE_OUTPUT_DIR = "outputs/noESM_outputs_2"


def test():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # 创建结果目录
    os.makedirs(os.path.join(BASE_OUTPUT_DIR, "test_results"), exist_ok=True)

    # 加载测试数据
    test_dataset = torch.load("processed_data/test_dataset.pt")
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # 加载最佳模型
    config = AutoConfig.from_pretrained(os.path.join(BASE_OUTPUT_DIR, "best_model"))
    model = GCM_MultiLabelModel.from_pretrained(
        os.path.join(BASE_OUTPUT_DIR, "best_model"),
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

    # ===== 准备保存结果的列表 =====
    overall_results = []
    class_results = []

    # 计算指标
    resistance_metrics = compute_metrics(
        np.array(all_preds["resistance"]),
        np.array(all_labels["resistance"]),
        task_type="binary"
    )

    # 计算Resistance准确率
    resistance_preds_binary = (np.array(all_preds["resistance"]) >= 0.5).astype(int)
    resistance_acc = accuracy_score(np.array(all_labels["resistance"]), resistance_preds_binary)

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

    # 保存Resistance整体结果
    overall_results.append({
        "Task": "Resistance",
        "Metric": "AUC",
        "Value": resistance_metrics['roc_auc']
    })
    overall_results.append({
        "Task": "Resistance",
        "Metric": "Precision",
        "Value": resistance_metrics['precision']
    })
    overall_results.append({
        "Task": "Resistance",
        "Metric": "Recall",
        "Value": resistance_metrics['recall']
    })
    overall_results.append({
        "Task": "Resistance",
        "Metric": "F1",
        "Value": resistance_metrics['f1']
    })
    # 添加准确率指标
    overall_results.append({
        "Task": "Resistance",
        "Metric": "Accuracy",
        "Value": resistance_acc
    })

    # 保存Resistance正类结果
    overall_results.append({
        "Task": "Resistance_Positive",
        "Metric": "Precision",
        "Value": pos_resistance_metrics['precision']
    })
    overall_results.append({
        "Task": "Resistance_Positive",
        "Metric": "Recall",
        "Value": pos_resistance_metrics['recall']
    })
    overall_results.append({
        "Task": "Resistance_Positive",
        "Metric": "F1",
        "Value": pos_resistance_metrics['f1']
    })
    # 添加正类准确率（准确率是整体指标，这里添加正类的支持度即可）
    overall_results.append({
        "Task": "Resistance_Positive",
        "Metric": "Support",
        "Value": pos_resistance_metrics['support']
    })

    # 保存Resistance负类结果
    overall_results.append({
        "Task": "Resistance_Negative",
        "Metric": "Precision",
        "Value": neg_resistance_metrics['precision']
    })
    overall_results.append({
        "Task": "Resistance_Negative",
        "Metric": "Recall",
        "Value": neg_resistance_metrics['recall']
    })
    overall_results.append({
        "Task": "Resistance_Negative",
        "Metric": "F1",
        "Value": neg_resistance_metrics['f1']
    })
    # 添加负类支持度
    overall_results.append({
        "Task": "Resistance_Negative",
        "Metric": "Support",
        "Value": neg_resistance_metrics['support']
    })

    # 计算机制指标
    mechanism_metrics = compute_metrics(
        np.array(all_preds["mechanism"]),
        np.array(all_labels["mechanism"]),
        task_type="multilabel"
    )

    # 计算Mechanism准确率（子集准确率）
    mechanism_preds_binary = (np.array(all_preds["mechanism"]) >= 0.5).astype(int)
    mechanism_acc = accuracy_score(np.array(all_labels["mechanism"]), mechanism_preds_binary)

    # 保存Mechanism整体结果
    overall_results.append({
        "Task": "Mechanism",
        "Metric": "Macro_AUC",
        "Value": mechanism_metrics['roc_auc_macro']
    })
    overall_results.append({
        "Task": "Mechanism",
        "Metric": "Macro_Precision",
        "Value": mechanism_metrics['precision_macro']
    })
    overall_results.append({
        "Task": "Mechanism",
        "Metric": "Macro_Recall",
        "Value": mechanism_metrics['recall_macro']
    })
    overall_results.append({
        "Task": "Mechanism",
        "Metric": "Macro_F1",
        "Value": mechanism_metrics['f1_macro']
    })
    overall_results.append({
        "Task": "Mechanism",
        "Metric": "Micro_Precision",
        "Value": mechanism_metrics['precision_micro']
    })
    overall_results.append({
        "Task": "Mechanism",
        "Metric": "Micro_Recall",
        "Value": mechanism_metrics['recall_micro']
    })
    overall_results.append({
        "Task": "Mechanism",
        "Metric": "Micro_F1",
        "Value": mechanism_metrics['f1_micro']
    })
    # 添加准确率指标
    overall_results.append({
        "Task": "Mechanism",
        "Metric": "Accuracy",
        "Value": mechanism_acc
    })

    # 计算每个机制类别的指标
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

        # 计算类别准确率
        class_preds_binary = (class_preds >= 0.5).astype(int)
        class_acc = accuracy_score(class_labels, class_preds_binary)

        # 保存每个机制类别的结果
        class_results.append({
            "Task": "Mechanism",
            "Class": class_name,
            "Metric": "AUC",
            "Value": class_metrics['roc_auc']
        })
        class_results.append({
            "Task": "Mechanism",
            "Class": class_name,
            "Metric": "Precision",
            "Value": class_metrics['precision']
        })
        class_results.append({
            "Task": "Mechanism",
            "Class": class_name,
            "Metric": "Recall",
            "Value": class_metrics['recall']
        })
        class_results.append({
            "Task": "Mechanism",
            "Class": class_name,
            "Metric": "F1",
            "Value": class_metrics['f1']
        })
        # 添加准确率指标
        class_results.append({
            "Task": "Mechanism",
            "Class": class_name,
            "Metric": "Accuracy",
            "Value": class_acc
        })

    # 计算抗生素指标
    antibiotic_metrics = compute_metrics(
        np.array(all_preds["antibiotic"]),
        np.array(all_labels["antibiotic"]),
        task_type="multilabel"
    )

    # 计算Antibiotic准确率（子集准确率）
    antibiotic_preds_binary = (np.array(all_preds["antibiotic"]) >= 0.5).astype(int)
    antibiotic_acc = accuracy_score(np.array(all_labels["antibiotic"]), antibiotic_preds_binary)

    # 保存Antibiotic整体结果
    overall_results.append({
        "Task": "Antibiotic",
        "Metric": "Macro_AUC",
        "Value": antibiotic_metrics['roc_auc_macro']
    })
    overall_results.append({
        "Task": "Antibiotic",
        "Metric": "Macro_Precision",
        "Value": antibiotic_metrics['precision_macro']
    })
    overall_results.append({
        "Task": "Antibiotic",
        "Metric": "Macro_Recall",
        "Value": antibiotic_metrics['recall_macro']
    })
    overall_results.append({
        "Task": "Antibiotic",
        "Metric": "Macro_F1",
        "Value": antibiotic_metrics['f1_macro']
    })
    overall_results.append({
        "Task": "Antibiotic",
        "Metric": "Micro_Precision",
        "Value": antibiotic_metrics['precision_micro']
    })
    overall_results.append({
        "Task": "Antibiotic",
        "Metric": "Micro_Recall",
        "Value": antibiotic_metrics['recall_micro']
    })
    overall_results.append({
        "Task": "Antibiotic",
        "Metric": "Micro_F1",
        "Value": antibiotic_metrics['f1_micro']
    })
    # 添加准确率指标
    overall_results.append({
        "Task": "Antibiotic",
        "Metric": "Accuracy",
        "Value": antibiotic_acc
    })

    # 计算每个抗生素类别的指标
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

        # 计算类别准确率
        class_preds_binary = (class_preds >= 0.5).astype(int)
        class_acc = accuracy_score(class_labels, class_preds_binary)

        # 保存每个抗生素类别的结果
        class_results.append({
            "Task": "Antibiotic",
            "Class": class_name,
            "Metric": "AUC",
            "Value": class_metrics['roc_auc']
        })
        class_results.append({
            "Task": "Antibiotic",
            "Class": class_name,
            "Metric": "Precision",
            "Value": class_metrics['precision']
        })
        class_results.append({
            "Task": "Antibiotic",
            "Class": class_name,
            "Metric": "Recall",
            "Value": class_metrics['recall']
        })
        class_results.append({
            "Task": "Antibiotic",
            "Class": class_name,
            "Metric": "F1",
            "Value": class_metrics['f1']
        })
        # 添加准确率指标
        class_results.append({
            "Task": "Antibiotic",
            "Class": class_name,
            "Metric": "Accuracy",
            "Value": class_acc
        })

    # 计算移除指标
    remove_metrics = compute_metrics(
        np.array(all_preds["remove"]),
        np.array(all_labels["remove"]),
        task_type="binary"
    )

    # 计算Remove准确率
    remove_preds_binary = (np.array(all_preds["remove"]) >= 0.5).astype(int)
    remove_acc = accuracy_score(np.array(all_labels["remove"]), remove_preds_binary)

    # 保存Remove整体结果
    overall_results.append({
        "Task": "Remove",
        "Metric": "AUC",
        "Value": remove_metrics['roc_auc']
    })
    overall_results.append({
        "Task": "Remove",
        "Metric": "Precision",
        "Value": remove_metrics['precision']
    })
    overall_results.append({
        "Task": "Remove",
        "Metric": "Recall",
        "Value": remove_metrics['recall']
    })
    overall_results.append({
        "Task": "Remove",
        "Metric": "F1",
        "Value": remove_metrics['f1']
    })
    # 添加准确率指标
    overall_results.append({
        "Task": "Remove",
        "Metric": "Accuracy",
        "Value": remove_acc
    })

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

    # 保存Remove正类结果
    overall_results.append({
        "Task": "Remove_Positive",
        "Metric": "Precision",
        "Value": pos_remove_metrics['precision']
    })
    overall_results.append({
        "Task": "Remove_Positive",
        "Metric": "Recall",
        "Value": pos_remove_metrics['recall']
    })
    overall_results.append({
        "Task": "Remove_Positive",
        "Metric": "F1",
        "Value": pos_remove_metrics['f1']
    })
    overall_results.append({
        "Task": "Remove_Positive",
        "Metric": "Support",
        "Value": pos_remove_metrics['support']
    })

    # 保存Remove负类结果
    overall_results.append({
        "Task": "Remove_Negative",
        "Metric": "Precision",
        "Value": neg_remove_metrics['precision']
    })
    overall_results.append({
        "Task": "Remove_Negative",
        "Metric": "Recall",
        "Value": neg_remove_metrics['recall']
    })
    overall_results.append({
        "Task": "Remove_Negative",
        "Metric": "F1",
        "Value": neg_remove_metrics['f1']
    })
    overall_results.append({
        "Task": "Remove_Negative",
        "Metric": "Support",
        "Value": neg_remove_metrics['support']
    })

    # ===== 保存结果到CSV文件 =====
    # 创建整体指标DataFrame
    overall_df = pd.DataFrame(overall_results)

    overall_csv_path = os.path.join(BASE_OUTPUT_DIR, "test_results", "overall_metrics.csv")
    overall_df.to_csv(overall_csv_path, index=False)
    print(f"保存整体指标到: {overall_csv_path}")

    # 创建类别指标DataFrame
    class_df = pd.DataFrame(class_results)

    class_csv_path = os.path.join(BASE_OUTPUT_DIR, "test_results", "class_metrics.csv")
    class_df.to_csv(class_csv_path, index=False)
    print(f"保存类别指标到: {class_csv_path}")

    # 打印结果摘要
    print("\n=== 测试结果摘要 ===")
    print(f"整体指标已保存至: {overall_csv_path}")
    print(f"类别详细指标已保存至: {class_csv_path}")


if __name__ == "__main__":
    test()