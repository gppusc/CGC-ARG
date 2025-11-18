from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import numpy as np


def compute_class_metrics(preds, labels, target_class=1, threshold=0.5):
    """
    计算指定类别的二分类指标

    参数:
        preds: 预测概率 (numpy数组)
        labels: 真实标签 (numpy数组)
        target_class: 目标类别 (0或1)
        threshold: 分类阈值

    返回:
        metrics: 包含精确率、召回率、F1和支持数的字典
    """
    # 将预测概率转换为二分类标签
    pred_labels = (preds >= threshold).astype(int)

    # 获取目标类别的指标
    true_mask = (labels == target_class)

    # 计算真正例、假正例、假反例
    tp = np.sum((pred_labels == target_class) & (labels == target_class))
    fp = np.sum((pred_labels == target_class) & (labels != target_class))
    fn = np.sum((pred_labels != target_class) & (labels == target_class))

    # 计算精确率
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # 计算召回率
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # 计算F1分数
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # 支持数
    support = np.sum(true_mask)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support
    }
def compute_metrics(preds, labels, task_type, rare_indices=None):
    """
    preds: np.ndarray, floats (概率值)
    labels: np.ndarray, int {0,1}
    task_type: "binary" or "multilabel"
    rare_indices: list, 稀有类别的索引列表
    """
    metrics = {}
    # 处理 NaN/Inf 值
    preds = np.nan_to_num(preds, nan=0.0, posinf=1.0, neginf=0.0)

    # 稀有类别指标
    rare_metrics = {"roc_auc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    rare_count = 0

    if task_type == "binary":
        y_true = labels.flatten()
        y_score = preds.flatten()

        # 处理单一类别情况
        if np.unique(y_true).size < 2:
            auc = 0.5
            y_pred = np.zeros_like(y_true)
        else:
            auc = roc_auc_score(y_true, y_score)
            y_pred = (y_score > 0.5).astype(int)

        metrics["roc_auc"] = auc
        metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
        metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)

    elif task_type == "multilabel":
        num_classes = labels.shape[1]

        # 全类别指标
        auc_list, p_list, r_list, f_list = [], [], [], []

        # 稀有类别指标
        rare_auc_list, rare_p_list, rare_r_list, rare_f_list = [], [], [], []

        for i in range(num_classes):
            y_true = labels[:, i]
            y_score = preds[:, i]

            # 跳过全0或全1的类别
            if np.unique(y_true).size < 2:
                continue

            # 确保无非法值
            y_score = np.nan_to_num(y_score, nan=0.0, posinf=1.0, neginf=0.0)
            y_pred = (y_score > 0.5).astype(int)

            try:
                auc = roc_auc_score(y_true, y_score)
            except ValueError:
                continue

            # 计算指标
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            # 添加到全类别列表
            auc_list.append(auc)
            p_list.append(precision)
            r_list.append(recall)
            f_list.append(f1)

            # 如果是稀有类别，添加到稀有类别列表
            if rare_indices and i in rare_indices:
                rare_auc_list.append(auc)
                rare_p_list.append(precision)
                rare_r_list.append(recall)
                rare_f_list.append(f1)
                rare_count += 1

        # 全类别宏平均
        metrics["roc_auc_macro"] = np.mean(auc_list) if auc_list else 0.0
        metrics["precision_macro"] = np.mean(p_list) if p_list else 0.0
        metrics["recall_macro"] = np.mean(r_list) if r_list else 0.0
        metrics["f1_macro"] = np.mean(f_list) if f_list else 0.0

        # 稀有类别宏平均
        if rare_count > 0:
            rare_metrics["roc_auc"] = np.mean(rare_auc_list)
            rare_metrics["precision"] = np.mean(rare_p_list)
            rare_metrics["recall"] = np.mean(rare_r_list)
            rare_metrics["f1"] = np.mean(rare_f_list)

        # Micro指标
        y_pred_all = (preds > 0.5).astype(int)
        try:
            metrics["precision_micro"] = precision_score(labels, y_pred_all, average='micro', zero_division=0)
            metrics["recall_micro"] = recall_score(labels, y_pred_all, average='micro', zero_division=0)
            metrics["f1_micro"] = f1_score(labels, y_pred_all, average='micro', zero_division=0)
        except:
            metrics["precision_micro"] = 0.0
            metrics["recall_micro"] = 0.0
            metrics["f1_micro"] = 0.0

        # 添加稀有类别指标
        metrics["rare_roc_auc"] = rare_metrics["roc_auc"]
        metrics["rare_precision"] = rare_metrics["precision"]
        metrics["rare_recall"] = rare_metrics["recall"]
        metrics["rare_f1"] = rare_metrics["f1"]
        metrics["rare_count"] = rare_count

    else:
        raise ValueError(f"Unknown task_type '{task_type}'")

    return metrics