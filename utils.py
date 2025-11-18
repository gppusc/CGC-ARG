from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import numpy as np


import numpy as np
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score
)


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
def compute_metrics(preds, labels, task_type):
    """
    preds: np.ndarray, floats (最好是概率)
    labels: np.ndarray, int {0,1}

    task_type: "binary" or "multilabel"
    """
    metrics = {}
    # 先把 preds 里的 NaN／+inf／-inf 全部替换成 [0,1] 内的数
    preds = np.nan_to_num(preds, nan=0.0, posinf=1.0, neginf=0.0)

    if task_type == "binary":
        y_true = labels.flatten()
        y_score = preds.flatten()
        # 如果 y_true 全 0 或全 1，则跳过 AUC，赋 0.5（随机猜的 baseline）
        if np.unique(y_true).size < 2:
            auc = 0.5
        else:
            auc = roc_auc_score(y_true, y_score)
        y_pred = (y_score > 0.5).astype(int)

        metrics["roc_auc"]    = auc
        metrics["precision"]  = precision_score(y_true, y_pred, zero_division=0)
        metrics["recall"]     = recall_score(y_true, y_pred, zero_division=0)
        metrics["f1"]         = f1_score(y_true, y_pred, zero_division=0)

    elif task_type == "multilabel":
        # per-class
        auc_list, p_list, r_list, f_list = [], [], [], []
        num_classes = labels.shape[1]

        for i in range(num_classes):
            y_true = labels[:, i]
            y_score = preds[:, i]
            # 忽略全0或全1的类别
            if np.unique(y_true).size < 2:
                continue

            # 再次确保 y_score 中无非法值
            y_score = np.nan_to_num(y_score, nan=0.0, posinf=1.0, neginf=0.0)
            y_pred = (y_score > 0.5).astype(int)

            try:
                auc = roc_auc_score(y_true, y_score)
            except ValueError:
                # 万一还是报错，就跳过
                continue

            auc_list.append(auc)
            p_list.append(precision_score(y_true, y_pred, zero_division=0))
            r_list.append(recall_score   (y_true, y_pred, zero_division=0))
            f_list.append(f1_score       (y_true, y_pred, zero_division=0))

        # Macro 版
        metrics["roc_auc_macro"]    = float(np.mean(auc_list)) if auc_list else 0.0
        metrics["precision_macro"]  = float(np.mean(p_list))   if p_list   else 0.0
        metrics["recall_macro"]     = float(np.mean(r_list))   if r_list   else 0.0
        metrics["f1_macro"]         = float(np.mean(f_list))   if f_list   else 0.0

        # Micro 版，直接把所有类别拼到一起算
        y_pred_all = (preds > 0.5).astype(int)
        try:
            metrics["precision_micro"] = precision_score(labels, y_pred_all,
                                                        average='micro',
                                                        zero_division=0)
            metrics["recall_micro"]    = recall_score   (labels, y_pred_all,
                                                        average='micro',
                                                        zero_division=0)
            metrics["f1_micro"]        = f1_score       (labels, y_pred_all,
                                                        average='micro',
                                                        zero_division=0)
        except ValueError:
            # 如样本极端，直接给 0
            metrics["precision_micro"] = 0.0
            metrics["recall_micro"]    = 0.0
            metrics["f1_micro"]        = 0.0

    else:
        raise ValueError(f"Unknown task_type '{task_type}'")

    return metrics

def search_best_thresholds(probs, truths, num_classes, grid=None):
    """
    probs: (N, A) 预测概率数组
    truths: (N, A) 真实标签(0/1)
    返回 thresholds: 长度 A 的最佳阈值列表
    """
    if grid is None:
        grid = np.linspace(0.0, 1.0, 101)

    best_thresh = np.zeros(num_classes)
    for c in range(num_classes):
        y_true = truths[:, c]
        y_prob = probs[:, c]
        best_f1, best_t = 0, 0.5
        for t in grid:
            y_pred = (y_prob >= t).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        best_thresh[c] = best_t
    return best_thresh


