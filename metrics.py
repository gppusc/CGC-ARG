from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

def compute_metrics(y_true, y_pred, average="macro"):
    return {
        "precision": precision_score(y_true, y_pred, average=average),
        "recall": recall_score(y_true, y_pred, average=average),
        "f1": f1_score(y_true, y_pred, average=average),
    }