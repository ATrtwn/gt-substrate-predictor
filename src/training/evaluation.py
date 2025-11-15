# Metrics: Accuracy, ROC‐AUC, F1‐score, MCC
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, matthews_corrcoef

def compute_accuracy(y_true, y_pred):
    """
    Compute classification accuracy
    """
    return accuracy_score(y_true, y_pred)


def compute_roc_auc(y_true, y_probs):
    """
    Compute ROC-AUC for binary classification
    """
    return roc_auc_score(y_true, y_probs)


def compute_f1_score(y_true, y_pred, average='binary'):
    """
    Compute F1-score
    """
    return f1_score(y_true, y_pred, average)


def compute_mcc(y_true, y_pred):
    """
    Compute Matthews Correlation Coefficient (MCC), robust to class imbalance
    """
    return matthews_corrcoef(y_true, y_pred)