# Metrics: Accuracy, ROC‐AUC, F1‐score, MCC

def compute_accuracy(y_true, y_pred):
    """
    Compute classification accuracy
    """
    pass


def compute_roc_auc(y_true, y_probs):
    """
    Compute ROC-AUC for binary classification
    """
    pass


def compute_f1_score(y_true, y_pred, average='binary'):
    """
    Compute F1-score
    """
    pass


def compute_mcc(y_true, y_pred):
    """
    Compute Matthews Correlation Coefficient (MCC), robust to class imbalance
    """
    pass


def evaluate_split(y_true, y_pred, y_probs=None):
    """
    Evaluate a single dataset split using multiple metrics
    """
    # TODO: call compute_accuracy, compute_f1_score, compute_mcc
    # TODO: if y_probs is provided, compute ROC-AUC
    pass