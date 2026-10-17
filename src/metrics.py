"""Per-class and summary metrics for the 3-class problem."""
from sklearn.metrics import classification_report, confusion_matrix, f1_score, matthews_corrcoef

def report(y_true, y_pred, labels=("lethal", "viable", "other")):
    return {
        "confusion": confusion_matrix(y_true, y_pred, labels=labels),
        "per_class": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
