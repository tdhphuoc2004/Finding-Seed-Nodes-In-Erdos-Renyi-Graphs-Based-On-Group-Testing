"""
Metric extraction utilities for seed node identification experiments.
"""

from sklearn.metrics import classification_report


def extract_metrics(y_true, y_pred):
    """
    Extracts Precision, Recall, and F1-Score for the seed class (label=1).

    Args:
        y_true: Ground-truth binary label array.
        y_pred: Predicted binary label array.

    Returns:
        dict with keys 'Precision', 'Recall', 'F1-Score'.
    """
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    seed_key = '1' if '1' in report else '1.0'
    if seed_key not in report:
        seed_key = list(report.keys())[1]

    return {
        'Precision': report[seed_key]['precision'],
        'Recall': report[seed_key]['recall'],
        'F1-Score': report[seed_key]['f1-score']
    }
