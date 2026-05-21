"""
evaluation.py — All metrics and plots. Only numpy + matplotlib. No sklearn metrics.
"""

def confusion_matrix(y_true, y_pred):
    """
    Build a 2×2 confusion matrix from scratch.
    - Count TP, FP, TN, FN manually
    - Return dict or 2D numpy array [[TN, FP], [FN, TP]]
    """
    pass

def precision_recall_f1(y_true, y_pred):
    """
    Compute precision, recall, and F1 from the confusion matrix.
    - Precision = TP / (TP + FP)
    - Recall    = TP / (TP + FN)
    - F1        = 2 * P * R / (P + R)
    - Handle division by zero gracefully
    - Return (precision, recall, f1) as floats
    """
    pass

def roc_curve(y_true, y_scores):
    """
    Compute the full ROC curve by sweeping thresholds.
    - Sort unique score values descending as candidate thresholds
    - At each threshold: compute TPR = TP/(TP+FN) and FPR = FP/(FP+TN)
    - Return (fpr_list, tpr_list, thresholds)
    """
    pass

def auc(fpr, tpr):
    """
    Area under the ROC curve via the trapezoidal rule.
    - AUC = sum of trapezoid areas between consecutive (fpr, tpr) points
    - Return scalar float in [0, 1]
    """
    pass

def best_threshold(y_true, y_scores, metric='f1'):
    """
    Find the decision threshold that maximises the chosen metric.
    - Sweep thresholds, compute metric at each step
    - metric options: 'f1'  or  'youden'  (Youden's J = TPR - FPR)
    - Return (best_threshold, best_score)
    """
    pass

def plot_roc(fpr_nb, tpr_nb, auc_nb, fpr_lr, tpr_lr, auc_lr, save_path):
    """
    Plot both ROC curves on one axes.
    - NB curve in one colour, LR curve in another
    - Label each with its AUC in the legend
    - Add the random-classifier diagonal (dashed grey)
    - Save figure to save_path
    """
    pass

def plot_loss_curve(loss_history, save_path):
    """
    Plot the logistic regression training loss over iterations.
    - x-axis: iteration,  y-axis: binary cross-entropy
    - Useful for spotting if learning rate is too high or too low
    - Save figure to save_path
    """
    pass

def print_report(model_name, precision, recall, f1, auc_score):
    """
    Pretty-print a one-line summary for a model.
    - Format: "ModelName  |  P: 0.xx  R: 0.xx  F1: 0.xx  AUC: 0.xx"
    """
    pass