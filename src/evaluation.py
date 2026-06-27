"""
evaluation.py — All metrics and plots. Only numpy + matplotlib. No sklearn metrics.
"""
import numpy as np
import matplotlib.pyplot as plt
import pylab as p


def confusion_matrix(y_true, y_pred):
    """
    Build a 2×2 confusion matrix from scratch.
    - Count TP, FP, TN, FN manually
    - Return dict or 2D numpy array [[TN, FP], [FN, TP]]
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = ((y_true == 1) & (y_pred == 1)).sum()          # both prediction and actual is 1
    fp = ((y_true == 0) & (y_pred == 1)).sum()          # the prediction giving 1 but actual is 0
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    return [[tn, fp], [fn, tp]]


def precision_recall_f1(y_true, y_pred):
    """
    Compute precision, recall, and F1 from the confusion matrix.
    - Precision = TP / (TP + FP)
    - Recall    = TP / (TP + FN)
    - F1        = 2 * P * R / (P + R)
    - Handle division by zero gracefully
    - Return (precision, recall, f1) as floats
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1

def roc_curve(y_true, y_scores):
    """
    Compute the full ROC curve by sweeping thresholds.
    - Sort unique score values descending as candidate thresholds
    - At each threshold: compute TPR = TP/(TP+FN) and FPR = FP/(FP+TN)
    - Return (fpr_list, tpr_list, thresholds)
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    P = (y_true == 1).sum()
    N = (y_true == 0).sum()

    # Candidate thresholds: unique scores desc + a sentinel above the max
    thresholds = np.concatenate(([y_scores.max() + 1.0], np.sort(np.unique(y_scores))[::-1]))    # np.concatenate(([sentinel], rest))
    '''
    Sentinel goes first in the threshold list
    At sentinel, nothing gets predicted positive → TP=0, FP=0 → (0, 0)
    Then real thresholds follow → curve builds up point by point
    Together they form the complete ROC curve starting from origin
    '''
    fpr_list, tpr_list = [], []
    for thres in thresholds:
        '''
        every y_score will give us a point as tpr,fpr
        '''
        y_pred = (y_scores >= thres).astype(int)
        tp = np.sum((y_pred == 1) &(y_true == 1))
        fp = np.sum((y_pred == 1) &(y_true == 0))

        tpr = tp / P if P > 0 else 0.0
        fpr = fp / N if N > 0 else 0.0

        tpr_list.append(tpr)
        fpr_list.append((fpr))

    return np.array(fpr_list), np.array(tpr_list), thresholds

def auc(fpr, tpr):
    """
    Area under the ROC curve via the trapezoidal rule.
    - AUC = sum of trapezoid areas between consecutive (fpr, tpr) points
    - Return scalar float in [0, 1]
    """
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)

    order = np.argsort(fpr)  # gives indices that would sort fpr
    fpr = fpr[order]  # reorder fpr
    tpr = tpr[order]  # reorder tpr the same way
    # it will keep point pairs(fpr, tpr) intact

    auc_val = float(np.trapz(tpr, fpr))
    return auc_val

def best_threshold(y_true, y_scores, metric='f1'):
    """
    Find the decision threshold that maximises the chosen metric.
    - Sweep thresholds, compute metric at each step
    - metric options: 'f1'  or  'youden'  (Youden's J = TPR - FPR)
    - Return (best_threshold, best_score)
    """
    '''
    Get all unique thresholds from y_scores
    Loop over each threshold
    Generate y_pred at that threshold
    Compute TP, FP, FN
    Compute the chosen metric
    If it's better than the best so far, update best
    '''
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    if metric not in ('f1', 'youden'):
        raise ValueError(f"metric must be 'f1' or 'youden', got '{metric}'")

    thresholds = np.sort(np.unique(y_scores))
    best_thresh, best_score = thresholds[0], -np.inf

    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)

    for thres in thresholds:
        y_pred = (y_scores >= thres).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        if metric == "f1":
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        else: # if youden
            tpr = tp / P if P > 0 else 0.0
            fpr = fp / N if N > 0 else 0.0
            score = tpr - fpr

        if score > best_score:
            best_score = score
            best_thresh = thres
    return float(best_thresh), float(best_score)

def plot_roc(fpr_nb, tpr_nb, auc_nb, fpr_lr, tpr_lr, auc_lr, save_path):
    """
    Plot both ROC curves on one axes.
    - NB curve in one colour, LR curve in another
    - Label each with its AUC in the legend
    - Add the random-classifier diagonal (dashed grey)
    - Save figure to save_path
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(fpr_nb, tpr_nb, color='steelblue', lw=2,
            label=f'Naive Bayes  (AUC = {auc_nb:.3f})')
    ax.plot(fpr_lr, tpr_lr, color='tomato', lw=2,
            label=f'Logistic Reg (AUC = {auc_lr:.3f})')
    ax.plot([0, 1], [0, 1], color='grey', lw=1,
            linestyle='--', label='Random classifier')

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve Comparison', fontsize=13)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[plot_roc] saved → {save_path}")


def plot_loss_curve(loss_history, save_path):
    """
    Plot the logistic regression training loss over iterations.
    - x-axis: iteration,  y-axis: binary cross-entropy
    - Useful for spotting if learning rate is too high or too low
    - Save figure to save_path
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    iterations = np.arange(1, len(loss_history) + 1)
    ax.plot(iterations, loss_history, color='darkorange', lw=2)

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Binary Cross-Entropy Loss', fontsize=12)
    ax.set_title('Logistic Regression — Training Loss', fontsize=13)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[plot_loss_curve] saved → {save_path}")

def print_report(model_name, precision, recall, f1, auc_score):
    """
    Pretty-print a one-line summary for a model.
    - Format: "ModelName  |  P: 0.xx  R: 0.xx  F1: 0.xx  AUC: 0.xx"
    """
    print(f"{model_name:<20s}  |  "
          f"P: {precision:.4f}  "
          f"R: {recall:.4f}  "
          f"F1: {f1:.4f}  "
          f"AUC: {auc_score:.4f}")