"""
src — Spam/Ham classifier package.

Public API
----------
from src import NaiveBayes, LogisticRegression
from src.preprocessing import load_data, clean_text, tokenize, ...
from src.evaluation import precision_recall_f1, roc_curve, auc, ...
"""

from .naive_bayes import NaiveBayes
from .logistic_regression import LogisticRegression
from .preprocessing import (
    load_data,
    clean_text,
    tokenize,
    build_vocabulary,
    vectorize_bow,
    fit_tfidf,
    transform_tfidf,
    train_test_split,
)
from .evaluation import (
    confusion_matrix,
    precision_recall_f1,
    accuracy_score,
    roc_curve,
    auc,
    best_threshold,
    plot_roc,
    plot_loss_curve,
    print_report,
    print_confusion_matrix,
)

__all__ = [
    # Models
    "NaiveBayes",
    "LogisticRegression",
    # Preprocessing
    "load_data",
    "clean_text",
    "tokenize",
    "build_vocabulary",
    "vectorize_bow",
    "fit_tfidf",
    "transform_tfidf",
    "train_test_split",
    # Evaluation
    "confusion_matrix",
    "precision_recall_f1",
    "accuracy_score",
    "roc_curve",
    "auc",
    "best_threshold",
    "plot_roc",
    "plot_loss_curve",
    "print_report",
    "print_confusion_matrix",
]
