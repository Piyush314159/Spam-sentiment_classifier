"""
main.py — Runs the full pipeline end-to-end.
Execute with:  python main.py

Steps
-----
1. Load & preprocess
      - load_data()  →  clean_text()  →  tokenize()
      - build_vocabulary() on TRAIN split only  (prevents data leakage)
      - vectorize_bow()   for Naive Bayes  (needs raw counts)
      - vectorize_tfidf() for Logistic Regression  (normalised weights)

2. Naive Bayes
      - NaiveBayes(alpha=1.0).fit(X_train_bow, y_train)
      - predict_proba() on test set
      - best_threshold() → predict() → precision_recall_f1()
      - roc_curve() + auc()

3. Logistic Regression
      - LogisticRegression(lr=0.1, n_iters=500).fit(X_train_tfidf, y_train)
      - Same evaluation flow as above
      - Save loss_history for the loss curve plot

4. Compare
      - print_report() for both models side by side
      - Note which wins on F1 vs AUC and think about why

5. Save outputs
      - plot_roc(...)        →  results/roc_curve.png
      - plot_loss_curve(...) →  results/loss_curve.png
      - Optionally dump metrics to results/metrics.json
"""

import os
import json
import numpy as np

from src.preprocessing import (
    load_data,
    clean_text,
    tokenize,
    build_vocabulary,
    vectorize_bow,
    vectorize_tfidf,
)
from src.naive_bayes import NaiveBayes
from src.logistic_regression import LogisticRegression
from src.evaluation import (
    precision_recall_f1,
    roc_curve,
    auc,
    best_threshold,
    plot_roc,
    plot_loss_curve,
    print_report,
)


# ── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join(os.path.dirname(__file__), "data", "SMSSpamCollection")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 1 — Load & preprocess
    # ═════════════════════════════════════════════════════════════════════════
    print("=" * 65)
    print("  STEP 1 — Load & preprocess")
    print("=" * 65)

    # 1a. Load raw data
    texts, labels = load_data(DATA_PATH)
    print(f"  Loaded {len(texts)} messages  "
          f"(ham: {labels.count(0)}, spam: {labels.count(1)})")

    # 1b. clean_text() → tokenize()
    cleaned   = [clean_text(t) for t in texts]
    tokenized = [tokenize(t) for t in cleaned]
    print(f"  Example (raw):     {texts[2][:80]}...")
    print(f"  Example (cleaned): {cleaned[2][:80]}")
    print(f"  Example (tokens):  {tokenized[2][:10]}")

    # 1c. Train / test split at the token level (80-20)
    labels_arr = np.array(labels)
    idx = np.arange(len(labels_arr))
    rng = np.random.default_rng(42)
    idx = rng.permutation(idx)
    split = int(len(idx) * 0.8)

    train_idx, test_idx = idx[:split], idx[split:]
    train_tokens = [tokenized[i] for i in train_idx]
    test_tokens  = [tokenized[i] for i in test_idx]
    y_train      = labels_arr[train_idx]
    y_test       = labels_arr[test_idx]
    print(f"  Train size: {len(train_tokens)}   Test size: {len(test_tokens)}")

    # 1d. build_vocabulary() on TRAIN split only (prevents data leakage)
    vocab = build_vocabulary(train_tokens)
    print(f"  Vocabulary size (train-only): {len(vocab)}")

    # 1e. vectorize_bow() for Naive Bayes (raw counts)
    X_train_bow = vectorize_bow(train_tokens, vocab)
    X_test_bow  = vectorize_bow(test_tokens, vocab)
    print(f"  BoW   — train: {X_train_bow.shape}   test: {X_test_bow.shape}")

    # 1f. vectorize_tfidf() for Logistic Regression (normalised weights)
    X_train_tfidf = vectorize_tfidf(X_train_bow)
    X_test_tfidf  = vectorize_tfidf(X_test_bow)
    print(f"  TF-IDF — train: {X_train_tfidf.shape}  test: {X_test_tfidf.shape}\n")

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 2 — Naive Bayes
    # ═════════════════════════════════════════════════════════════════════════
    print("=" * 65)
    print("  STEP 2 — Naive Bayes")
    print("=" * 65)

    # 2a. Fit on BoW counts
    nb = NaiveBayes(alpha=1.0)
    nb.fit(X_train_bow, y_train)
    print("  ✓ NaiveBayes(alpha=1.0).fit(X_train_bow, y_train)")

    # 2b. predict_proba() on test set
    nb_probs = nb.predict_proba(X_test_bow)

    # 2c. best_threshold() → predict() → precision_recall_f1()
    nb_best_t, nb_best_f1 = best_threshold(y_test, nb_probs, metric='f1')
    nb_preds = nb.predict(X_test_bow, threshold=nb_best_t)
    nb_p, nb_r, nb_f1 = precision_recall_f1(y_test, nb_preds)
    print(f"  Best threshold (F1): {nb_best_t:.4f}")
    print(f"  Precision: {nb_p:.4f}   Recall: {nb_r:.4f}   F1: {nb_f1:.4f}")

    # 2d. roc_curve() + auc()
    fpr_nb, tpr_nb, _ = roc_curve(y_test, nb_probs)
    auc_nb = auc(fpr_nb, tpr_nb)
    print(f"  AUC: {auc_nb:.4f}\n")

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 3 — Logistic Regression
    # ═════════════════════════════════════════════════════════════════════════
    print("=" * 65)
    print("  STEP 3 — Logistic Regression")
    print("=" * 65)

    # 3a. Fit on TF-IDF (with L2 regularisation + validation loss tracking)
    lr = LogisticRegression(lr=0.1, n_iters=2000, lambda_=0.01)
    lr.fit(X_train_tfidf, y_train, X_val=X_test_tfidf, y_val=y_test)
    print(f"  ✓ LogisticRegression(lr=0.1, n_iters=2000, lambda_=0.01)")
    print(f"    .fit(X_train_tfidf, y_train, X_val=X_test_tfidf, y_val=y_test)")
    print(f"  Final train loss: {lr.loss_history[-1]:.4f}")
    print(f"  Final test  loss: {lr.val_loss_history[-1]:.4f}")
    gap = lr.val_loss_history[-1] - lr.loss_history[-1]
    print(f"  Gap (test − train): {gap:+.4f}  "
          f"{'→ OK, no overfitting' if abs(gap) < 0.05 else '→ ⚠️  possible overfitting'}")

    # 3b. predict_proba() on test set
    lr_probs = lr.predict_proba(X_test_tfidf)

    # 3c. best_threshold() → predict() → precision_recall_f1()
    lr_best_t, lr_best_f1 = best_threshold(y_test, lr_probs, metric='f1')
    lr_preds = lr.predict(X_test_tfidf, threshold=lr_best_t)
    lr_p, lr_r, lr_f1 = precision_recall_f1(y_test, lr_preds)
    print(f"  Best threshold (F1): {lr_best_t:.4f}")
    print(f"  Precision: {lr_p:.4f}   Recall: {lr_r:.4f}   F1: {lr_f1:.4f}")

    # 3d. roc_curve() + auc()
    fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probs)
    auc_lr = auc(fpr_lr, tpr_lr)
    print(f"  AUC: {auc_lr:.4f}")

    # 3e. Save loss histories for the loss curve plot (used in Step 5)
    train_loss_history = lr.loss_history
    val_loss_history   = lr.val_loss_history
    print(f"  Loss histories saved ({len(train_loss_history)} iterations)\n")

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 4 — Compare
    # ═════════════════════════════════════════════════════════════════════════
    print("=" * 65)
    print("  STEP 4 — Compare (side by side)")
    print("=" * 65)

    print_report("Naive Bayes",          nb_p, nb_r, nb_f1, auc_nb)
    print_report("Logistic Regression",  lr_p, lr_r, lr_f1, auc_lr)

    # Note which wins on F1 vs AUC
    print()
    f1_winner  = "Naive Bayes" if nb_f1 > lr_f1 else "Logistic Regression"
    auc_winner = "Naive Bayes" if auc_nb > auc_lr else "Logistic Regression"
    print(f"  → F1 winner:  {f1_winner}  "
          f"(NB: {nb_f1:.4f}  vs  LR: {lr_f1:.4f})")
    print(f"  → AUC winner: {auc_winner}  "
          f"(NB: {auc_nb:.4f}  vs  LR: {auc_lr:.4f})")

    if f1_winner != auc_winner:
        print("  💡 Different winners — F1 rewards precision/recall balance at a")
        print("     single threshold, while AUC measures ranking quality across")
        print("     all thresholds. A model can rank well overall (high AUC)")
        print("     yet have a less optimal decision boundary at a specific cut-off.")
    print()

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 5 — Save outputs
    # ═════════════════════════════════════════════════════════════════════════
    print("=" * 65)
    print("  STEP 5 — Save outputs")
    print("=" * 65)

    # 5a. plot_roc(...) → results/roc_curve.png
    roc_path = os.path.join(RESULTS_DIR, "roc_curve.png")
    plot_roc(fpr_nb, tpr_nb, auc_nb, fpr_lr, tpr_lr, auc_lr, save_path=roc_path)

    # 5b. plot_loss_curve(...) → results/loss_curve.png (train + test)
    loss_path = os.path.join(RESULTS_DIR, "loss_curve.png")
    plot_loss_curve(train_loss_history, save_path=loss_path,
                    val_loss_history=val_loss_history)

    # 5c. Dump metrics to results/metrics.json
    metrics = {
        "naive_bayes": {
            "precision":        round(nb_p, 4),
            "recall":           round(nb_r, 4),
            "f1":               round(nb_f1, 4),
            "auc":              round(auc_nb, 4),
            "best_threshold":   round(nb_best_t, 4),
        },
        "logistic_regression": {
            "precision":        round(lr_p, 4),
            "recall":           round(lr_r, 4),
            "f1":               round(lr_f1, 4),
            "auc":              round(auc_lr, 4),
            "best_threshold":   round(lr_best_t, 4),
        },
    }
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  ✓ Metrics saved → {metrics_path}")

    print("\n" + "=" * 65)
    print("  Done. Check results/ for plots and metrics.")
    print("=" * 65)


if __name__ == "__main__":
    main()
