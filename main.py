"""
main.py — Runs the full Spam/Ham classification pipeline end-to-end.
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

# ── src imports ──────────────────────────────────────────────────────────────
from src.preprocessing import (
    load_data,
    clean_text,
    tokenize,
    build_vocabulary,
    vectorize_bow,
    vectorize_tfidf,
    train_test_split,
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

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH    = os.path.join("data", "SMSSpamCollection")
RESULTS_DIR  = "results"
ROC_PATH     = os.path.join(RESULTS_DIR, "roc_curve.png")
LOSS_PATH    = os.path.join(RESULTS_DIR, "loss_curve.png")
METRICS_PATH = os.path.join(RESULTS_DIR, "metrics.json")

os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load & Preprocess
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — Loading and preprocessing data")
print("=" * 60)

texts, labels = load_data(DATA_PATH)
print(f"  Total messages loaded : {len(texts)}")
print(f"  Spam count            : {sum(labels)}")
print(f"  Ham  count            : {len(labels) - sum(labels)}")

# Clean and tokenize all messages
cleaned   = [clean_text(t) for t in texts]
tokenized = [tokenize(t)   for t in cleaned]

labels_np = np.array(labels)

# --- Train / Test split on raw token lists & labels -------------------------
# train_test_split() from src/preprocessing does np.array(X) internally,
# producing a numpy object array for list-of-lists; .tolist() converts it
# back to a plain Python list so build_vocabulary / vectorize_bow are happy.
# Vocabulary is built ONLY on training tokens — no data leakage.
tokenized_train, tokenized_test, y_train, y_test = train_test_split(
    tokenized, labels_np, test_size=0.2, seed=42
)
tokenized_train = tokenized_train.tolist()
tokenized_test  = tokenized_test.tolist()

# Build vocabulary from TRAIN only (no data leakage)
vocab = build_vocabulary(tokenized_train)
print(f"\n  Vocabulary size (train) : {len(vocab)}")
print(f"  Train samples           : {len(y_train)}")
print(f"  Test  samples           : {len(y_test)}")

# BoW matrices (for Naive Bayes — raw counts)
X_train_bow = vectorize_bow(tokenized_train, vocab)
X_test_bow  = vectorize_bow(tokenized_test,  vocab)

# TF-IDF matrices (for Logistic Regression — normalised weights)
X_train_tfidf = vectorize_tfidf(X_train_bow)
X_test_tfidf  = vectorize_tfidf(X_test_bow)

print(f"\n  BoW   shape  — train: {X_train_bow.shape},   test: {X_test_bow.shape}")
print(f"  TF-IDF shape — train: {X_train_tfidf.shape}, test: {X_test_tfidf.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Naive Bayes
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — Naive Bayes (Multinomial, alpha=1.0)")
print("=" * 60)

nb = NaiveBayes(alpha=1.0)
nb.fit(X_train_bow, y_train)

nb_proba = nb.predict_proba(X_test_bow)

# Tune decision threshold to maximise F1
nb_thresh, nb_best_f1 = best_threshold(y_test, nb_proba, metric='f1')
print(f"  Best threshold (F1)  : {nb_thresh:.4f}  →  F1 = {nb_best_f1:.4f}")

nb_preds = nb.predict(X_test_bow, threshold=nb_thresh)
nb_precision, nb_recall, nb_f1 = precision_recall_f1(y_test, nb_preds)

nb_fpr, nb_tpr, _ = roc_curve(y_test, nb_proba)
nb_auc            = auc(nb_fpr, nb_tpr)

print(f"  Precision : {nb_precision:.4f}")
print(f"  Recall    : {nb_recall:.4f}")
print(f"  F1        : {nb_f1:.4f}")
print(f"  AUC       : {nb_auc:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Logistic Regression
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — Logistic Regression (lr=0.1, n_iters=500, lambda_=0.01)")
print("=" * 60)

lr_model = LogisticRegression(lr=0.1, n_iters=500, lambda_=0.01)
lr_model.fit(
    X_train_tfidf, y_train,
    X_val=X_test_tfidf, y_val=y_test   # track val loss → diagnose overfitting
)

lr_proba = lr_model.predict_proba(X_test_tfidf)

# Tune decision threshold to maximise F1
lr_thresh, lr_best_f1 = best_threshold(y_test, lr_proba, metric='f1')
print(f"  Best threshold (F1)  : {lr_thresh:.4f}  →  F1 = {lr_best_f1:.4f}")

lr_preds = lr_model.predict(X_test_tfidf, threshold=lr_thresh)
lr_precision, lr_recall, lr_f1 = precision_recall_f1(y_test, lr_preds)

lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_proba)
lr_auc            = auc(lr_fpr, lr_tpr)

print(f"  Precision : {lr_precision:.4f}")
print(f"  Recall    : {lr_recall:.4f}")
print(f"  F1        : {lr_f1:.4f}")
print(f"  AUC       : {lr_auc:.4f}")
print(f"  Final train loss : {lr_model.loss_history[-1]:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Side-by-side comparison
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — Model Comparison")
print("=" * 60)
header = f"{'Model':<20}  |  {'P':>6}  {'R':>6}  {'F1':>6}  {'AUC':>6}"
print(header)
print("-" * len(header))
print_report("Naive Bayes",          nb_precision, nb_recall, nb_f1, nb_auc)
print_report("Logistic Regression",  lr_precision, lr_recall, lr_f1, lr_auc)

# Quick human-readable commentary
nb_wins_f1  = nb_f1  > lr_f1
nb_wins_auc = nb_auc > lr_auc
print()
print(f"  F1  winner  : {'Naive Bayes' if nb_wins_f1  else 'Logistic Regression'}")
print(f"  AUC winner  : {'Naive Bayes' if nb_wins_auc else 'Logistic Regression'}")
print()
print("  Why?  NB is strong at sparse text due to its independence assumption,")
print("  while LR leverages TF-IDF weights for better probabilistic ranking,")
print("  often yielding a higher AUC even when per-class precision/recall differs.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Save outputs
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 — Saving results")
print("=" * 60)

# ROC curve (both models on one plot)
plot_roc(
    nb_fpr, nb_tpr, nb_auc,
    lr_fpr, lr_tpr, lr_auc,
    save_path=ROC_PATH,
)

# Logistic Regression loss curve (train + val)
plot_loss_curve(
    loss_history=lr_model.loss_history,
    save_path=LOSS_PATH,
    val_loss_history=lr_model.val_loss_history,
)

# Metrics JSON dump
metrics = {
    "naive_bayes": {
        "threshold" : nb_thresh,
        "precision" : nb_precision,
        "recall"    : nb_recall,
        "f1"        : nb_f1,
        "auc"       : nb_auc,
    },
    "logistic_regression": {
        "threshold"       : lr_thresh,
        "precision"       : lr_precision,
        "recall"          : lr_recall,
        "f1"              : lr_f1,
        "auc"             : lr_auc,
        "final_train_loss": lr_model.loss_history[-1],
    },
}

with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"[metrics]        saved → {METRICS_PATH}")

print("\n" + "=" * 60)
print("Pipeline complete.")
print("=" * 60)
