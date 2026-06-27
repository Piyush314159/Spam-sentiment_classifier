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
"""
import os
import numpy as np

import src.preprocessing as prep
import src.naive_bayes as nbs
import src.logistic_regression as lgr
import src.evaluation as evl

DATA_PATH   = '/Users/piyushmaji/Desktop/ML_Project/Spam_sentiment_classifier/data/SMSSpamCollection'
RESULTS_DIR = '/Users/piyushmaji/Desktop/ML_Project/Spam_sentiment_classifier/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── 1. Load & preprocess ────────────────────────────────────────────────────
texts, labels = prep.load_data(DATA_PATH)
cleaned   = [prep.clean_text(t) for t in texts]
tokenized = [prep.tokenize(t)   for t in cleaned]
y = np.array(labels)

# Split indices first so vocab is built on train only — prevents data leakage
n   = len(y)
rng = np.random.default_rng(42)
idx = rng.permutation(n)
split = int(n * 0.8)
train_idx, test_idx = idx[:split], idx[split:]

tokenized_train = [tokenized[i] for i in train_idx]
tokenized_test  = [tokenized[i] for i in test_idx]
y_train, y_test = y[train_idx], y[test_idx]

vocab = prep.build_vocabulary(tokenized_train)      # train split only

X_train_bow = prep.vectorize_bow(tokenized_train, vocab)   # Naive Bayes (raw counts)
X_test_bow  = prep.vectorize_bow(tokenized_test,  vocab)

X_train_tfidf = prep.vectorize_tfidf(X_train_bow)          # Logistic Regression
X_test_tfidf  = prep.vectorize_tfidf(X_test_bow)

print(f"Train: {X_train_bow.shape}   Test: {X_test_bow.shape}   Vocab: {len(vocab)}")

# ── 2. Naive Bayes ──────────────────────────────────────────────────────────
nb = nbs.NaiveBayes(alpha=1.0)
nb.fit(X_train_bow, y_train)

nb_proba          = nb.predict_proba(X_test_bow)
nb_thresh, _      = evl.best_threshold(y_test, nb_proba, metric='f1')
nb_pred           = nb.predict(X_test_bow, threshold=nb_thresh)

nb_precision, nb_recall, nb_f1 = evl.precision_recall_f1(y_test, nb_pred)
nb_fpr, nb_tpr, _              = evl.roc_curve(y_test, nb_proba)
nb_auc                         = evl.auc(nb_fpr, nb_tpr)

# ── 3. Logistic Regression ──────────────────────────────────────────────────
lr = lgr.LogisticRegression(lr=0.1, n_iters=500)
lr.fit(X_train_tfidf, y_train)

lr_proba          = lr.predict_proba(X_test_tfidf)
lr_thresh, _      = evl.best_threshold(y_test, lr_proba, metric='f1')
lr_pred           = lr.predict(X_test_tfidf, threshold=lr_thresh)

lr_precision, lr_recall, lr_f1 = evl.precision_recall_f1(y_test, lr_pred)
lr_fpr, lr_tpr, _              = evl.roc_curve(y_test, lr_proba)
lr_auc                         = evl.auc(lr_fpr, lr_tpr)

# ── 4. Compare ──────────────────────────────────────────────────────────────
print(f"\n{'Model':<22s}  |  {'Precision':^9s}  {'Recall':^9s}  {'F1':^9s}  {'AUC':^9s}")
print("-" * 65)
evl.print_report("Naive Bayes",         nb_precision, nb_recall, nb_f1, nb_auc)
evl.print_report("Logistic Regression", lr_precision, lr_recall, lr_f1, lr_auc)

# ── 5. Save outputs ─────────────────────────────────────────────────────────
evl.plot_roc(
    nb_fpr, nb_tpr, nb_auc,
    lr_fpr, lr_tpr, lr_auc,
    save_path=os.path.join(RESULTS_DIR, 'roc_curve.png'),
)
evl.plot_loss_curve(
    lr.loss_history,
    save_path=os.path.join(RESULTS_DIR, 'loss_curve.png'),
)
