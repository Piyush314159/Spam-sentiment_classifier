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