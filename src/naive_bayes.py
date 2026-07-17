"""
naive_bayes.py — Multinomial Naive Bayes built entirely from scratch using numpy.
"""
import numpy as np


class NaiveBayes:

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.log_prior = None
        self.log_likelihood = None

    def fit(self, X, y):
        n_docs = len(y)
        n_spam = (y == 1).sum()
        n_ham = (y == 0).sum()
        log_priors = np.array([np.log(n_ham / n_docs), np.log(n_spam / n_docs)])

        x_ham = X[y == 0]
        x_spam = X[y == 1]
        # np.asarray().ravel() normalises sparse matrix sums (returns np.matrix) to 1-D arrays
        ham_counts = np.asarray(x_ham.sum(axis=0)).ravel()
        spam_counts = np.asarray(x_spam.sum(axis=0)).ravel()

        vocab_size = X.shape[1]
        ham_probs = (ham_counts + self.alpha) / (ham_counts.sum() + self.alpha * vocab_size)
        spam_probs = (spam_counts + self.alpha) / (spam_counts.sum() + self.alpha * vocab_size)

        self.log_prior = log_priors
        self.log_likelihood = np.array([np.log(ham_probs), np.log(spam_probs)])

    def predict_proba(self, X):
        # np.asarray() normalises sparse @ dense result to a plain ndarray
        log_scores = np.asarray(X @ self.log_likelihood.T) + self.log_prior
        exp_scores = np.exp(log_scores - log_scores.max(axis=1, keepdims=True))
        probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        return probs[:, 1]

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)