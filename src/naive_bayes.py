"""
naive_bayes.py — Multinomial Naive Bayes built entirely from scratch using numpy.
"""
import numpy as np


class NaiveBayes:

    def __init__(self, alpha=1.0):
        """
        alpha : Laplace smoothing constant (default 1.0)
        Store class priors and per-class word log-likelihoods after fit().
        """
        self.alpha = alpha
        self.log_prior = None
        self.log_likelyhood = None

    def fit(self, X, y):
        """
        Learn parameters from training data.
        - Compute P(spam) and P(ham) from class counts  → store as log priors
        - For each class, sum word counts across all documents in that class
        - Apply Laplace smoothing: (count + alpha) / (total + alpha * vocab_size)
        - Store log-probabilities (not raw probabilities) to avoid underflow
        """

        #1-class priors
        n_docs = len(y)
        n_spam = (y==1).sum()
        n_ham = (y==0).sum()
        log_priors = np.array([np.log(n_ham/n_docs), np.log(n_spam/n_docs)])

        #2-word count per class
        x_ham = X[y==0]
        x_spam = X[y==1]
        ham_counts = x_ham.sum(axis=0)      # adding individual weights of all ham tokens individually(n,)
        spam_counts = x_spam.sum(axis=0)    # adding individual weights of all spam tokens individually

        #3-Laplace smoothing
        vocab_size = X.shape[1]
        ham_probs = (ham_counts + self.alpha)/(ham_counts.sum() + self.alpha * vocab_size)
        spam_probs = (spam_counts + self.alpha)/(spam_counts.sum() + self.alpha * vocab_size)

        #4-log probabilities
        self.log_prior = log_priors
        self.log_likelyhood = np.array([np.log(ham_probs), np.log(spam_probs)])

    def predict_proba(self, X):
        """
        Compute posterior probability of spam for each document.
        - For each row: log_prior + sum of log_likelihoods for words present
        - Convert log-posteriors to probabilities via softmax or by exponentiating
        - Return array of shape (n_docs,) with P(spam | doc)
        """
        log_scores = X @ self.log_likelyhood.T + log_prior

        exp_scores = np.exp(log_scores-log_scores.max(axis= 1, keepdims=True))     # stability trick
        """
        log_scores = [[-120.3,  -98.7],    ← doc 0
                      [-340.1, -401.2]]    ← doc 1

        # without trick
        np.exp(-340.1) → 0.0  underflow!  numbers too small

        # max per row
        log_scores.max(axis=1, keepdims=True) = [[-98.7],
                                                [-340.1]]
        # subtract max
        log_scores - max = [[-21.6,  0.0],
                            [  0.0, -61.1]]

        # now exp is safe
        np.exp(0.0)   = 1.0      ← manageable
        np.exp(-21.6) = very small but not zero
        """
        probs = exp_scores/exp_scores.sum(axis=1, keepdims=True)

        return probs[:, 1]

    def predict(self, X, threshold=0.5):
        """
        Classify each document as spam (1) or ham (0).
        - Call predict_proba(), compare to threshold
        - Return integer array of shape (n_docs,)
        """
        return (self.predict_proba(X) >= threshold).astype(int)