"""
naive_bayes.py — Multinomial Naive Bayes built entirely from scratch using numpy.
"""

class NaiveBayes:

    def __init__(self, alpha=1.0):
        """
        alpha : Laplace smoothing constant (default 1.0)
        Store class priors and per-class word log-likelihoods after fit().
        """
        pass

    def fit(self, X, y):
        """
        Learn parameters from training data.
        - Compute P(spam) and P(ham) from class counts  → store as log priors
        - For each class, sum word counts across all documents in that class
        - Apply Laplace smoothing: (count + alpha) / (total + alpha * vocab_size)
        - Store log-probabilities (not raw probabilities) to avoid underflow
        """
        pass

    def predict_proba(self, X):
        """
        Compute posterior probability of spam for each document.
        - For each row: log_prior + sum of log_likelihoods for words present
        - Convert log-posteriors to probabilities via softmax or by exponentiating
        - Return array of shape (n_docs,) with P(spam | doc)
        """
        pass

    def predict(self, X, threshold=0.5):
        """
        Classify each document as spam (1) or ham (0).
        - Call predict_proba(), compare to threshold
        - Return integer array of shape (n_docs,)
        """
        pass