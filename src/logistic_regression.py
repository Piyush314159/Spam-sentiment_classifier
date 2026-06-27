"""
logistic_regression.py — Logistic regression trained with gradient descent. No sklearn.
"""
import numpy as np


class LogisticRegression:

    def __init__(self, lr=0.01, n_iters=1000):
        """
        lr       : learning rate
        n_iters  : number of gradient descent steps
        Store weights, bias, and loss_history list after fit().
        """
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None         # w--> weights
        self.bias = None            # b--->bias
        self.loss_history = None

    def _sigmoid(self, z):
        """
        Numerically stable sigmoid: 1 / (1 + exp(-z))
        - Clip z to [-500, 500] before calling exp to prevent overflow
        - Return value in (0, 1)
        """
        z = np.clip(z, -500, 500)
        return 1/(1 + np.exp(-z))

    def _loss(self, y, y_hat):
        """
        Binary cross-entropy loss averaged over the batch.
        - L = -mean[ y*log(y_hat) + (1-y)*log(1-y_hat) ]
        - Clip y_hat away from 0 and 1 to avoid log(0)
        - Return scalar loss
        """
        y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
        return -(np.mean(y * np.log(y_hat) + (1-y) * np.log(1 - y_hat)))

    def fit(self, X, y):
        """
        Train weights via batch gradient descent.
        - Initialise weights to zeros, bias to 0
        - Each iteration:
            · forward pass  → sigmoid(X @ w + b)
            · compute and store loss in self.loss_history
            · gradients:    dw = X.T @ (y_hat - y) / n,  db = mean(y_hat - y)
            · update:       w -= lr * dw,  b -= lr * db
        - Return self so you can chain .fit().predict()
        """
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        self.loss_history = []

        for i in range(self.n_iters):
            y_hat = self._sigmoid(X @ self.weights + self.bias)
            loss = self._loss(y, y_hat)
            self.loss_history.append(loss)

            n = len(y)
            dw = X.T @ (y_hat - y) / n
            db = np.mean(y_hat - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        return self

    def predict_proba(self, X):
        """
        Forward pass on new data.
        - Compute sigmoid(X @ self.weights + self.bias)
        - Return array of shape (n_docs,)
        """
        return self._sigmoid(X @ self.weights + self.bias)

    def predict(self, X, threshold=0.5):
        """
        Classify using a tunable decision threshold.
        - Call predict_proba(), compare to threshold
        - Return integer array of shape (n_docs,)
        - Try 0.3 / 0.5 / 0.7 in main.py to observe precision-recall trade-off
        """
        return (self.predict_proba(X) >= threshold).astype(int)