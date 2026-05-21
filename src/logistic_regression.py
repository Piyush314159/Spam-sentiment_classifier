"""
logistic_regression.py — Logistic regression trained with gradient descent. No sklearn.
"""

class LogisticRegression:

    def __init__(self, lr=0.01, n_iters=1000):
        """
        lr       : learning rate
        n_iters  : number of gradient descent steps
        Store weights, bias, and loss_history list after fit().
        """
        pass

    def _sigmoid(self, z):
        """
        Numerically stable sigmoid: 1 / (1 + exp(-z))
        - Clip z to [-500, 500] before calling exp to prevent overflow
        - Return value in (0, 1)
        """
        pass

    def _loss(self, y, y_hat):
        """
        Binary cross-entropy loss averaged over the batch.
        - L = -mean[ y*log(y_hat) + (1-y)*log(1-y_hat) ]
        - Clip y_hat away from 0 and 1 to avoid log(0)
        - Return scalar loss
        """
        pass

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
        pass

    def predict_proba(self, X):
        """
        Forward pass on new data.
        - Compute sigmoid(X @ self.weights + self.bias)
        - Return array of shape (n_docs,)
        """
        pass

    def predict(self, X, threshold=0.5):
        """
        Classify using a tunable decision threshold.
        - Call predict_proba(), compare to threshold
        - Return integer array of shape (n_docs,)
        - Try 0.3 / 0.5 / 0.7 in main.py to observe precision-recall trade-off
        """
        pass