"""
logistic_regression.py — Logistic regression trained with gradient descent. No sklearn.

Supports:
  - L2 regularisation (weight decay) to prevent overfitting on high-dim sparse data
  - Optional validation loss tracking to diagnose overfitting vs underfitting
"""
import numpy as np


class LogisticRegression:

    def __init__(self, lr=0.01, n_iters=1000, lambda_=0.0,
                 early_stopping=False, patience=20, min_delta=1e-4):
        """
        lr             : learning rate
        n_iters        : maximum number of gradient descent steps
        lambda_        : L2 regularisation strength (0 = no regularisation)
        early_stopping : if True, halt training when val loss stops improving
        patience       : how many consecutive iterations without improvement
                         before stopping (only used when early_stopping=True
                         and validation data is supplied to fit())
        min_delta      : minimum improvement in val loss to reset the patience
                         counter (treats smaller changes as "no improvement")
        Store weights, bias, and loss_history list after fit().
        """
        self.lr             = lr
        self.n_iters        = n_iters
        self.lambda_        = lambda_
        self.early_stopping = early_stopping
        self.patience       = patience
        self.min_delta      = min_delta
        self.weights = None         # w --> weights
        self.bias    = None         # b --> bias
        self.loss_history     = None
        self.val_loss_history = None
        self.stopped_at       = None  # iteration at which training was halted

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
        - Return scalar loss (without L2 penalty — that's added separately)
        """
        y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
        return -(np.mean(y * np.log(y_hat) + (1-y) * np.log(1 - y_hat)))

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Train weights via batch gradient descent with optional L2 regularisation.
        - Initialise weights to zeros, bias to 0
        - Each iteration:
            · forward pass  → sigmoid(X @ w + b)
            · compute and store loss in self.loss_history
            · if validation data provided, compute and store val loss
            · gradients:    dw = X.T @ (y_hat - y) / n + lambda_ * w
            ·               db = mean(y_hat - y)
            · update:       w -= lr * dw,  b -= lr * db
        - Early stopping: if early_stopping=True and X_val is provided,
          training halts when val loss fails to improve by at least min_delta
          for `patience` consecutive iterations.
        - Return self so you can chain .fit().predict()
        """
        self.weights = np.zeros(X.shape[1])
        self.bias    = 0
        self.loss_history     = []
        self.val_loss_history = [] if X_val is not None else None
        self.stopped_at       = self.n_iters  # default: ran to completion

        n = len(y)

        # Early-stopping state
        best_val_loss    = np.inf
        patience_counter = 0

        for i in range(self.n_iters):
            # Forward pass
            y_hat = self._sigmoid(X @ self.weights + self.bias)

            # Training loss (BCE + L2 penalty)
            bce       = self._loss(y, y_hat)
            l2_penalty = (self.lambda_ / 2) * np.sum(self.weights ** 2)
            loss       = bce + l2_penalty
            self.loss_history.append(loss)

            # Validation loss (if provided)
            if X_val is not None:
                y_hat_val = self._sigmoid(X_val @ self.weights + self.bias)
                val_bce   = self._loss(y_val, y_hat_val)
                val_loss  = val_bce + l2_penalty
                self.val_loss_history.append(val_loss)

                # Early stopping check
                if self.early_stopping:
                    if best_val_loss - val_loss > self.min_delta:
                        best_val_loss    = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    if patience_counter >= self.patience:
                        self.stopped_at = i + 1
                        print(f"  [early stopping] halted at iteration {self.stopped_at} "
                              f"(patience={self.patience}, min_delta={self.min_delta})")
                        break

            # Gradients with L2 regularisation
            dw = X.T @ (y_hat - y) / n + self.lambda_ * self.weights
            db = np.mean(y_hat - y)

            # Update
            self.weights -= self.lr * dw
            self.bias    -= self.lr * db

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