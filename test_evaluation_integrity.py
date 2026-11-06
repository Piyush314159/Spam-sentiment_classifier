import unittest
import numpy as np

from src.logistic_regression import LogisticRegression
from src.preprocessing import train_test_split


class EvaluationIntegrityTests(unittest.TestCase):
    def test_three_way_split_preserves_test_and_is_disjoint(self):
        ids = np.arange(100)
        development, test, y_dev, y_test = train_test_split(ids, ids, seed=42)
        train, valid, _, _ = train_test_split(development, y_dev, seed=43)
        self.assertEqual((len(train), len(valid), len(test)), (64, 16, 20))
        self.assertFalse(set(train) & set(valid))
        self.assertFalse(set(train) & set(test))
        self.assertFalse(set(valid) & set(test))
        np.testing.assert_array_equal(test, y_test)

    def test_early_stopping_restores_best_validation_weights(self):
        x = np.array([[1.], [-1.]])
        y = np.array([1, 0])
        # Validation labels deliberately oppose training: the initial weights
        # are the best validation checkpoint, before any training updates.
        model = LogisticRegression(lr=1., n_iters=10, early_stopping=True, patience=2)
        model.fit(x, y, X_val=x, y_val=1-y)
        np.testing.assert_array_equal(model.weights, [0.])
        self.assertEqual(model.bias, 0.)
