# Spam / Sentiment Classifier

Naive Bayes + Logistic Regression built from scratch. No sklearn for modelling or metrics.

## Project layout

| File | Responsibility |
|------|----------------|
| `src/preprocessing.py` | Cleaning, tokenization, BoW, TF-IDF, train-test split |
| `src/naive_bayes.py` | MLE priors, Laplace smoothing, log-likelihood prediction |
| `src/logistic_regression.py` | Sigmoid, cross-entropy loss, gradient descent |
| `src/evaluation.py` | Confusion matrix, P/R/F1, ROC curve, AUC, plots |
| `main.py` | Wires everything together |

## Dataset
Download the UCI SMS Spam Collection:
https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
Save as `data/sms.tsv`  (tab-separated, two columns: label, message).

## Key concepts you will practise
- MLE for parameter estimation (NB priors + likelihoods)
- Laplace smoothing to handle zero-probability words
- Decision boundaries and the precision ↔ recall trade-off
- Gradient descent and the effect of learning rate on convergence
- ROC curve construction and AUC via the trapezoidal rule

## Run
python main.py
Plots are saved to results/.