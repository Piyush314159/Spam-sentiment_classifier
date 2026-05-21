# 📬 Spam Classifier — From Scratch

> *"CONGRATULATIONS!! You have WON a FREE iPhone! Click NOW!!!"*
>
> — every spam message ever

This project teaches your computer to be as suspicious of that message as you are.
No magic libraries. No black boxes. Just math, numpy, and your own two hands.

---

## What this actually is

Two classic text classifiers — **Naive Bayes** and **Logistic Regression** — built entirely from scratch on the [UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection), a dataset of 5,574 real SMS messages labelled `spam` or `ham`.

By the end you will have:
- Implemented **Maximum Likelihood Estimation** to learn word probabilities
- Written **gradient descent** by hand, watched a loss curve fall in real time
- Plotted a **ROC curve** you computed yourself, not one sklearn handed you
- Understood *why* you'd ever move a decision threshold away from 0.5

---

## Project layout

```
spam-classifier/
├── data/
│   └── sms.tsv                  # raw dataset (download separately)
├── notebooks/
│   └── exploration.ipynb        # EDA — class balance, word frequencies, length dist.
├── src/
│   ├── preprocessing.py         # text cleaning → tokens → BoW / TF-IDF matrix
│   ├── naive_bayes.py           # MLE + Laplace smoothing + log-likelihood classifier
│   ├── logistic_regression.py   # sigmoid + cross-entropy + gradient descent
│   └── evaluation.py           # confusion matrix, P/R/F1, ROC, AUC, plots
├── main.py                      # runs the full pipeline top to bottom
├── results/                     # roc_curve.png, loss_curve.png, metrics.json
└── README.md
```

---

## The concepts, explained plainly

### 🔤 Bag of Words & TF-IDF
Before any model can touch text, you need to turn words into numbers.
**Bag of Words** counts how many times each word appears in a message — order ignored, just raw counts.
**TF-IDF** goes further: it downweights words that appear in *every* message (like "the") and upweights words that are rare but diagnostic (like "WINNER" or "FREE!!").
Naive Bayes gets raw counts. Logistic Regression gets TF-IDF weights.

### 🧮 Naive Bayes — the probabilistic filter
Bayes' theorem says: *given the words in this message, what's the probability it's spam?*

The "naive" part is the assumption that every word is **independent** of every other — obviously false (CONGRATULATIONS and FREE often travel together), but it works surprisingly well in practice.

You estimate `P(word | spam)` from the training data using **MLE**: just count.
You use **Laplace smoothing** (add 1 to every count) so that a word never seen in spam doesn't instantly assign zero probability to a whole message.
You work in **log space** to avoid multiplying thousands of tiny probabilities into floating-point zero.

### 📉 Logistic Regression — the geometric boundary
Instead of counting probabilities, LR learns a **weight** for every word that says how much it pushes a message toward spam or ham.

Training is **gradient descent**: start with all weights at zero, make a prediction, measure how wrong you were with **binary cross-entropy loss**, nudge every weight slightly in the direction that reduces the error, repeat.

After training, the model draws a linear boundary through the high-dimensional word-count space. Messages on one side are ham, the other side spam.

### 🎚️ Decision threshold — precision vs recall
Both models output a probability, not a hard label. The **threshold** decides where you draw the line.

- At `0.5` (default): balanced. You miss some spam but rarely flag real messages.
- At `0.3`: more aggressive. You catch more spam but occasionally nuke a real message.
- At `0.7`: conservative. Almost nothing gets flagged unless the model is very confident.

Try all three in `main.py`. Watch precision and recall move in opposite directions. That tension is the whole point.

### 📈 ROC curve & AUC
The **ROC curve** plots every possible threshold at once — True Positive Rate (spam caught) on the y-axis, False Positive Rate (ham wrongly flagged) on the x-axis.

A random classifier hugs the diagonal. A perfect one hits the top-left corner.
**AUC** (Area Under the Curve) collapses it to one number: 0.5 = random, 1.0 = perfect.

You compute both from scratch using the trapezoidal rule, no sklearn.

---

## Dataset

Download from UCI or Kaggle:
```
https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
```
Save to `data/sms.tsv`. It's tab-separated with two columns: `label` (ham/spam) and `message`.

**Class balance heads-up:** ~87% ham, ~13% spam. If your model just predicts ham every time it gets 87% accuracy. Watch F1, not raw accuracy.

---

## Setup

```bash
pip install numpy matplotlib
# optional, only if you use nltk stopwords in preprocessing.py
pip install nltk
```

No sklearn. That's the whole point.

---

## Run

```bash
python main.py
```

Outputs saved to `results/`:
| File | What it shows |
|------|---------------|
| `roc_curve.png` | NB vs LR curves on the same axes |
| `loss_curve.png` | LR training loss falling over iterations |
| `metrics.json` | Final P / R / F1 / AUC for both models |

---

## Implementation order

Fill in the files in this sequence — each one builds on the last:

```
preprocessing.py  →  naive_bayes.py  →  logistic_regression.py  →  evaluation.py  →  main.py
```

Don't touch `main.py` until the other four are working. Test each module in isolation first.

---

## Common traps

| Trap | Why it matters |
|------|----------------|
| Building vocabulary on the full dataset | Data leakage — the test set influences your feature space. Build vocab on train only. |
| Using raw probabilities instead of logs in NB | Multiplying 5,000 small floats → underflow → everything becomes zero. |
| Forgetting Laplace smoothing | One unseen word gives the whole message a zero probability and kills the classifier. |
| Evaluating on accuracy alone | 87% accuracy by predicting ham every time. Use F1. |
| High learning rate in LR | Loss oscillates or diverges instead of falling. Start at 0.01, tune up. |

---

## What you'll understand when you're done

- Why Naive Bayes is fast to train but makes a strong independence assumption
- How gradient descent actually moves weights — not abstractly, but in numpy
- Why the decision threshold is a design choice, not a fixed number
- What AUC really measures and why it's more honest than accuracy
- How TF-IDF beats raw counts by penalising common words

---

*Built for the ML fundamentals track — lectures 1–6.*