# 📬 Spam Classifier — From Scratch

> *"CONGRATULATIONS!! You have WON a FREE iPhone! Click NOW!!!"*
>
> — every spam message ever

This project teaches your computer to be as suspicious of that message as you are.
No magic libraries. No black boxes. Just math, numpy, and your own two hands.

---

## 🏆 Results at a Glance

| Metric | Naive Bayes | Logistic Regression |
|--------|:-----------:|:-------------------:|
| **Precision** | 0.9924 | 0.8188 |
| **Recall** | 0.9097 | 0.9097 |
| **F1 Score** | **0.9493** ✅ | 0.8618 |
| **AUC** | 0.9786 | **0.9823** ✅ |
| **Best Threshold** | 0.7762 | 0.1511 |

> **Key insight:** Naive Bayes wins on **F1** (better precision at its optimal threshold), while Logistic Regression edges ahead on **AUC** (better ranking quality across all thresholds). Different metrics, different winners — and that's exactly the trade-off you'll learn to reason about.

### ROC Curve Comparison

<p align="center">
  <img src="results/roc_curve.png" alt="ROC Curve — Naive Bayes vs Logistic Regression" width="600"/>
</p>

Both models hug the top-left corner (near-perfect classification). The random-classifier diagonal sits far below — our models are doing real work, not guessing.

### Logistic Regression — Training Loss

<p align="center">
  <img src="results/loss_curve.png" alt="Logistic Regression Training Loss Curve" width="600"/>
</p>

Binary cross-entropy drops steeply in the first ~100 iterations and then slowly converges, confirming the learning rate (`0.1`) is well-tuned — no oscillation, no divergence.

---

## What this actually is

Two classic text classifiers — **Naive Bayes** and **Logistic Regression** — built entirely from scratch on the [UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection), a dataset of 5,572 real SMS messages labelled `spam` or `ham`.

By the end you will have:
- Implemented **Maximum Likelihood Estimation** to learn word probabilities
- Written **gradient descent** by hand, watched a loss curve fall in real time
- Plotted a **ROC curve** you computed yourself, not one sklearn handed you
- Understood *why* you'd ever move a decision threshold away from 0.5

---

## 📁 Project Layout

```
Spam_sentiment_classifier/
├── data/
│   └── SMSSpamCollection          # UTF-8 text, tab-separated, no header
├── notebooks/
│   └── exploration.ipynb          # EDA — class balance, word frequencies, length dist.
├── src/
│   ├── preprocessing.py           # text cleaning → tokens → BoW / TF-IDF matrix
│   ├── naive_bayes.py             # MLE + Laplace smoothing + log-likelihood classifier
│   ├── logistic_regression.py     # sigmoid + cross-entropy + gradient descent
│   └── evaluation.py              # confusion matrix, P/R/F1, ROC, AUC, plots
├── main.py                        # runs the full 5-step pipeline
├── results/
│   ├── roc_curve.png              # NB vs LR ROC curves
│   ├── loss_curve.png             # LR training loss over iterations
│   └── metrics.json               # final P / R / F1 / AUC for both models
└── Readme.md
```

---

## 🔧 Pipeline — 5 Steps

`main.py` runs end-to-end with a single command. Here's what each step does:

### Step 1 — Load & Preprocess

```
load_data() → clean_text() → tokenize() → train/test split (80/20)
→ build_vocabulary()  [train only — no data leakage]
→ vectorize_bow()     [for Naive Bayes — raw counts]
→ vectorize_tfidf()   [for Logistic Regression — normalised weights]
```

- **5,572 messages** loaded (4,825 ham / 747 spam)
- **Vocabulary size:** 7,609 unique tokens (built on training split only)
- Feature matrices: `(4457, 7609)` train, `(1115, 7609)` test

### Step 2 — Naive Bayes

```python
NaiveBayes(alpha=1.0).fit(X_train_bow, y_train)
```

- Trained on **BoW counts** (raw word frequencies)
- `predict_proba()` → `best_threshold()` → `predict()` → `precision_recall_f1()`
- `roc_curve()` + `auc()` computed from scratch

### Step 3 — Logistic Regression

```python
LogisticRegression(lr=0.1, n_iters=500).fit(X_train_tfidf, y_train)
```

- Trained on **TF-IDF weights** (normalised, downweights common words)
- Same evaluation flow as Naive Bayes
- `loss_history` saved for the training loss curve (final loss: **0.3555**)

### Step 4 — Compare

Both models evaluated side by side with `print_report()`. The pipeline automatically identifies which model wins on F1 vs AUC and explains why the winners may differ.

### Step 5 — Save Outputs

| Output | Path |
|--------|------|
| ROC curve plot | `results/roc_curve.png` |
| Loss curve plot | `results/loss_curve.png` |
| Metrics JSON | `results/metrics.json` |

---

## 🧠 The Concepts, Explained Plainly

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

### 🎚️ Decision Threshold — Precision vs Recall
Both models output a probability, not a hard label. The **threshold** decides where you draw the line.

| Threshold | Behaviour | Trade-off |
|-----------|-----------|-----------|
| `0.3` | Aggressive — catches more spam | Higher recall, lower precision |
| `0.5` | Balanced default | Middle ground |
| `0.7` | Conservative — high confidence only | Higher precision, lower recall |

The pipeline uses `best_threshold()` to find the F1-maximising cut-off automatically.

### 📈 ROC Curve & AUC
The **ROC curve** plots every possible threshold at once — True Positive Rate (spam caught) on the y-axis, False Positive Rate (ham wrongly flagged) on the x-axis.

A random classifier hugs the diagonal. A perfect one hits the top-left corner.
**AUC** (Area Under the Curve) collapses it to one number: 0.5 = random, 1.0 = perfect.

You compute both from scratch using the trapezoidal rule, no sklearn.

---

## 📊 Dataset

### The file: `SMSSpamCollection`

The dataset is a plain **UTF-8 text file** — no `.csv`, no `.tsv` extension, no header row.
Each line is one message, tab-separated into exactly two columns:

```
ham    Go until jurong point, crazy.. Available only in bugis n great world...
spam   Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005...
```

To load it in pandas:
```python
pd.read_csv('data/SMSSpamCollection', sep='\t', header=None, names=['label', 'message'])
```

### Where it comes from

The **SMS Spam Collection v.1** was compiled by Tiago Agostinho de Almeida and José María Gómez Hidalgo from four sources:

- **425 spam messages** scraped manually from [Grumbletext](http://www.grumbletext.co.uk/) — a UK forum where users publicly reported spam texts
- **450 ham messages** from Caroline Tagg's PhD thesis on SMS linguistics
- **3,375 ham messages** from the NUS SMS Corpus at the National University of Singapore — real messages from students, collected with consent
- **1,002 ham + 322 spam** from the SMS Spam Corpus v.0.1 by Gómez Hidalgo

### Stats

| Class | Count | Percentage |
|-------|:-----:|:----------:|
| Ham (legitimate) | 4,825 | 86.6% |
| Spam | 747 | 13.4% |
| **Total** | **5,572** | |

> **⚠️ Class imbalance:** 86.6% of messages are ham. A model that predicts ham for everything scores 86.6% accuracy while catching zero spam. This is why we evaluate on **F1**, not accuracy.

### Citation

> Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A. *Contributions to the study of SMS Spam Filtering: New Collection and Results.* Proceedings of the 2011 ACM Symposium on Document Engineering (ACM DOCENG'11), Mountain View, CA, USA, 2011.

### License

Free for research use. The corpus is provided as-is with no warranty.
Copyright © Tiago Agostinho de Almeida and José María Gómez Hidalgo.

---

## 🚀 Setup & Run

### Install dependencies

```bash
pip install numpy pandas matplotlib
```

No sklearn. That's the whole point.

### Run the pipeline

```bash
python main.py
```

You'll see the full 5-step output in the terminal:

```
=================================================================
  STEP 1 — Load & preprocess
=================================================================
  Loaded 5572 messages  (ham: 4825, spam: 747)
  ...
=================================================================
  STEP 4 — Compare (side by side)
=================================================================
Naive Bayes           |  P: 0.9924  R: 0.9097  F1: 0.9493  AUC: 0.9786
Logistic Regression   |  P: 0.8187  R: 0.9097  F1: 0.8618  AUC: 0.9823
  ...
=================================================================
  Done. Check results/ for plots and metrics.
=================================================================
```

---

## ⚠️ Common Traps

| Trap | Why it matters |
|------|----------------|
| Building vocabulary on the full dataset | Data leakage — the test set influences your feature space. Build vocab on train only. |
| Using raw probabilities instead of logs in NB | Multiplying 5,000 small floats → underflow → everything becomes zero. |
| Forgetting Laplace smoothing | One unseen word gives the whole message a zero probability and kills the classifier. |
| Evaluating on accuracy alone | 87% accuracy by predicting ham every time. Use F1. |
| High learning rate in LR | Loss oscillates or diverges instead of falling. Start at 0.01, tune up. |

---

## 🎓 What You'll Understand When You're Done

- Why Naive Bayes is fast to train but makes a strong independence assumption
- How gradient descent actually moves weights — not abstractly, but in numpy
- Why the decision threshold is a design choice, not a fixed number
- What AUC really measures and why it's more honest than accuracy
- How TF-IDF beats raw counts by penalising common words

---

*Built for the ML fundamentals track.*