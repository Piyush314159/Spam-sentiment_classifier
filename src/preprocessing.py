"""
preprocessing.py — Converts raw SMS text into a numeric matrix ready for modelling.

Public API
----------
load_data(filepath)                    → (texts, labels)
clean_text(text)                       → str
tokenize(text)                         → list[str]
build_vocabulary(tokenized_corpus)     → dict[str, int]
vectorize_bow(tokenized_docs, vocab)   → np.ndarray  (n_docs, vocab_size)
fit_tfidf(bow_matrix)                  → np.ndarray  (vocab_size,)  ← IDF vector
transform_tfidf(bow_matrix, idf)       → np.ndarray  (n_docs, vocab_size)
train_test_split(X, y, ...)            → X_train, X_test, y_train, y_test
"""
import pandas as pd
import numpy as np
from collections import Counter
import string
import re

STOPWORDS = {
    'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'she', 'they', 'it',
    'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do',
    'did', 'will', 'would', 'can', 'could', 'the', 'a', 'an', 'and', 'or',
    'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'this', 'that',
    'not', 'so', 'if', 'up', 'out', 'get', 'just', 'no', 'go', 'about'
}


def load_data(filepath):
    """
    Load the TSV file into a dataframe.
    - Read sms.tsv (tab-separated, columns: label, message)
    - Map 'ham' → 0, 'spam' → 1
    - Return (texts, labels) as lists
    """
    df = pd.read_csv(
        filepath,
        sep= '\t',
        header= None,
        usecols= [0,1],
        names= ['label', 'message']
    )
    df['label'] = df['label'].map({'ham' : 0, 'spam': 1 })
    return df['message'].tolist(), df['label'].tolist()     #returns a list of texts, label

def clean_text(text):
    """
    Normalise a single SMS string.
    - Lowercase everything
    - Remove punctuation and digits
    - Strip extra whitespace
    - Return cleaned string
    """
    text = text.lower()
    text = re.sub(r'\d+','', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+',' ', text).strip()
    return text


def tokenize(text):
    """
    Split a cleaned string into word tokens.
    - Split on whitespace
    - Remove stopwords (hardcoded set or nltk)
    - Return list of tokens
    """
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


def build_vocabulary(tokenized_corpus):
    """
    Build a word → index mapping from the training corpus only.
    - Collect every unique token across all documents
    - Assign each a stable integer index
    - Return vocab dict  {word: idx}
    """
    all_tokens = [token for doc in tokenized_corpus for token in doc]
    unique_tokens = sorted(set(all_tokens))
    vocab = {w:idx for idx,w in enumerate(unique_tokens)}
    return vocab


def vectorize_bow(tokenized_docs, vocab):
    """
    Convert tokenized documents into a Bag-of-Words count matrix.
    - Shape: (n_docs, vocab_size)
    - Entry [i, j] = count of vocab word j in document i
    - Use numpy only, no sklearn
    - Return numpy array
    """
    n_docs = len(tokenized_docs)            # rows
    vocab_size = len(vocab)                 # columns
    bow = np.zeros((n_docs, vocab_size))
    for i, doc in enumerate(tokenized_docs):
        counts = Counter(doc)
        for word, count in counts.items():
            if word in vocab:
                bow[i, vocab[word]] = count  # count of word j in message i
    return bow



def fit_tfidf(bow_matrix):
    """
    Compute IDF weights from a training BoW matrix.
    Must be called on TRAINING data only — never on test data.

    - IDF = log(N / (df + 1))  where df = number of docs containing the word
      (+1 smoothing prevents division-by-zero for words absent in every doc)
    - Return 1-D numpy array of shape (vocab_size,)

    Usage
    -----
    idf = fit_tfidf(X_train_bow)
    X_train_tfidf = transform_tfidf(X_train_bow, idf)
    X_test_tfidf  = transform_tfidf(X_test_bow,  idf)   # same idf — no leakage
    """
    n_docs = bow_matrix.shape[0]
    df = (bow_matrix > 0).sum(axis=0)       # document frequency per word
    idf = np.log(n_docs / (df + 1))         # inverse document frequency
    return idf


def transform_tfidf(bow_matrix, idf):
    """
    Apply a pre-fitted IDF vector to a BoW count matrix.
    Safe to call on both training and test data with the *same* idf.

    - TF  = count / total tokens in document  (row-normalised)
    - TF-IDF = TF × IDF
    - Return numpy array of same shape as bow_matrix
    """
    row_totals = bow_matrix.sum(axis=1, keepdims=True)
    row_totals[row_totals == 0] = 1          # avoid division by zero for empty docs
    tf = bow_matrix / row_totals             # term frequency
    return tf * idf


def train_test_split(X, y, test_size=0.2, seed=42):
    """
    Randomly split X and y into train and test sets.
    - Shuffle indices with the given seed
    - Use test_size fraction for the test set
    - Return X_train, X_test, y_train, y_test
    """
    X, y = np.array(X, dtype=object), np.array(y)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(y))
    split = int(len(y) * (1- test_size))
    train_idx, test_idx = indices[:split], indices[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

if __name__ == '__main__':
    import os
    _data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'SMSSpamCollection')
    texts, labels = load_data(_data_path)
    print(f"Loaded {len(texts)} messages")

    cleaned   = [clean_text(t)  for t in texts]
    tokenized = [tokenize(t)    for t in cleaned]

    y = np.array(labels)
    tok_arr = np.array(tokenized, dtype=object)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(y))
    split = int(len(y) * 0.8)
    tok_train = tok_arr[idx[:split]].tolist()
    tok_test  = tok_arr[idx[split:]].tolist()
    y_train, y_test = y[idx[:split]], y[idx[split:]]

    vocab = build_vocabulary(tok_train)
    print(f"Vocabulary size: {len(vocab)}")

    X_train_bow = vectorize_bow(tok_train, vocab)
    X_test_bow  = vectorize_bow(tok_test,  vocab)

    idf = fit_tfidf(X_train_bow)
    X_train_tfidf = transform_tfidf(X_train_bow, idf)
    X_test_tfidf  = transform_tfidf(X_test_bow,  idf)
    print(f"BoW shape: {X_train_bow.shape},  TF-IDF shape: {X_train_tfidf.shape}")
    print(f"Test  BoW: {X_test_bow.shape},   TF-IDF test:  {X_test_tfidf.shape}")