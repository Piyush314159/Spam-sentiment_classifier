"""
preprocessing.py — Converts raw SMS text into a numeric matrix ready for modelling.
"""
import pandas as pd
import numpy as np
from collections import Counter
import string
import re
from enum import unique

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
    n_docs = len(tokenized_docs)            #row
    vocab_size = len(vocab)                 #coloumn
    bow = np.zeros((n_docs, vocab_size))
    for i,doc in enumerate(tokenized_docs):
        counts = Counter(doc)
        for word, count in counts.items():
            if word in vocab:
                bow[i,vocab[word]] = count  # so it add the count in particular message and for specific word
                #   message, word
    return bow



def vectorize_tfidf(bow_matrix):
    """
    Transform a raw BoW count matrix into TF-IDF weights.
    - TF  = count / total tokens in document
    - IDF = log(N / df)  where df = number of docs containing the word
    - Multiply TF × IDF element-wise
    - Return numpy array of same shape
    """
    n_docs = bow_matrix.shape[0]

    row_totals = bow_matrix.sum(axis=1, keepdims=True)
    row_totals[row_totals==0] = 1
    tf = bow_matrix / row_totals            #term frequency

    df = (bow_matrix > 0).sum(axis = 0)     #doc ferquency or how many message conatins this word
    idf = np.log(n_docs / (df+1))           #inverse doc frequency

    return tf * idf


def train_test_split(X, y, test_size=0.2, seed=42):
    """
    Randomly split X and y into train and test sets.
    - Shuffle indices with the given seed
    - Use test_size fraction for the test set
    - Return X_train, X_test, y_train, y_test
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(y))
    split = int(len(y) * (1- test_size))
    train_idx, test_idx = indices[:split], indices[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

if __name__ == '__main__':
    texts, labels = load_data('/Users/piyushmaji/Desktop/ML_Project/Spam_sentiment_classifier/data/SMSSpamCollection')
    print(f"Loaded {len(texts)} messages")

    cleaned = [clean_text(t) for t in texts]
    tokenized = [tokenize(t) for t in cleaned]
    vocab = build_vocabulary(tokenized)
    print(f"Vocabulary size: {len(vocab)}")

    bow = vectorize_bow(tokenized, vocab)
    tfidf = vectorize_tfidf(bow)
    print(f"BoW shape: {bow.shape}, TF-IDF shape: {tfidf.shape}")

    y = np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(tfidf, y)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")