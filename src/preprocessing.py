"""
preprocessing.py — Converts raw SMS text into a numeric matrix ready for modelling.
"""

def load_data(filepath):
    """
    Load the TSV file into a dataframe.
    - Read sms.tsv (tab-separated, columns: label, message)
    - Map 'ham' → 0, 'spam' → 1
    - Return (texts, labels) as lists
    """
    pass

def clean_text(text):
    """
    Normalise a single SMS string.
    - Lowercase everything
    - Remove punctuation and digits
    - Strip extra whitespace
    - Return cleaned string
    """
    pass

def tokenize(text):
    """
    Split a cleaned string into word tokens.
    - Split on whitespace
    - Remove stopwords (hardcoded set or nltk)
    - Return list of tokens
    """
    pass

def build_vocabulary(tokenized_corpus):
    """
    Build a word → index mapping from the training corpus only.
    - Collect every unique token across all documents
    - Assign each a stable integer index
    - Return vocab dict  {word: idx}
    """
    pass

def vectorize_bow(tokenized_docs, vocab):
    """
    Convert tokenized documents into a Bag-of-Words count matrix.
    - Shape: (n_docs, vocab_size)
    - Entry [i, j] = count of vocab word j in document i
    - Use numpy only, no sklearn
    - Return numpy array
    """
    pass

def vectorize_tfidf(bow_matrix):
    """
    Transform a raw BoW count matrix into TF-IDF weights.
    - TF  = count / total tokens in document
    - IDF = log(N / df)  where df = number of docs containing the word
    - Multiply TF × IDF element-wise
    - Return numpy array of same shape
    """
    pass

def train_test_split(X, y, test_size=0.2, seed=42):
    """
    Randomly split X and y into train and test sets.
    - Shuffle indices with the given seed
    - Use test_size fraction for the test set
    - Return X_train, X_test, y_train, y_test
    """
    pass