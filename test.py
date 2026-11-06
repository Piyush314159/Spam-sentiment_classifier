import pandas as pd
df = pd.read_csv(
    '/Users/piyushmaji/Desktop/Project/Spam_sentiment_classifier/data/SMSSpamCollection',
    sep='\t',           # '\t'---> is the tab character
    header=None,
    usecols=[0, 1],
    names=['label', 'message']
)

df['length'] = df['message'].str.len()  #creates a new column 'length' which contains the length of each message
df['word_count'] = df['message'].str.split().str.len()  #creates a new column 'word_count' which contains the number of words in each message

print(df.groupby('label')[['length', 'word_count']].describe())