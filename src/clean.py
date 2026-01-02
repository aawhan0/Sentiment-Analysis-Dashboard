import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
stop_words = set(stopwords.words('english'))

print("Loading data...")
df = pd.read_csv('data/raw/twitter_data.csv', encoding='latin-1', header=None, 
                 usecols=[5], names=['text'])
df = df.dropna().reset_index(drop=True)
df['text'] = df['text'].astype(str)
print(f"Raw: {len(df)} tweets")

def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # URLs
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Punct
    text = re.sub(r'\s+', ' ', text).strip().lower()
    words = [w for w in text.split() if w not in stop_words and len(w) > 2]
    return ' '.join(words)

df['clean_text'] = df['text'].apply(clean_text)  # FIXED: df['text']
df = df[df['clean_text'].str.len() > 10]
df.to_csv('data/processed/cleaned_data.csv', index=False)
print(f"✅ Cleaned {len(df)} tweets!")
print("\nSample:")
print(df[['text', 'clean_text']].head())
