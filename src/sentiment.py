import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np

analyzer = SentimentIntensityAnalyzer()
df = pd.read_csv('data/processed/cleaned_data.csv')

def get_sentiment(text):
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        return 'positive'
    elif compound <= -0.05:
        return 'negative'
    else:
        return 'neutral'

print("Analyzing sentiment...")
df['sentiment'] = df['clean_text'].apply(get_sentiment)
df['compound_score'] = df['clean_text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

# Stats
print("\nSentiment distribution:")
print(df['sentiment'].value_counts(normalize=True).round(3))

df.to_csv('data/processed/sentiment_data.csv', index=False)
print("\n✅ Added sentiment column!")
print(df[['clean_text', 'sentiment', 'compound_score']].head())
print("\nPos: {:.1%}, Neg: {:.1%}, Neu: {:.1%}".format(
    (df['sentiment']=='positive').mean(),
    (df['sentiment']=='negative').mean(),
    (df['sentiment']=='neutral').mean()
))
