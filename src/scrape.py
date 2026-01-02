import snscrape.modules.twitter as sntwitter
import pandas as pd
query = '"Tesla" since:2025-11-01 lang:en -is:retweet'
tweets_list = []
print("Scraping...")
for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
    if i > 2000: break
    tweets_list.append([tweet.date, tweet.rawContent, tweet.id, 'Twitter'])
df = pd.DataFrame(tweets_list, columns=['date', 'text', 'id', 'source'])
df['date'] = pd.to_datetime(df['date']).dt.date
df.to_csv('data/raw/tesla_tweets.csv', index=False)
print(f"✅ {len(df)} tweets!")
print(df.head(3)['text'].tolist())
