import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

df = pd.read_csv('data/processed/sentiment_data.csv')
docs = df['clean_text'].fillna('').tolist()[:10000]  # 10k sample

print("Vectorizing...")
vectorizer = CountVectorizer(max_df=0.95, min_df=5, stop_words='english', max_features=2000)
X = vectorizer.fit_transform(docs)

print("Fitting LDA...")
lda = LatentDirichletAllocation(n_components=8, random_state=42, max_iter=10)
topics = lda.fit_transform(X)

df_subset = df.iloc[:10000].copy()
df_subset['topic'] = np.argmax(topics, axis=1)

def print_top_words(model, feature_names, n_top_words=8):
    for topic_idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words-1:-1]]
        print(f"Topic {topic_idx}: {', '.join(top_words)}")

print("\n📊 Top 8 Topics:")
print_top_words(lda, vectorizer.get_feature_names_out())

df_subset.to_csv('data/processed/topics_data.csv', index=False)
print(f"\n✅ LDA topics on {len(df_subset)} tweets!")
print("\nTopic counts:")
print(df_subset['topic'].value_counts().head())
