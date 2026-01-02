import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(page_title="Sentiment Dashboard", layout="wide")
st.title("🚀 Sentiment Analysis Dashboard")
st.markdown("**1.55M tweets analyzed** | VADER + LDA")

@st.cache_data
def load_data():
    df_sent = pd.read_csv('data/processed/sentiment_data.csv')
    try:
        df_topics = pd.read_csv('data/processed/topics_data.csv')
        df = pd.concat([df_sent, df_topics[['topic']].fillna(-1)], axis=1)
    except:
        df = df_sent.copy()
        df['topic'] = -1  # No topics
    return df.sample(min(50000, len(df)), random_state=42)

df = load_data()
st.write(f"Loaded {len(df):,} tweets")  # Debug

# Sidebar
st.sidebar.header("🔍 Filter")
sent_f = st.sidebar.multiselect("Sentiment", sorted(df['sentiment'].unique()), default=['positive', 'negative', 'neutral'])
df_f = df[df['sentiment'].isin(sent_f)]

if 'topic' in df.columns:
    top_f = st.sidebar.multiselect("Topic", sorted(df['topic'].dropna().unique()), default=sorted(df['topic'].dropna().unique())[:4])
    df_f = df_f[df_f['topic'].isin(top_f)]

# Charts
col1, col2 = st.columns(2)
with col1:
    fig1 = px.pie(df_f, names='sentiment', title="Sentiment Distribution")
    st.plotly_chart(fig1, use_container_width=True)
with col2:
    if 'topic' in df_f.columns:
        fig2 = px.bar(df_f['topic'].value_counts().head(10).reset_index(), x='topic', y='count', title="Top Topics")
        st.plotly_chart(fig2, use_container_width=True)

# Heatmap
if 'topic' in df_f.columns:
    fig3 = px.density_heatmap(df_f, x='topic', y='sentiment', nbinsx=10, title="Sentiment by Topic")
    st.plotly_chart(fig3, use_container_width=True)

# Samples
st.subheader("📝 Sample Tweets")
st.dataframe(df_f[['clean_text', 'sentiment', 'compound_score']].head(10), use_container_width=True)

# Metrics
c1, c2, c3 = st.columns(3)
c1.metric("Tweets", len(df_f))
c2.metric("Positive", f"{(df_f['sentiment']=='positive').mean():.1%}")
c3.metric("Avg Score", df_f['compound_score'].mean().round(3))
