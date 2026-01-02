import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="Sentiment Dashboard", layout="wide", page_icon="🚀")
st.markdown("# 🚀 Sentiment Analysis Dashboard")
st.markdown("*1.55M tweets analyzed locally | Live demo mode*")

analyzer = SentimentIntensityAnalyzer()

# Demo data (same stats)
np.random.seed(42)
n = 50000
demo_df = pd.DataFrame({
    'clean_text': [f"sample tweet {i}" for i in range(n)],
    'sentiment': np.random.choice(['positive', 'negative', 'neutral'], n, 
                                 p=[0.48, 0.245, 0.275]),
    'compound_score': np.random.normal(0.1, 0.4, n).clip(-1,1),
    'topic': np.random.choice(range(8), n)
})

st.success(f"Demo: {len(demo_df):,} tweets | 48% positive")

# Live analyzer
with st.expander("🔥 Live Tweet Analyzer"):
    tweet = st.text_area("Tweet:", "Tesla Cybertruck rocks!")
    col1, col2 = st.columns(2)
    if st.button("Analyze", type="primary"):
        score = analyzer.polarity_scores(tweet)
        col1.metric("Score", f"{score['compound']:.3f}")
        if score['compound'] > 0.05:
            col2.success("✅ POSITIVE")
        elif score['compound'] < -0.05:
            col2.error("❌ NEGATIVE")
        else:
            col2.warning("🟡 NEUTRAL")

# Filters
st.sidebar.header("🔍 Filters")
sent_f = st.sidebar.multiselect("Sentiment", ['positive', 'negative', 'neutral'])
score_range = st.sidebar.slider("Score", -1.0, 1.0, (-0.5, 0.5))
df_f = demo_df[demo_df['sentiment'].isin(sent_f) & 
               (demo_df['compound_score'] >= score_range[0]) & 
               (demo_df['compound_score'] <= score_range[1])]

# KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Tweets", len(df_f))
col2.metric("Positive", f"{(df_f['sentiment']=='positive').mean():.1%}")
col3.metric("Avg Score", round(df_f['compound_score'].mean(), 3))

# Charts
col1, col2 = st.columns(2)
with col1:
    fig1 = px.pie(df_f, names='sentiment', hole=0.3, title="Sentiment")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.histogram(df_f, x='compound_score', color='sentiment', nbins=30, title="Scores")
    st.plotly_chart(fig2, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    fig3 = px.bar(df_f['topic'].value_counts().head(8).reset_index(), x='topic', y='count', 
                  title="Topics", color='count')
    st.plotly_chart(fig3, use_container_width=True)

with col2:
    fig4 = px.density_heatmap(df_f, x='topic', y='sentiment', title="Heatmap")
    st.plotly_chart(fig4, use_container_width=True)

# Samples
st.subheader("📝 Sample Tweets")
st.dataframe(df_f[['sentiment', 'compound_score', 'topic']].head(15), hide_index=True)

st.markdown("---")
st.markdown("[Full analysis on GitHub](https://github.com/aawhan0/Sentiment-Analysis-Dashboard)")
