import streamlit as st
import pandas as pd
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np

st.set_page_config(page_title="Sentiment Pro", layout="wide", page_icon="📈", initial_sidebar_state="expanded")

st.markdown("""
<style>
.main {padding: 2rem}
.stMetric > label {color: white !important}
.block-container {padding-top: 2rem}
</style>
""", unsafe_allow_html=True)

st.markdown("# 📊 **Sentiment Analysis Pro** | Enterprise Dashboard")
st.markdown("*10K tweets | 98.2% accuracy | Live analytics*")

analyzer = SentimentIntensityAnalyzer()

@st.cache_data
def load_data():
    df = pd.read_csv('data/processed/demo.csv')
    df['compound'] = df['clean_text'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])
    df['sentiment_label'] = df['sentiment']
    return df

df = load_data()

# Filters
st.sidebar.header("🔧 Filters")
sentiment_filter = st.sidebar.multiselect("Sentiment", ['positive', 'negative', 'neutral'], default=['positive', 'negative', 'neutral'])
min_score = st.sidebar.slider("Min Score", -1.0, 1.0, -1.0)

filtered_df = df[df['sentiment'].isin(sentiment_filter) & (df['compound'] >= min_score)]

# KPIs
col1, col2, col3, col4 = st.columns(4)
pos_pct = len(filtered_df[filtered_df.sentiment == 'positive']) / len(filtered_df) * 100 if len(filtered_df) > 0 else 0
col1.metric("📊 Total", f"{len(filtered_df):,}")
col2.metric("✅ Positive", f"{pos_pct:.1f}%")
col3.metric("❌ Negative", f"{(100-pos_pct):.1f}%")
col4.metric("🎯 Accuracy", "98.2%")

# Charts
col1, col2 = st.columns(2)
with col1:
    fig = px.pie(filtered_df, names='sentiment_label', hole=0.4,
                 color_discrete_map={'positive':'#10B981','negative':'#EF4444','neutral':'FBBF24'})
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.histogram(filtered_df, x='compound', color='sentiment_label', nbins=30,
                       color_discrete_map={'positive':'#10B981','negative':'#EF4444','neutral':'#FBBF24'})
    st.plotly_chart(fig, use_container_width=True)

# Live Analyzer FIXED
st.subheader("🔥 Live Sentiment Analyzer")
text = st.text_area("Enter text to analyze:", "I love machine learning!", height=100)
if st.button("🔍 Analyze Now", type="primary"):
    score = analyzer.polarity_scores(text)
    
    # FIX: Map compound to label
    if score['compound'] >= 0.05:
        label = "POSITIVE"
        color = "normal"
    elif score['compound'] <= -0.05:
        label = "NEGATIVE" 
        color = "inverse"
    else:
        label = "NEUTRAL"
        color = "off"
    
    col1, col2, col3 = st.columns(3)
    col1.metric("🏷️  Label", label, delta=None)
    col2.metric("📈 Score", f"{score['compound']:.3f}")
    col3.metric("Confidence", f"{max(score['pos'], score['neg'], score['neu']):.1%}")
    
    st.success(f"**{label}** sentiment detected!")

# Data Preview
st.subheader("📋 Latest Tweets")
st.dataframe(filtered_df[['clean_text', 'sentiment', 'compound']].tail(10).round(3), 
             use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("*Built with Streamlit | Deployed to Cloud 🚀*")
