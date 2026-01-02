import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Sentiment Dashboard", layout="wide")
st.title("🚀 Sentiment Analysis Dashboard")
st.markdown("**1.55M tweets analyzed** | VADER + LDA topics")

@st.cache_data
def load_data():
    df = pd.read_csv('data/processed/topics_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    return df.sample(50000)  # Fast load

df = load_data()

# Sidebar filters
st.sidebar.header("Filters")
sentiment_filter = st.sidebar.multiselect("Sentiment", df['sentiment'].unique(), default=df['sentiment'].unique())
topic_filter = st.sidebar.multiselect("Topics", sorted(df['topic'].unique()), default=[0,1,2,3])

df_filtered = df[df['sentiment'].isin(sentiment_filter) & df['topic'].isin(topic_filter)]

# Row 1: Sentiment pie + topic bar
col1, col2 = st.columns(2)
with col1:
    fig_pie = px.pie(df_filtered, names='sentiment', title="Sentiment Distribution")
    st.plotly_chart(fig_pie, use_container_width=True)
with col2:
    fig_bar = px.bar(df_filtered['topic'].value_counts().reset_index(), 
                     x='topic', y='count', title="Topic Counts")
    st.plotly_chart(fig_bar, use_container_width=True)

# Row 2: Sentiment by topic
fig_heatmap = px.density_heatmap(df_filtered, x='topic', y='sentiment', 
                                 title="Sentiment Heatmap by Topic")
st.plotly_chart(fig_heatmap, use_container_width=True)

# Row 3: Sample tweets
st.subheader("Sample Tweets")
st.dataframe(df_filtered[['clean_text', 'sentiment', 'topic']].head(10), use_container_width=True)

# Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Tweets", len(df_filtered))
col2.metric("Positive %", f"{(df_filtered['sentiment']=='positive').mean():.1%}")
col3.metric("Avg Topics", df_filtered['topic'].nunique())
