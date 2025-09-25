import streamlit as st
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px

# --------------------------
# Load Data
# --------------------------

@st.cache_data
def load_data():
    df = pd.read_csv('../data/sample_with_sentiment.csv', parse_dates=['date'])
    return df

df = load_data()

# --------------------------
# Sidebar Filters
# --------------------------

st.sidebar.title("🔎 Filters")

min_date = df['date'].min()
max_date = df['date'].max()

# Filter inputs
date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
selected_sentiment = st.sidebar.selectbox("Select Sentiment", ["All", "Positive", "Neutral", "Negative"])
locations = ["All"] + sorted(df['location_clean'].dropna().unique().tolist())
selected_location = st.sidebar.selectbox("Select Location", locations)

# Apply filters
filtered_df = df.copy()

if selected_sentiment != "All":
    filtered_df = filtered_df[filtered_df['sentiment_label'] == selected_sentiment]

if selected_location != "All":
    filtered_df = filtered_df[filtered_df['location_clean'] == selected_location]

if isinstance(date_range, list) and len(date_range) == 2:
    start_date, end_date = pd.to_datetime(date_range)
    filtered_df = filtered_df[(filtered_df['date'] >= start_date) & (filtered_df['date'] <= end_date)]

# --------------------------
# App Title
# --------------------------

st.title("🧠 COVID Tweet Sentiment Dashboard")

# --------------------------
# Word Cloud
# --------------------------

st.subheader("☁️ Word Cloud")

if not filtered_df['clean_text'].dropna().empty:
    text = ' '.join(filtered_df['clean_text'].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)
else:
    st.warning("No tweets available for this selection to generate word cloud.")

# --------------------------
# Sentiment Pie Chart
# --------------------------

st.subheader("📊 Sentiment Distribution")

sentiment_counts = filtered_df['sentiment_label'].value_counts().reset_index()
sentiment_counts.columns = ['Sentiment', 'Count']

if not sentiment_counts.empty:
    fig = px.pie(
        sentiment_counts,
        values='Count',
        names='Sentiment',
        title='Sentiment Breakdown',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    st.plotly_chart(fig)
else:
    st.warning("No sentiment data for selected filters.")

# --------------------------
# Sentiment Over Time
# --------------------------

st.subheader("📈 Tweet Sentiment Over Time")

time_df = filtered_df.groupby(['date', 'sentiment_label']).size().unstack().fillna(0)

if not time_df.empty:
    fig = px.line(time_df, title="Sentiment Trend Over Time")
    st.plotly_chart(fig)
else:
    st.warning("No time series data available.")

# --------------------------
# Location-Based Sentiment Analysis
# --------------------------

# Location aggregation
loc_sent = filtered_df.dropna(subset=['location_clean']) \
    .groupby(['location_clean', 'sentiment_label']) \
    .size().unstack().fillna(0)

if not loc_sent.empty:
    loc_sent['total'] = loc_sent.sum(axis=1)
    loc_sent['negative_pct'] = loc_sent['Negative'] / loc_sent['total']

    # Top 10 countries
    st.subheader("🌍 Top 10 Countries by Negative Sentiment (%)")
    top10_neg = loc_sent.sort_values(by='negative_pct', ascending=False).head(10)

    fig = px.bar(
        top10_neg,
        x='negative_pct',
        y=top10_neg.index,
        orientation='h',
        color='negative_pct',
        labels={'negative_pct': 'Negative Sentiment %'},
        title="Top 10 Countries by % Negative Tweets",
        color_continuous_scale='Reds'
    )
    st.plotly_chart(fig)

    # Choropleth map
    st.subheader("🗺️ Global Negative Sentiment Map")

    fig = px.choropleth(
        loc_sent,
        locations=loc_sent.index,
        locationmode='country names',
        color='negative_pct',
        hover_name=loc_sent.index,
        color_continuous_scale='Reds',
        title='Negative Sentiment % by Country'
    )
    st.plotly_chart(fig)
else:
    st.warning("Not enough location data to display top countries or map.")

# --------------------------
# Sample Tweets
# --------------------------

st.subheader("📄 Sample Tweets (Filtered)")

if not filtered_df.empty:
    st.dataframe(filtered_df[['date', 'user_location', 'sentiment_label', 'text']].sample(5))
else:
    st.warning("No tweets found for this filter combination.")



