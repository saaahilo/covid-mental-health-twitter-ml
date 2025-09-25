import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px

# Load your preprocessed CSV
@st.cache_data
def load_data():
    df = pd.read_csv('../data/sample_with_sentiment.csv', parse_dates=['date'])
    return df

df = load_data()

# Sidebar filters
st.sidebar.title("Filters")
selected_sentiment = st.sidebar.selectbox("Select Sentiment", ["All", "Positive", "Neutral", "Negative"])
selected_location = st.sidebar.selectbox("Select Location", ["All"] + sorted(df['location_clean'].dropna().unique().tolist()))

# Apply filters
filtered_df = df.copy()
if selected_sentiment != "All":
    filtered_df = filtered_df[filtered_df['sentiment_label'] == selected_sentiment]
if selected_location != "All":
    filtered_df = filtered_df[filtered_df['location_clean'] == selected_location]

# Title
st.title("🌍 COVID Tweet Sentiment Dashboard")

# Word Cloud
st.subheader("Word Cloud")
text = ' '.join(filtered_df['clean_text'].dropna())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
st.pyplot(plt)

# Sentiment Over Time
st.subheader("Tweet Sentiment Over Time")
time_df = filtered_df.groupby(['date', 'sentiment_label']).size().unstack().fillna(0)
fig = px.line(time_df, title="Sentiment Trend Over Time")
st.plotly_chart(fig)

# Top 10 Countries by Negative Sentiment
st.subheader("Top 10 Countries by % Negative Sentiment")
loc_sent = df.dropna(subset=['location_clean']).groupby(['location_clean', 'sentiment_label']).size().unstack().fillna(0)
loc_sent['total'] = loc_sent.sum(axis=1)
loc_sent['negative_pct'] = loc_sent['Negative'] / loc_sent['total']
top10 = loc_sent.sort_values(by='negative_pct', ascending=False).head(10)
fig = px.bar(top10, x='negative_pct', y=top10.index, orientation='h', color='negative_pct',
             labels={'negative_pct': 'Negative Sentiment %'}, title="Top 10 Negative Countries")
st.plotly_chart(fig)

# Choropleth Map
st.subheader("Global Negative Sentiment Map")
fig = px.choropleth(
    loc_sent,
    locations=loc_sent.index,
    locationmode='country names',
    color='negative_pct',
    color_continuous_scale='Reds',
    title='Negative Sentiment % by Country'
)
st.plotly_chart(fig)



