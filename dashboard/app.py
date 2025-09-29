import streamlit as st
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px


# --------------------------
# Load Data
# --------------------------

import os

@st.cache_data
def load_data():
    data_path = os.path.join(os.path.dirname(__file__), '../data/sample_with_sentiment.csv')
    df = pd.read_csv(data_path, parse_dates=['date'])
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

st.title("COVID Tweet Sentiment Dashboard")

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

# --------------------------
# Location-based Sentiment
# --------------------------

loc_sent = filtered_df.dropna(subset=['location_clean']) \
    .groupby(['location_clean', 'sentiment_label']) \
    .size().unstack().fillna(0)

# ✅ Ensure all expected sentiment columns exist
for sentiment in ['Positive', 'Neutral', 'Negative']:
    if sentiment not in loc_sent.columns:
        loc_sent[sentiment] = 0

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
# Sample Tweets (Filtered)
# --------------------------

st.subheader("📄 Sample Tweets (Filtered)")

if not filtered_df.empty:
    st.dataframe(
        filtered_df[['date', 'user_location', 'sentiment_label', 'text']].sample(5)
    )
else:
    st.warning("No tweets found for this filter combination.")



from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

st.subheader("🔍 Discover Topics in Tweets (LDA)")

if not filtered_df['clean_text'].dropna().empty:
    # Sidebar: choose number of topics
    n_topics = st.sidebar.slider("Number of Topics", 2, 10, 5)

    # Vectorize clean_text
    vectorizer = CountVectorizer(
        max_df=0.95, 
        min_df=2, 
        stop_words='english'
    )
    text_data = filtered_df['clean_text'].dropna()
    dtm = vectorizer.fit_transform(text_data)

    # Fit LDA
    lda = LatentDirichletAllocation(
        n_components=n_topics, 
        random_state=42
    )
    lda.fit(dtm)

    # Get top words per topic
    words = vectorizer.get_feature_names_out()
    topics = []
    for idx, topic in enumerate(lda.components_):
        top_words = [words[i] for i in topic.argsort()[-10:]]
        topics.append({"Topic": f"Topic {idx+1}", "Top Words": ", ".join(top_words[::-1])})
    topics_df = pd.DataFrame(topics)

    st.subheader("🧾 Topics Discovered")
    st.dataframe(topics_df)

    # Assign topics to tweets
    topic_probs = lda.transform(dtm)
    dominant_topics = topic_probs.argmax(axis=1)

    # Cleaned df with matching length and metadata
    assigned_df = text_data.reset_index(drop=True).to_frame()
    assigned_df['Assigned_Topic'] = dominant_topics
    assigned_df['Assigned_Topic_Label'] = assigned_df['Assigned_Topic'].apply(lambda x: f"Topic {x+1}")

    # Add back original metadata (safe alignment)
    meta_cols = ['date', 'user_location', 'sentiment_label', 'text']
    for col in meta_cols:
        assigned_df[col] = filtered_df[col].dropna().reset_index(drop=True)

    # Sidebar: topic filter
    topic_options = ["All"] + sorted(assigned_df['Assigned_Topic_Label'].dropna().unique().tolist())
    selected_topic = st.sidebar.selectbox("Filter by Topic", topic_options)

    if selected_topic != "All":
        assigned_df = assigned_df[assigned_df['Assigned_Topic_Label'] == selected_topic]

    # Topic bar chart
    st.subheader("📊 Top Words in Selected Topic")
    topic_choice = st.selectbox("Select a Topic", topics_df['Topic'])
    topic_idx = int(topic_choice.split()[-1]) - 1
    top_words = [words[i] for i in lda.components_[topic_idx].argsort()[-10:]]
    word_weights = lda.components_[topic_idx][lda.components_[topic_idx].argsort()[-10:]]
    fig = px.bar(
        x=word_weights[::-1],
        y=top_words[::-1],
        orientation='h',
        title=f"Top Words in {topic_choice}"
    )
    st.plotly_chart(fig)

    # Show sample tweets for selected topic
    st.subheader("📄 Sample Tweets for Selected Topic")
    if not assigned_df.empty:
        st.dataframe(
            assigned_df[['date','user_location','sentiment_label','text','Assigned_Topic_Label']].sample(5)
        )
    else:
        st.warning("No tweets found for this topic selection.")
else:
    st.warning("No tweets available for topic modeling.")
