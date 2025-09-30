import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# --------------------------
# Load Data
# --------------------------

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

date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
selected_sentiment = st.sidebar.selectbox("Select Sentiment", ["All", "Positive", "Neutral", "Negative"])
locations = ["All"] + sorted(df['location_clean'].dropna().unique().tolist())
selected_location = st.sidebar.selectbox("Select Location", locations)

filtered_df = df.copy()

if selected_sentiment != "All":
    filtered_df = filtered_df[filtered_df['sentiment_label'] == selected_sentiment]
if selected_location != "All":
    filtered_df = filtered_df[filtered_df['location_clean'] == selected_location]
if isinstance(date_range, list) and len(date_range) == 2:
    start_date, end_date = pd.to_datetime(date_range)
    filtered_df = filtered_df[(filtered_df['date'] >= start_date) & (filtered_df['date'] <= end_date)]

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

# Group and pivot sentiment counts per country
loc_sent = filtered_df.dropna(subset=['location_clean']) \
    .groupby(['location_clean', 'sentiment_label']) \
    .size().unstack().fillna(0)

# Ensure all sentiment columns exist
for sentiment in ['Positive', 'Neutral', 'Negative']:
    if sentiment not in loc_sent.columns:
        loc_sent[sentiment] = 0

if not loc_sent.empty:
    # Calculate sentiment percentages
    loc_sent['total'] = loc_sent[['Positive', 'Neutral', 'Negative']].sum(axis=1)
    loc_sent['positive_pct'] = loc_sent['Positive'] / loc_sent['total']
    loc_sent['neutral_pct'] = loc_sent['Neutral'] / loc_sent['total']
    loc_sent['negative_pct'] = loc_sent['Negative'] / loc_sent['total']

    # Let user choose sentiment to visualize
    st.subheader("🌍 Top 10 Countries by Sentiment %")
    selected_sentiment = st.selectbox("Choose sentiment", ['Positive', 'Neutral', 'Negative'])

    selected_pct = f"{selected_sentiment.lower()}_pct"
    color_scale = {
        "Positive": "Greens",
        "Neutral": "Blues",
        "Negative": "Reds"
    }

    top10 = loc_sent.sort_values(by=selected_pct, ascending=False).head(10)

    fig = px.bar(
        top10,
        x=selected_pct,
        y=top10.index,
        orientation='h',
        color=selected_pct,
        labels={selected_pct: f'{selected_sentiment} Sentiment %'},
        title=f"Top 10 Countries by % {selected_sentiment} Tweets",
        color_continuous_scale=color_scale[selected_sentiment]
    )
    st.plotly_chart(fig)

    # Choropleth map
    st.subheader(f"🗺️ Global {selected_sentiment} Sentiment Map")

    fig = px.choropleth(
        loc_sent,
        locations=loc_sent.index,
        locationmode='country names',
        color=selected_pct,
        hover_name=loc_sent.index,
        color_continuous_scale=color_scale[selected_sentiment],
        title=f'{selected_sentiment} Sentiment % by Country',
        labels={selected_pct: f'{selected_sentiment} Sentiment %'}
    )
    st.plotly_chart(fig)

else:
    st.warning("Not enough location data to display sentiment maps or charts.")

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

# --------------------------
# LDA Topic Modeling
# --------------------------

st.subheader("🔍 Discover Topics in Tweets (LDA)")

text_data_all = filtered_df['clean_text'].dropna()

if not text_data_all.empty:
    st.markdown("### 🛠️ LDA Debug Info")
    st.write("Filtered rows:", filtered_df.shape[0])
    st.write("Non-empty clean_text rows:", text_data_all.shape[0])
    st.dataframe(text_data_all.sample(5))

    # Sample data for LDA
    sample_size = min(2000, len(text_data_all))  # Cap at 2000 for Streamlit Cloud
    text_data = text_data_all.sample(sample_size, random_state=42)

    st.info(f"Using a sample of **{sample_size} tweets** for topic modeling.")

    n_topics = st.sidebar.slider("Number of Topics", 2, 10, 4)

    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    dtm = vectorizer.fit_transform(text_data)

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(dtm)

    words = vectorizer.get_feature_names_out()
    topics = []
    for idx, topic in enumerate(lda.components_):
        top_words = [words[i] for i in topic.argsort()[-10:]]
        topic_keywords = ", ".join(top_words[::-1])
        topic_label = f"🧠 Topic: {', '.join(top_words[::-1][:2])}"
        topics.append({"Topic": f"Topic {idx+1}", "Top Words": topic_keywords, "Topic_Label": topic_label})
    
    topics_df = pd.DataFrame(topics)

    st.subheader("📒 Topics Discovered")
    st.dataframe(topics_df[['Topic_Label', 'Top Words']])

    # Assign topics to tweets
    topic_probs = lda.transform(dtm)
    dominant_topics = topic_probs.argmax(axis=1)

    assigned_df = text_data.reset_index(drop=True).to_frame()
    assigned_df['Assigned_Topic'] = dominant_topics
    assigned_df['Assigned_Topic_Label'] = assigned_df['Assigned_Topic'].apply(
        lambda x: topics_df.iloc[x]['Topic_Label']
    )

    # Sidebar topic filter
    topic_options = ["All"] + sorted(assigned_df['Assigned_Topic_Label'].unique().tolist())
    selected_topic = st.sidebar.selectbox("Filter by Topic", topic_options)

    if selected_topic != "All":
        assigned_df = assigned_df[assigned_df['Assigned_Topic_Label'] == selected_topic]

    # Topic bar chart
    st.subheader("📊 Top Words in Selected Topic")
    topic_choice = st.selectbox("Select a Topic", topics_df['Topic_Label'])
    topic_idx = topics_df[topics_df['Topic_Label'] == topic_choice].index[0]

    top_words = [words[i] for i in lda.components_[topic_idx].argsort()[-10:]]
    word_weights = lda.components_[topic_idx][lda.components_[topic_idx].argsort()[-10:]]

    fig = px.bar(
        x=word_weights[::-1],
        y=top_words[::-1],
        orientation='h',
        title=f"Top Words in {topic_choice}"
    )
    st.plotly_chart(fig)

    # Sample tweets for the topic
    st.subheader("📄 Sample Tweets for Selected Topic")
    if not assigned_df.empty:
        st.dataframe(
            assigned_df[['clean_text', 'Assigned_Topic_Label']].sample(min(5, len(assigned_df)))
        )
    else:
        st.warning("No tweets found for this topic selection.")
else:
    st.warning("No tweets available for topic modeling.")
