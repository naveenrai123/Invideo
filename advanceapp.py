
"""import streamlit as st
from googleapiclient.discovery import build
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
from collections import Counter
from transformers import pipeline

# Load custom model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
#hf_pipeline = pipeline("sentiment-analysis")  # Hugging Face model
hf_pipeline = pipeline("sentiment-analysis", device=-1)  # Force CPU
# ------------------- Utility Functions ------------------- #

def clean_text(text):
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

def extract_video_id(url):
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    return None

def get_youtube_comments(api_key, video_id, max_results=100):
    youtube = build('youtube', 'v3', developerKey=api_key)
    comments = []
    request = youtube.commentThreads().list(
        part='snippet', videoId=video_id,
        maxResults=max_results, textFormat='plainText'
    )
    response = request.execute()
    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
        comments.append(comment)
    return comments

def compare_sentiments(comments):
    results = []
    for comment in comments:
        clean = clean_text(comment)
        vector = vectorizer.transform([clean])
        custom_pred = model.predict(vector)[0]
        custom_label = 'Positive' if custom_pred == 1 else 'Negative'
        hf_result = hf_pipeline(comment[:512])[0]
        hf_label = 'Positive' if 'POS' in hf_result['label'].upper() else 'Negative'
        results.append({
            "Comment": comment,
            "Custom Model": custom_label,
            "HF Model": hf_label
        })
    return pd.DataFrame(results)


def plot_top_tfidf_words(vectorizer, model, top_n=20):
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_[0]
    top_pos = np.argsort(coefs)[-top_n:]
    top_neg = np.argsort(coefs)[:top_n]
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top Positive Words**")
        st.write(feature_names[top_pos])
    with col2:
        st.markdown("**Top Negative Words**")
        st.write(feature_names[top_neg])

# ------------------- Streamlit App ------------------- #

st.set_page_config(page_title="YouTube Sentiment Analyzer", layout="wide")
st.title("🎬 Sentiment Analyzer (IMDb Model + Hugging Face)")

# Sidebar info
with st.sidebar:
    st.header("ℹ️ Model Info")
    st.markdown("""
    - **Trained On:** IMDb Reviews  
    - **Model:** Logistic Regression  
    - **Vectorizer:** TF-IDF (1-2 grams)  
    - **Tuned with:** GridSearchCV  
    - **Compared With:** distilBERT (HF)  
    """)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📺 YouTube Sentiment Comparison",
    "🧪 Try Custom Review",
    "📊 Word Importance",
    "📂 Bulk Review Upload"
])

# ------------------- Tab 1: YouTube Sentiment Comparison ------------------- #
with tab1:
    st.subheader("📺 Analyze YouTube Video Comments")

    video_url = st.text_input("Enter YouTube Video URL:")
    max_results = st.slider("Number of Comments", 50, 250, 100)

  
   
    api_key = st.secrets["youtube"]["api_key"]
  

    if st.button("Compare Models"):
        if not api_key or not video_url:
            st.error("Please provide both API key and video URL.")
        else:
            with st.spinner("Fetching and analyzing comments..."):
                try:
                    video_id = extract_video_id(video_url)
                    comments = get_youtube_comments(api_key, video_id, max_results)
                    df_compare = compare_sentiments(comments)

                    st.subheader("📝 Results")
                    st.dataframe(df_compare)

                    df_compare["Match"] = df_compare["Custom Model"] == df_compare["HF Model"]
                    match_rate = df_compare["Match"].mean() * 100
                    st.success(f"✅ Agreement: {match_rate:.2f}%")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Custom Model**")
                        fig1, ax1 = plt.subplots()
                        df_compare["Custom Model"].value_counts().plot(kind="bar", color="skyblue", ax=ax1)
                        ax1.set_ylabel("Count")
                        st.pyplot(fig1)

                    with col2:
                        st.markdown("**Hugging Face Model**")
                        fig2, ax2 = plt.subplots()
                        df_compare["HF Model"].value_counts().plot(kind="bar", color="orange", ax=ax2)
                        ax2.set_ylabel("Count")
                        st.pyplot(fig2)

                except Exception as e:
                    st.error(f"Error: {e}")

# ------------------- Tab 2: Custom IMDb-Style Review ------------------- #
with tab2:
    st.subheader("🧪 Try Your Own Review (IMDb-style)")
    user_review = st.text_area("Enter a movie review:")
    if user_review:
        cleaned = clean_text(user_review)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]
        st.info(f"Prediction: {'Positive' if pred == 1 else 'Negative'}")
        st.markdown(f"Confidence: `{max(proba):.2f}`")

# ------------------- Tab 3: Word Importance ------------------- #
with tab3:
    st.subheader("📊 Top Influential Words from IMDb Model")
    plot_top_tfidf_words(vectorizer, model, top_n=20)

# ------------------- Tab 4: Bulk Review Upload ------------------- #
with tab4:
    st.subheader("📂 Upload a CSV of Reviews")
    uploaded_file = st.file_uploader("Upload CSV with column `review`", type="csv")
    if uploaded_file:
        df_upload = pd.read_csv(uploaded_file)
        if "review" in df_upload.columns:
            df_upload['clean'] = df_upload['review'].apply(clean_text)
            vectors = vectorizer.transform(df_upload['clean'])
            df_upload['Prediction'] = model.predict(vectors)
            df_upload['Sentiment'] = df_upload['Prediction'].apply(lambda x: 'Positive' if x == 1 else 'Negative')
            st.dataframe(df_upload[['review', 'Sentiment']])
        else:
            st.error("CSV must have a 'review' column.")
    """
import streamlit as st
from googleapiclient.discovery import build
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
from transformers import pipeline

# Load custom model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
hf_pipeline = pipeline("sentiment-analysis", device=-1)  # Use CPU

# ------------------- Utility Functions ------------------- #

def clean_text(text):
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

def extract_video_id(url):
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    return None

def get_youtube_comments(api_key, video_id, max_results=100):
    youtube = build('youtube', 'v3', developerKey=api_key)
    comments = []
    request = youtube.commentThreads().list(
        part='snippet', videoId=video_id,
        maxResults=max_results, textFormat='plainText'
    )
    response = request.execute()
    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
        comments.append(comment)
    return comments

def compare_sentiments(comments):
    results = []
    for comment in comments:
        clean = clean_text(comment)
        vector = vectorizer.transform([clean])
        custom_pred = model.predict(vector)[0]
        custom_label = 'Positive' if custom_pred == 1 else 'Negative'

        hf_result = hf_pipeline(comment[:512])[0]
        hf_label = hf_result['label'].capitalize()  # Handles 'POSITIVE', 'LABEL_1', etc.

        if "NEG" in hf_label.upper():
            hf_label = "Negative"
        elif "POS" in hf_label.upper():
            hf_label = "Positive"
        elif "NEU" in hf_label.upper():
            hf_label = "Neutral"

        results.append({
            "Comment": comment,
            "Custom Model": custom_label,
            "HF Model": hf_label
        })
    return pd.DataFrame(results)

def plot_top_tfidf_words(vectorizer, model, top_n=20):
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_[0]
    top_pos = np.argsort(coefs)[-top_n:]
    top_neg = np.argsort(coefs)[:top_n]
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top Positive Words**")
        st.write(feature_names[top_pos])
    with col2:
        st.markdown("**Top Negative Words**")
        st.write(feature_names[top_neg])

# ------------------- Streamlit App ------------------- #

st.set_page_config(page_title="YouTube Sentiment Analyzer", layout="wide")
st.title("🎬 Sentiment Analyzer (IMDb Model + Hugging Face)")

# Sidebar info
with st.sidebar:
    st.header("ℹ️ Model Info")
    st.markdown("""
    - **Trained On:** IMDb Reviews  
    - **Model:** Logistic Regression  
    - **Vectorizer:** TF-IDF (1-2 grams)  
    - **Tuned with:** GridSearchCV  
    - **Compared With:** distilBERT (HF)  
    """)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📺 YouTube Sentiment Comparison",
    "🧪 Try Custom Review",
    "📊 Word Importance",
    "📂 Bulk Review Upload"
])

# ------------------- Tab 1: YouTube Sentiment Comparison ------------------- #
with tab1:
    st.subheader("📺 Analyze YouTube Video Comments")
    video_url = st.text_input("Enter YouTube Video URL:")
    max_results = st.slider("Number of Comments", 50, 250, 100)

    api_key = st.secrets.get("youtube", {}).get("api_key")

    if st.button("Compare Models"):
        if not api_key:
            st.error("❗ API key not found in secrets.")
        elif not video_url:
            st.error("❗ Please enter a valid YouTube video URL.")
        else:
            with st.spinner("Fetching and analyzing comments..."):
                try:
                    video_id = extract_video_id(video_url)
                    comments = get_youtube_comments(api_key, video_id, max_results)

                    if not comments:
                        st.warning("No comments found.")
                    else:
                        df_compare = compare_sentiments(comments)
                        st.subheader("📝 Results")
                        st.dataframe(df_compare)

                        df_compare["Match"] = df_compare["Custom Model"] == df_compare["HF Model"]
                        match_rate = df_compare["Match"].mean() * 100
                        st.success(f"✅ Agreement Rate: {match_rate:.2f}%")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Custom Model**")
                            fig1, ax1 = plt.subplots()
                            df_compare["Custom Model"].value_counts().plot(kind="bar", color="skyblue", ax=ax1)
                            ax1.set_ylabel("Count")
                            st.pyplot(fig1)

                        with col2:
                            st.markdown("**Hugging Face Model**")
                            fig2, ax2 = plt.subplots()
                            df_compare["HF Model"].value_counts().plot(kind="bar", color="orange", ax=ax2)
                            ax2.set_ylabel("Count")
                            st.pyplot(fig2)

                except Exception as e:
                    st.error(f"❌ Error: {e}")

# ------------------- Tab 2: Custom IMDb-Style Review ------------------- #
with tab2:
    st.subheader("🧪 Try Your Own Review (IMDb-style)")
    user_review = st.text_area("Enter a movie review:")
    if user_review:
        cleaned = clean_text(user_review)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]
        st.info(f"Prediction: {'Positive' if pred == 1 else 'Negative'}")
        st.markdown(f"Confidence: `{max(proba):.2f}`")

# ------------------- Tab 3: Word Importance ------------------- #
with tab3:
    st.subheader("📊 Top Influential Words from IMDb Model")
    plot_top_tfidf_words(vectorizer, model, top_n=20)

# ------------------- Tab 4: Bulk Review Upload ------------------- #
with tab4:
    st.subheader("📂 Upload a CSV of Reviews")
    uploaded_file = st.file_uploader("Upload CSV with column `review`", type="csv")
    if uploaded_file:
        try:
            df_upload = pd.read_csv(uploaded_file)
            if "review" in df_upload.columns:
                df_upload['clean'] = df_upload['review'].apply(clean_text)
                vectors = vectorizer.transform(df_upload['clean'])
                df_upload['Prediction'] = model.predict(vectors)
                df_upload['Sentiment'] = df_upload['Prediction'].apply(lambda x: 'Positive' if x == 1 else 'Negative')
                st.dataframe(df_upload[['review', 'Sentiment']])
            else:
                st.error("CSV must have a column named 'review'.")
        except Exception as e:
            st.error(f"Error reading file: {e}")

