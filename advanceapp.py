import streamlit as st
from googleapiclient.discovery import build
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
from collections import Counter
import random
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from pytube import YouTube   # Added for fallback

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.docstore.document import Document
import requests

# ------------------- Load custom model and vectorizer ------------------- #
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# ------------------- Utility Functions ------------------- #
st.set_page_config(page_title="Invideo", layout="wide")

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

def analyze_sentiments(comments):
    results = []
    for comment in comments:
        clean = clean_text(comment)
        vector = vectorizer.transform([clean])
        custom_pred = model.predict(vector)[0]
        custom_label = 'Positive' if custom_pred == 1 else 'Negative'
        results.append({
            "Comment": comment,
            "Custom Model": custom_label
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

# ------------------- YouTube Summarizer ------------------- #
def fetch_transcript(video_id, target_lang="auto"):
    """
    Try YouTubeTranscriptApi first.
    If fails, fall back to Pytube captions.
    """
    transcript = None

    proxy_list = list(st.secrets.get("youtube_proxies", {}).values())

    # ---------- Try YouTubeTranscriptApi ----------
    for proxy_url in proxy_list + [None]:
        try:
            if proxy_url:
                from youtube_transcript_api._transcripts import Transcripts
                session = requests.Session()
                session.proxies.update({"http": proxy_url, "https": proxy_url})
                transcript_list = Transcripts(video_id, session=session).find_transcript(['en', 'hi'])
                transcript = transcript_list.fetch()
            else:
                languages = ['en', 'hi'] if target_lang == "auto" else [target_lang]
                transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)

            if transcript:
                return " ".join([t['text'] for t in transcript])

        except (TranscriptsDisabled, NoTranscriptFound):
            continue
        except Exception as e:
            print(f"Transcript API error: {e}")
            continue

    # ---------- Fallback: Pytube ----------
    try:
        yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")

        if not yt.captions:
            print("No captions available in Pytube.")
            return None

        fallback_order = []
        if target_lang == "auto":
            fallback_order = ["en", "hi"]
        else:
            fallback_order = [target_lang, "en", "hi"]

        caption = None
        for lang in fallback_order:
            caption = yt.captions.get_by_language_code(lang)
            if caption:
                break

        if not caption:
            caption = next(iter(yt.captions.values()), None)

        if caption:
            srt_captions = caption.generate_srt_captions()
            cleaned_lines = []
            for line in srt_captions.split("\n"):
                if re.match(r"^\d+$", line):
                    continue
                if re.match(r"^\d{2}:\d{2}:\d{2},\d{3}", line):
                    continue
                if line.strip():
                    cleaned_lines.append(line.strip())
            return " ".join(cleaned_lines)

    except Exception as e:
        print("Pytube failed:", e)

    return None
def fetch_transcript(video_id, target_lang="auto"):
    """
    Try YouTubeTranscriptApi first.
    If fails, fall back to Pytube captions.
    """
    transcript = None

    proxy_list = list(st.secrets.get("youtube_proxies", {}).values())

    # ---------- Try YouTubeTranscriptApi ----------
    for proxy_url in proxy_list + [None]:
        try:
            if proxy_url:
                from youtube_transcript_api._transcripts import Transcripts
                session = requests.Session()
                session.proxies.update({"http": proxy_url, "https": proxy_url})
                transcript_list = Transcripts(video_id, session=session).find_transcript(['en', 'hi'])
                transcript = transcript_list.fetch()
            else:
                languages = ['en', 'hi'] if target_lang == "auto" else [target_lang]
                transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)

            if transcript:
                return " ".join([t['text'] for t in transcript])

        except (TranscriptsDisabled, NoTranscriptFound):
            continue
        except Exception as e:
            print(f"Transcript API error: {e}")
            continue

    # ---------- Fallback: Pytube ----------
    try:
        yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")

        if not yt.captions:
            print("No captions available in Pytube.")
            return None

        fallback_order = []
        if target_lang == "auto":
            fallback_order = ["en", "hi"]
        else:
            fallback_order = [target_lang, "en", "hi"]

        caption = None
        for lang in fallback_order:
            caption = yt.captions.get_by_language_code(lang)
            if caption:
                break

        if not caption:
            caption = next(iter(yt.captions.values()), None)

        if caption:
            srt_captions = caption.generate_srt_captions()
            cleaned_lines = []
            for line in srt_captions.split("\n"):
                if re.match(r"^\d+$", line):
                    continue
                if re.match(r"^\d{2}:\d{2}:\d{2},\d{3}", line):
                    continue
                if line.strip():
                    cleaned_lines.append(line.strip())
            return " ".join(cleaned_lines)

    except Exception as e:
        print("Pytube failed:", e)

    return None


def summarize_youtube_video(url, llm, target_lang="auto"):
    try:
        video_id = extract_video_id(url)
        if not video_id:
            return " Could not extract a valid video ID."

        text = fetch_transcript(video_id, target_lang)

        if not text:
            return " Could not retrieve a transcript or captions."

        # LangChain summarization
        docs = [Document(page_content=text)]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(docs)

        if target_lang == "hi":
            instruction = "इस वीडियो ट्रांसक्रिप्ट का संक्षेप हिंदी में लिखिए। अगर मूल पाठ अंग्रेज़ी में है तो पहले अनुवाद कर संक्षेप हिंदी में लिखें।"
        elif target_lang == "en":
            instruction = "Summarize this video transcript in English."
        else:
            instruction = "Summarize this video transcript in its original language."

        prompt_template = PromptTemplate(
            template=f"""{instruction}

Transcript:
{{text}}

Summary:""",
            input_variables=["text"]
        )

        chain = LLMChain(llm=llm, prompt=prompt_template)
        combined_text = " ".join([d.page_content for d in split_docs])
        summary = chain.run({"text": combined_text})

        return summary
    except Exception as e:
        return f"⚠️ Error while summarizing: {e}"

# ------------------- Streamlit App ------------------- #
st.title("🎬 INVIDEO Analyzer")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📝 YouTube Video Summarizer",
    "📺 YouTube Sentiment Analysis",
    "🧪 Try Custom Review",
    "📊 Word Importance",
    "📂 Bulk Review Upload",
])

# ------------------- Tab 1: YouTube Comments Analysis ------------------- #
with tab2:
    st.subheader("📺 Analyze YouTube Video Comments")
    video_url = st.text_input("Enter YouTube Video URL:")
    max_results = st.slider("Number of Comments", 50, 250, 100)
    api_key = st.secrets["youtube"]["api_key"]

    if st.button("Analyze Comments"):
        if not api_key or not video_url:
            st.error("Please provide both API key and video URL.")
        else:
            with st.spinner("Fetching and analyzing comments..."):
                try:
                    video_id = extract_video_id(video_url)
                    comments = get_youtube_comments(api_key, video_id, max_results)
                    df_results = analyze_sentiments(comments)
                    st.subheader("📝 Results")
                    st.dataframe(df_results)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Custom Model Sentiment Distribution**")
                        fig, ax = plt.subplots()
                        df_results["Custom Model"].value_counts().plot(kind="bar", color="skyblue", ax=ax)
                        ax.set_ylabel("Count")
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error: {e}")

# ------------------- Tab 2: Custom Review ------------------- #
with tab3:
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
with tab4:
    st.subheader("📊 Top Influential Words from IMDb Model")
    plot_top_tfidf_words(vectorizer, model, top_n=20)

# ------------------- Tab 4: Bulk Review Upload ------------------- #
with tab5:
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

# ------------------- Tab 5: YouTube Summarizer ------------------- #
with tab1:
    st.subheader("📝 Summarize YouTube Video")
    video_url_sum = st.text_input("Enter YouTube Video URL for summarization:")
    lang_choice = st.radio(
        "Select summary language:",
        ["Auto ", "English", "Hindi"],
        index=0,
        horizontal=True
    )

    if st.button("Summarize Video"):
        if not video_url_sum:
            st.error("Please enter a valid YouTube URL")
        else:
            with st.spinner("Generating summary..."):
                llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    google_api_key=st.secrets["google"]["api_key"],
                    temperature=0
                )
                if lang_choice == "English":
                    lang_code = "en"
                elif lang_choice == "Hindi":
                    lang_code = "hi"
                else:
                    lang_code = "auto"

                summary = summarize_youtube_video(video_url_sum, llm, target_lang=lang_code)
                st.success("✅ Summary Generated!")
                st.write(summary)





  why it is not using pytube
