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
# Additional imports for Whisper fallback
import xml.etree.ElementTree as ET
import tempfile
import openai
import os
import time
import subprocess
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


# Set OpenAI key
openai.api_key = st.secrets["openai"]["api_key"]


def fetch_transcript(video_id, target_lang="auto", use_whisper=True, use_ytdlp=True):
    """
    Fetch transcript in this order:
    1. YouTubeTranscriptApi (with proxy rotation)
    2. Whisper fallback (audio transcription)
    3. yt-dlp auto-subtitles fallback (if installed)
    Returns transcript text or None.
    """
    st.write("⏳ Fetching transcript...")
    transcript_text = None

    # Proxy list
    proxy_list = list(st.secrets.get("youtube_proxies", {}).values()) if "youtube_proxies" in st.secrets else [None]
    

    # ---------- 1. YouTubeTranscriptApi with proxy rotation ----------
    for proxy_url in random.sample(proxy_list, len(proxy_list)):
        try:
            st.write(f"Trying YouTubeTranscriptApi with proxy")
            if proxy_url:
                session = requests.Session()
                session.proxies.update({"http": proxy_url, "https": proxy_url})
                from youtube_transcript_api import _api
                _api.requests = session  # patch requests

            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

            # Case 1: target language auto → just grab English (base transcript)
            if target_lang == "auto":
                base = transcript_list.find_generated_transcript(['en'])
                transcript = base.fetch()

            # Case 2: explicit Hindi or other translation
            elif target_lang == "hi":
                base = transcript_list.find_generated_transcript(['en'])
                transcript = base.translate('hi').fetch()

            # Case 3: explicit English
            elif target_lang == "en":
                base = transcript_list.find_generated_transcript(['en'])
                transcript = base.fetch()

            # Fallback: try direct fetch in requested language
            else:
                transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[target_lang])

            if transcript:
                transcript_text = " ".join([t["text"] for t in transcript if t["text"].strip()])
                st.success(f"✅ Transcript fetched in {target_lang}")
                return transcript_text

        except Exception as e:
            st.warning(f"Transcript API failed with: {e}")
            time.sleep(1)
    # ---------- 2. Whisper fallback ----------
    if use_whisper:
        try:
            st.write("🎤 Trying Whisper transcription (audio fallback)...")
            with tempfile.TemporaryDirectory() as tmp_dir:
                yt = YouTube(f"https://www.youtube.com/watch?v={video_id}",
                             proxies={"http": proxy_list[0], "https": proxy_list[0]} if proxy_list[0] else None)
                audio_file = yt.streams.filter(only_audio=True).first().download(
                    output_path=tmp_dir, filename="video_audio.mp4"
                )

                # Retry on rate limit
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        with open(audio_file, "rb") as f:
                            transcript = openai.audio.transcriptions.create(
                                model="whisper-1",
                                file=f
                            )
                        transcript_text = transcript["text"]
                        st.success("✅ Transcript generated via Whisper")
                        return transcript_text
                    except openai.error.RateLimitError:
                        wait = 2 ** attempt
                        st.warning(f"Rate limited by Whisper, retrying in {wait}s...")
                        time.sleep(wait)
        except Exception as e:
            st.error(f"Whisper transcription failed: {e}")

    # ---------- 3. yt-dlp fallback ----------
    if use_ytdlp:
        try:
            st.write("📝 Trying yt-dlp auto-captions fallback...")
            with tempfile.TemporaryDirectory() as tmp_dir:
                output_path = os.path.join(tmp_dir, "subs")
                cmd = [
                    "yt-dlp",
                    "--skip-download",
                    "--write-auto-subs",
                    "--sub-lang", "en",
                    "-o", output_path,
                    f"https://www.youtube.com/watch?v={video_id}"
                ]
                subprocess.run(cmd, capture_output=True)
                
                # Look for downloaded .vtt file
                for f in os.listdir(tmp_dir):
                    if f.endswith(".vtt"):
                        vtt_file = os.path.join(tmp_dir, f)
                        lines = []
                        with open(vtt_file, "r", encoding="utf-8") as vf:
                            for line in vf:
                                if line.strip() and not line[0].isdigit() and "-->" not in line:
                                    lines.append(line.strip())
                        if lines:
                            transcript_text = " ".join(lines)
                            st.success("✅ Transcript fetched via yt-dlp auto-captions")
                            return transcript_text
        except Exception as e:
            st.warning(f"yt-dlp fallback failed: {e}")

    st.error("❌ Could not retrieve transcript by any method.")
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
                    model="gemini-2.5-flash",
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





  

















