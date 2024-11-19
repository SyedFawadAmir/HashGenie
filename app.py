import os
import streamlit as st
import whisper
import openai
from transformers import pipeline
import spacy
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from sklearn.feature_extraction.text import CountVectorizer

# Add FFmpeg path to system environment
os.environ["PATH"] += os.pathsep + r"C:\Users\Fawad\Downloads\ffmpeg-2024-11-13-git-322b240cea-full_build\bin"

# Set OpenAI API key (Removed mine you can use yours)
openai.api_key = 'YOUR_OPENAI_API_KEY'

# Load Whisper for audio transcription
model = whisper.load_model("base")

# Load sentiment analysis tool
sentiment_analyzer = pipeline("sentiment-analysis")

# Load spaCy model for keyword extraction
nlp = spacy.load("en_core_web_sm")

# Transcribe audio file
def transcribe_audio(file_path):
    result = model.transcribe(file_path)
    return result['text']

# Analyze sentiment from text
def analyze_sentiment(transcript):
    result = sentiment_analyzer(transcript)
    return result[0]['label'], result[0]['score']

# Extract keywords using spaCy
def extract_keywords(transcript, num_keywords=5):
    doc = nlp(transcript)
    return [chunk.text for chunk in doc.noun_chunks][:num_keywords]

# Identify topics using LDA
def categorize_topics(transcript, num_topics=3, num_words=3):
    dictionary = Dictionary([transcript.split()])
    corpus = [dictionary.doc2bow(transcript.split())]
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)
    return lda_model.print_topics(num_topics=num_topics, num_words=num_words)

# Streamlit page setup
st.set_page_config(page_title="HashGenie", page_icon="🎧", layout="centered")
st.markdown(
    """
    <style>
    .css-1cpxqw2 { background-color: #2d2d2d; color: #ffffff; }
    .stButton > button { background-color: #ff6347; color: white; border-radius: 10px; }
    .stHeader { font-size: 2.5rem; font-weight: bold; color: #ff6347; }
    .stSubtitle { font-size: 1.5rem; color: #add8e6; }
    .stTextArea textarea { background-color: #f5f5f5; border-radius: 5px; color: #000000; }
    .stCode { background-color: #f0f0f0; padding: 10px; border-radius: 5px; color: #000000; }
    </style>
    """,
    unsafe_allow_html=True
)

# App header
st.markdown("<h1 class='stHeader'>HashGenie</h1>", unsafe_allow_html=True)
st.markdown("<h3 class='stSubtitle'>Upload your audio or video and receive context-based hashtags generated from sentiment analysis, keyword extraction, and topic categorization!</h3>", unsafe_allow_html=True)

# File upload section
uploaded_file = st.file_uploader("Choose a file", type=['mp3', 'wav', 'mp4'])

if uploaded_file is not None:
    st.markdown(f"**File uploaded:** {uploaded_file.name}")
    with st.spinner("Transcribing the audio, please wait..."):
        with open("temp_audio_file", "wb") as f:
            f.write(uploaded_file.read())
        transcript = transcribe_audio("temp_audio_file")
    st.success("Transcription completed!")

    # Show the transcript
    st.subheader("Transcript")
    st.text_area("Transcribed Text", transcript, height=200)
    st.button("Copy Transcript", on_click=lambda: st.session_state.update({"_clipboard": transcript}))

    # Sentiment analysis display
    st.subheader("Sentiment Analysis")
    with st.spinner("Analyzing sentiment..."):
        sentiment, score = analyze_sentiment(transcript)
    st.write(f"The sentiment of the content is: **{sentiment}** (Confidence: {score:.2f})")

    # Extract keywords and identify topics (used internally)
    keywords = extract_keywords(transcript)
    topics = categorize_topics(transcript)

    # Generate hashtags with OpenAI API
    with st.spinner("Generating hashtags, please wait..."):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant that outputs hashtags only based on the given context."},
                {"role": "user", "content": f"Analyze this text and suggest relevant hashtags based on its context, keywords ({', '.join(keywords)}), and topics:\n\n{transcript}\n\nOutput only the hashtags."}
            ],
            max_tokens=50,
            temperature=0.5
        )

    hashtags = response['choices'][0]['message']['content'].strip()
    st.subheader("Generated Hashtags")
    st.code(hashtags, language="")

    # Copy button for hashtags
    st.button("Copy Hashtags", on_click=lambda: st.session_state.update({"_clipboard": hashtags}))
