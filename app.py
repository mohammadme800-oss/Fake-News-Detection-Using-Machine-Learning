import streamlit as st
import pickle
import os
import logging
import numpy as np
import re
import time

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models/naive_bayes_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models/tfidf_vectorizer.pkl")
MAX_INPUT_LENGTH = 5000

LABEL_MAP = {0: "Fake News", 1: "Real News"}

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Fake News AI Pro", layout="wide", page_icon="🧠")

# -----------------------------
# THEME TOGGLE
# -----------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

theme_toggle = st.toggle("🌗 Toggle Theme", value=(st.session_state.theme == "dark"))
st.session_state.theme = "dark" if theme_toggle else "light"

# -----------------------------
# CUSTOM CSS
# -----------------------------
if st.session_state.theme == "dark":
    bg = "linear-gradient(135deg, #141e30, #243b55)"
    text_color = "white"
else:
    bg = "linear-gradient(135deg, #f5f7fa, #c3cfe2)"
    text_color = "black"

st.markdown(f"""
<style>
.stApp {{
    background: {bg};
    color: {text_color};
}}

h1 {{text-align:center; font-size: 3em;}}

.stButton>button {{
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 12px;
    height: 3em;
    font-weight: bold;
    transition: 0.3s;
}}

.stButton>button:hover {{
    transform: scale(1.05);
}}

.stTextArea textarea {{
    border-radius: 12px;
}}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.title("🧠 Fake News Detection AI Pro")
st.caption("Next-gen AI system for misinformation detection 🚀")

st.warning("⚠️ AI prediction only. Always verify news from trusted sources.")

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_artifacts()

# -----------------------------
# SESSION STATE
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------
# CLEAN TEXT
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# -----------------------------
# INPUT
# -----------------------------
st.subheader("📝 Analyze News Article")

user_input = st.text_area("Paste news content", height=200)

if len(user_input) > MAX_INPUT_LENGTH:
    st.error("Input too long")
    st.stop()

if len(user_input.strip()) > 0 and len(user_input.strip()) < 20:
    st.warning("Input too short")
    st.stop()

col1, col2, col3 = st.columns(3)
predict_btn = col1.button("🔍 Predict", use_container_width=True)
clear_btn = col2.button("🧹 Clear", use_container_width=True)
sample_btn = col3.button("✨ Sample", use_container_width=True)

if sample_btn:
    st.session_state.sample = "Breaking: Government announces major economic reforms impacting global markets..."
    st.rerun()

if clear_btn:
    st.session_state.history = []

# -----------------------------
# PREDICTION
# -----------------------------
@st.cache_data
def predict(text):
    X = vectorizer.transform([clean_text(text)])
    pred = model.predict(X)[0]
    probs = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else None
    confidence = float(np.max(probs)) * 100 if probs is not None else None
    return pred, confidence, probs, X

# -----------------------------
# ANIMATION FUNCTION
# -----------------------------
def animate_text(text):
    placeholder = st.empty()
    displayed = ""
    for char in text:
        displayed += char
        placeholder.markdown(f"## {displayed}")
        time.sleep(0.02)

# -----------------------------
# RUN
# -----------------------------
if predict_btn and user_input.strip():
    with st.spinner("Analyzing with AI..."):
        pred, confidence, probs, X = predict(user_input)
        result = LABEL_MAP[pred]

        st.session_state.history.append((result, confidence))

        st.divider()

        # Animated Result
        if result == "Real News":
            animate_text("🟢 REAL NEWS")
        else:
            animate_text("🔴 FAKE NEWS")

        # Confidence Bar
        if confidence:
            st.progress(int(confidence))
            st.write(f"Confidence: {confidence:.2f}%")

        # Probability Chart
        if probs is not None:
            st.subheader("📊 Prediction Breakdown")
            st.bar_chart({LABEL_MAP[i]: probs[i] for i in range(len(probs))})

        # Explanation
        with st.expander("🧠 AI Explanation"):
            feature_names = vectorizer.get_feature_names_out()
            values = X.toarray()[0]
            top_idx = np.argsort(values)[-10:][::-1]
            for i in top_idx:
                if values[i] > 0:
                    st.write(f"• {feature_names[i]} ({values[i]:.4f})")

        # Download
        st.download_button("📥 Download Result", data=f"{result} ({confidence})")

# -----------------------------
# HISTORY
# -----------------------------
st.subheader("📜 History")

if st.session_state.history:
    for res, conf in reversed(st.session_state.history[-5:]):
        st.write(f"{res} - {conf:.1f}%" if conf else f"{res} - N/A")
else:
    st.info("No history yet")

# -----------------------------
# FOOTER
# -----------------------------
st.caption("Built with ❤️ using Streamlit | AI Model: Naive Bayes")