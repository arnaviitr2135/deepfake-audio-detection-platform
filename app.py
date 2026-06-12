from __future__ import annotations

import io
import sys
from pathlib import Path

import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deepfake_audio_detector.audio_features import extract_features_from_wav


MODEL_PATH = ROOT / "models" / "deepfake_audio_model.joblib"


def decision_confidence(fake_score: float, threshold: float) -> float:
    if fake_score >= threshold:
        return 0.5 + 0.5 * ((fake_score - threshold) / max(1.0 - threshold, 1e-8))
    return 0.5 + 0.5 * ((threshold - fake_score) / max(threshold, 1e-8))


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="Deepfake Audio Detection", layout="centered")
st.title("Deepfake Audio Detection")

if not MODEL_PATH.exists():
    st.error("Model not found. Train it first with: python scripts/train.py")
    st.stop()

uploaded = st.file_uploader("Upload a WAV audio file", type=["wav"])
if uploaded is None:
    st.info("Upload a WAV file to classify it as Genuine or Deepfake.")
    st.stop()

st.audio(uploaded, format="audio/wav")

try:
    package = load_model()
    vector, _ = extract_features_from_wav(io.BytesIO(uploaded.getvalue()))
    fake_probability = float(package["model"].predict_proba([vector])[0, 1])
    threshold = float(package.get("threshold", 0.5))
    is_fake = fake_probability >= threshold
    label = "Deepfake (AI-Generated)" if is_fake else "Genuine (Human)"
    confidence = decision_confidence(fake_probability, threshold)
except Exception as exc:
    st.error(f"Could not analyze this file: {exc}")
    st.stop()

st.metric("Prediction", label)
st.metric("Confidence", f"{confidence * 100:.1f}%")
st.progress(fake_probability, text=f"Deepfake probability: {fake_probability * 100:.1f}%")

with st.expander("Model details"):
    st.write(f"Decision threshold: `{threshold:.4f}`")
    st.write(f"Model file: `{MODEL_PATH}`")
