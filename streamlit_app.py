"""
GoEmotions emotion-detection demo (Streamlit Community Cloud).
Loads the fine-tuned DeBERTa-v3 + Asymmetric Loss model from the Hugging Face
Hub and applies the per-class thresholds tuned during training.
"""
import json
import numpy as np
import torch
import streamlit as st
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification

REPO = "dev-abdullah-0909/goemotions-deberta-v3-asl"

st.set_page_config(page_title="Emotion Detector", page_icon="🎭", layout="centered")

@st.cache_resource(show_spinner="Loading model (first run only)...")
def load():
    tok = AutoTokenizer.from_pretrained(REPO)
    mdl = AutoModelForSequenceClassification.from_pretrained(REPO)
    mdl.eval()
    with open(hf_hub_download(REPO, "thresholds.json")) as f:
        cfg = json.load(f)
    return tok, mdl, cfg["labels"], np.array(cfg["thresholds"], dtype=np.float32)

tokenizer, model, LABELS, THRESHOLDS = load()

def predict(text):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        logits = model(**enc).logits[0]
    return torch.sigmoid(logits).numpy()

EXAMPLES = [
    "I'm so grateful for your help, but honestly I'm really nervous about tomorrow.",
    "This is genuinely the best news I've heard all year!",
    "I can't believe they cancelled it again after I rearranged everything.",
    "Wait... it actually worked? I did not expect that at all.",
    "I feel terrible about what I said to her yesterday.",
]

st.title("🎭 Fine-Grained Emotion Detector")
st.markdown(
    "Type a sentence and the model predicts which of **28 emotions** it expresses "
    "(it can detect several at once). Built on **DeBERTa-v3**, fine-tuned on "
    "**GoEmotions** with an imbalance-aware loss so it also catches rarer emotions."
)

if "text" not in st.session_state:
    st.session_state.text = EXAMPLES[0]

st.caption("Try an example:")
cols = st.columns(len(EXAMPLES))
for i, ex in enumerate(EXAMPLES):
    if cols[i].button(f"Ex {i+1}", help=ex):
        st.session_state.text = ex

text = st.text_area("Your text", value=st.session_state.text, height=100)

if st.button("Detect emotions", type="primary"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        probs = predict(text)
        firing = sorted(
            [(LABELS[i], float(probs[i])) for i in range(len(LABELS))
             if probs[i] >= THRESHOLDS[i]],
            key=lambda x: x[1], reverse=True,
        )
        if firing:
            st.subheader("Detected emotions")
            for name, p in firing:
                st.write(f"**{name}** — {p*100:.0f}%")
                st.progress(min(p, 1.0))
        else:
            top = int(probs.argmax())
            st.info(
                f"No emotion cleared its confidence threshold. "
                f"Closest was **{LABELS[top]}** ({probs[top]*100:.0f}%)."
            )

with st.expander("How it works & honest notes"):
    st.markdown(
        "Each emotion is scored independently; an emotion is shown only if its "
        "probability clears a threshold tuned per-emotion on validation data.\n\n"
        "GoEmotions is subjective (annotators disagree a lot) and very imbalanced, "
        "so some rare emotions (e.g. *grief*) are hard to detect reliably. This "
        "model reaches ~0.53 Macro-F1; the gains over a plain baseline come from "
        "an Asymmetric Loss that specifically targets the rare classes.\n\n"
        f"Model: https://huggingface.co/{REPO}"
    )
