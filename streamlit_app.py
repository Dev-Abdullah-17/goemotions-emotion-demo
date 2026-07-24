"""
GoEmotions emotion-detection demo (Streamlit Community Cloud) — styled UI.
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

# ---------------------------------------------------------------- styling ----
st.markdown("""
<style>
:root { --pos:#2FA96B; --neg:#E4572E; --amb:#8367C7; --neu:#8A94A6; }
.block-container { max-width: 760px; padding-top: 2.2rem; }
#MainMenu, footer, header { visibility: hidden; }

.hero { text-align:center; margin-bottom: 1.4rem; }
.hero h1 { font-size: 2.15rem; font-weight: 800; letter-spacing:-0.02em; margin:0; }
.hero p  { color:#6b7280; font-size:1.02rem; margin:.4rem auto 0; max-width:520px; line-height:1.5;}

.legend { display:flex; gap:1.1rem; justify-content:center; margin:.6rem 0 1.3rem;
          font-size:.8rem; color:#6b7280; flex-wrap:wrap;}
.dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:.35rem; vertical-align:middle;}

.emo-row { margin: .55rem 0; }
.emo-head { display:flex; justify-content:space-between; align-items:baseline;
            font-size:.95rem; margin-bottom:.28rem; }
.emo-name { font-weight:650; text-transform:capitalize; color:#1f2937; }
.emo-pct  { font-variant-numeric:tabular-nums; color:#6b7280; font-size:.85rem; }
.emo-track{ background:#eef1f5; border-radius:999px; height:11px; overflow:hidden; }
.emo-fill { height:100%; border-radius:999px; transition:width .5s ease; }
.faded .emo-name, .faded .emo-pct { color:#9aa2ad; }
.faded .emo-fill { opacity:.45; }

div.stButton > button {
    border-radius:999px; border:1px solid #e5e7eb; background:#fff;
    padding:.28rem .8rem; font-size:.8rem; color:#374151; font-weight:500;
}
div.stButton > button:hover { border-color:#c7cdd6; background:#fafbfc; }
</style>
""", unsafe_allow_html=True)

# --- GoEmotions sentiment grouping (from the dataset paper) ---
GROUP = {
    "positive": {"admiration","amusement","approval","caring","desire","excitement",
                 "gratitude","joy","love","optimism","pride","relief"},
    "negative": {"anger","annoyance","disappointment","disapproval","disgust",
                 "embarrassment","fear","grief","nervousness","remorse","sadness"},
    "ambiguous": {"confusion","curiosity","realization","surprise"},
    "neutral": {"neutral"},
}
COLOR = {"positive":"var(--pos)","negative":"var(--neg)",
         "ambiguous":"var(--amb)","neutral":"var(--neu)"}

def color_for(name):
    for g, members in GROUP.items():
        if name in members:
            return COLOR[g]
    return COLOR["neutral"]

def bar(name, prob, faded=False):
    pct = prob * 100
    cls = "emo-row faded" if faded else "emo-row"
    return (f'<div class="{cls}"><div class="emo-head">'
            f'<span class="emo-name">{name}</span>'
            f'<span class="emo-pct">{pct:.0f}%</span></div>'
            f'<div class="emo-track"><div class="emo-fill" '
            f'style="width:{pct:.0f}%;background:{color_for(name)};"></div></div></div>')

# ---------------------------------------------------------------- model ----
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

# ---------------------------------------------------------------- header ----
st.markdown("""
<div class="hero">
  <h1>🎭 Emotion Detector</h1>
  <p>Type a sentence and the model predicts which of <b>28 emotions</b> it expresses —
     it can detect several at once. Built on DeBERTa-v3, fine-tuned on GoEmotions.</p>
</div>
<div class="legend">
  <span><span class="dot" style="background:#2FA96B"></span>Positive</span>
  <span><span class="dot" style="background:#E4572E"></span>Negative</span>
  <span><span class="dot" style="background:#8367C7"></span>Ambiguous</span>
  <span><span class="dot" style="background:#8A94A6"></span>Neutral</span>
</div>
""", unsafe_allow_html=True)

if "text" not in st.session_state:
    st.session_state.text = EXAMPLES[0]

st.caption("Try an example:")
cols = st.columns(len(EXAMPLES))
for i, ex in enumerate(EXAMPLES):
    if cols[i].button(f"Example {i+1}", help=ex, use_container_width=True):
        st.session_state.text = ex

text = st.text_area("Your text", value=st.session_state.text, height=110,
                    label_visibility="collapsed")

if st.button("Detect emotions", type="primary", use_container_width=True):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        probs = predict(text)
        scored = sorted([(LABELS[i], float(probs[i]), probs[i] >= THRESHOLDS[i])
                         for i in range(len(LABELS))], key=lambda x: x[1], reverse=True)
        firing = [s for s in scored if s[2]]
        near   = [s for s in scored if not s[2]][:3]  # top near-misses

        if firing:
            st.markdown("#### Detected emotions")
            st.markdown("".join(bar(n, p) for n, p, _ in firing), unsafe_allow_html=True)
        else:
            st.info("No emotion cleared its confidence threshold — see the closest matches below.")

        if near:
            with st.expander("Also considered (below confidence threshold)"):
                st.markdown("".join(bar(n, p, faded=True) for n, p, _ in near),
                            unsafe_allow_html=True)

with st.expander("How it works & honest notes"):
    st.markdown(
        "Each emotion is scored independently; it's shown as **detected** only if its "
        "probability clears a threshold tuned per-emotion on validation data. The "
        "*also considered* list shows near-misses so you can see what the model was weighing.\n\n"
        "GoEmotions is subjective (annotators disagree a lot) and very imbalanced, so "
        "some rare emotions (e.g. *grief*, *nervousness*) are harder to detect confidently. "
        "This model reaches ~0.53 Macro-F1; the gains over a plain baseline come from an "
        "Asymmetric Loss that specifically targets those rare classes.\n\n"
        f"Model: https://huggingface.co/{REPO}"
    )
