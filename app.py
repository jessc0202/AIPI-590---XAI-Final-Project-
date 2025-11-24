import streamlit as st
import numpy as np
import re

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TextClassificationPipeline,
)
from lime.lime_text import LimeTextExplainer

st.set_page_config(page_title="Explainable Mental Health Classifier", layout="centered")

# ----- UI Styling -----
st.markdown("""
<style>
.main { max-width: 850px; margin: 0 auto; }
textarea { border-radius: 10px !important; border: 1px solid #ccc !important; }
</style>
""", unsafe_allow_html=True)

emoji_map = {"neutral": "😐", "positive": "🙂", "help": "🆘"}


# ============================================================
# Crisis Detection Patterns
# ============================================================
CRISIS_PATTERNS = [
    r"kill\s*my\s*self", r"killing\s*my\s*self", r"end\s*my\s*life",
    r"suicid[e]?", r"\bkys\b", r"unalive",
    r"\bi\s*want\s*to\s*die\b", r"\bwanna\s*die\b",
    r"wan+\s*die", r"wanna\s*d[iy]+", r"want\s*d[iy]+", r"d[iy]+e?",
    r"die\s*my\s*self",
    r"i\s*don'?t\s*want\s*to\s*live", r"tired\s*of\s*living",
    r"no\s*reason\s*to\s*live", r"life\s*is\s*meaningless", r"nothing\s*matters",
    r"wish\s*i\s*was\s*dead", r"wish\s*i\s*were\s*dead",
    r"won'?t\s*wake\s*up", r"wouldn'?t\s*wake\s*up",
]

def check_crisis(text):
    t = text.lower()
    matches = [m.group(0) for p in CRISIS_PATTERNS if (m := re.search(p, t))]
    return matches


# ============================================================
# Load Model FROM HUGGINGFACE
# ============================================================

MODEL_REPO = "Jess02/goemotions-3class-distilbert"

@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_REPO)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_REPO)
    pipeline = TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True,
        batch_size=16
    )
    return tokenizer, model, pipeline

tokenizer, model, pipe = load_model_and_tokenizer()
# ============================================================
# LIME Wrapper
# ============================================================
CLASS_NAMES = ["neutral", "positive", "help"]

def lime_predict(text_list):
    outputs = []
    for txt in text_list:
        preds = pipe(txt)[0]
        scores = [float(d["score"]) for d in preds]
        outputs.append(scores)
    return np.array(outputs)


# ============================================================
# Streamlit UI
# ============================================================
st.markdown("# 🧠 Explainable Mental Health Classifier")
st.write("Understanding emotional tone in short messages using a fine-tuned DistilBERT model.")

user_input = st.text_area("Enter a short description of how you are feeling:", height=120)


# ============================================================
# Main Logic
# ============================================================
if st.button("Analyze"):

    text = user_input.strip()

    if not text:
        st.warning("Please enter text before running the analysis.")
    else:
        # ---------- Crisis Override ----------
        crisis_hits = check_crisis(text)
        if crisis_hits:
            st.subheader("Crisis-Related Language Detected")
            st.write("Matched phrases: " + ", ".join(f"`{m}`" for m in crisis_hits))
            st.subheader("Final Prediction (Safety Override)")
            st.write("**Label:** help 🆘")
            st.info("A real system would escalate to human support.")
            st.stop()

        # ---------- Model Prediction ----------
        pipe_outputs = pipe(text)[0]
        prediction = max(pipe_outputs, key=lambda x: x["score"])["label"]
        prob_dict = {d["label"]: float(d["score"]) for d in pipe_outputs}

        st.subheader("Model Prediction")
        st.write(f"Predicted label: **{emoji_map[prediction]}  {prediction}**")

        st.subheader("Probability Scores")
        st.json(prob_dict)

        # ---------- LIME Explanation ----------
        st.subheader("LIME Explanation")

        explainer = LimeTextExplainer(
            class_names=CLASS_NAMES,
            bow=False,
            split_expression=r"\s+",
            mask_string="",
            char_level=False,
            random_state=42,
            feature_selection="lasso_path"
        )

        label_index = CLASS_NAMES.index(prediction)

        explanation = explainer.explain_instance(
            text,
            lime_predict,
            labels=[label_index],
            num_features=6,
            num_samples=150,
        )

        st.write("Top influential words:")
        st.write(explanation.as_list(label=label_index))

        st.components.v1.html(explanation.as_html(), height=600, scrolling=True)


st.markdown("---")
st.caption("Prototype developed for the Responsible AI Final Project")