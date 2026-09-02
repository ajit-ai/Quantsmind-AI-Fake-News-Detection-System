"""Streamlit demo: calls same inference module as API.

Run: streamlit run demo/app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("Fake News Detector — Demo")
st.caption("Style/pattern triage filter. Flags likely-fake content for human review; not a ground-truth arbiter.")

@st.cache_resource
def get_model():
    model_path = os.getenv("MODEL_PATH", "models/checkpoints/fold_0/best")
    model_name = os.getenv("MODEL_NAME", "roberta-base")
    baseline_path = os.getenv("BASELINE_PATH", "models/baseline.joblib")
    # try HF local
    if Path(model_path).exists() and any(Path(model_path).glob("*")):
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(model_path)
            mod = AutoModelForSequenceClassification.from_pretrained(model_path)
            mod.eval()
            return ("hf", mod, tok)
        except Exception as e:
            st.warning(f"HF load failed: {e}")
    if Path(baseline_path).exists():
        try:
            import joblib

            pipe = joblib.load(baseline_path)
            return ("baseline", pipe, None)
        except Exception as e:
            st.warning(f"Baseline load failed: {e}")
    # try hub
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_name)
        mod = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        mod.eval()
        return ("hf", mod, tok)
    except Exception as e:
        st.error(f"No model available: {e}")
        return (None, None, None)


mode, model, tokenizer = get_model()

title = st.text_input("Title / headline", placeholder="e.g., Government announces new infrastructure plan")
body = st.text_area("Body", height=200, placeholder="Paste article body here...")

if st.button("Detect", type="primary"):
    if not title.strip() and not body.strip():
        st.error("Please provide title or body.")
    elif mode is None:
        st.error("Model not loaded. Train a model first (see README).")
    elif mode == "baseline":
        from src.baseline import predict_baseline

        label, p_fake = predict_baseline(model, title, body)
        st.metric("Prediction", label.upper(), delta=f"p_fake={p_fake:.3f}")
        st.progress(p_fake, text=f"Probability fake: {p_fake:.2%}")
    else:
        import torch

        from src.utils import truncate_for_model

        text = truncate_for_model(title, body, tokenizer, max_length=512)
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
            p_fake = float(torch.softmax(logits, dim=-1)[0, 1].item())
        label = "fake" if p_fake >= 0.5 else "real"
        st.metric("Prediction", label.upper(), delta=f"p_fake={p_fake:.3f}")
        st.progress(p_fake, text=f"Probability fake: {p_fake:.2%}")

        # explain
        try:
            from src.explain import explain_prediction

            feats = explain_prediction(model, tokenizer, title, body, top_k=8)
            if feats:
                st.subheader("Top influential tokens")
                for f in feats:
                    st.write(f"`{f['token']}` — score `{f['score']:.4f}`")
                # highlight in text
                html = text
                for f in feats[:3]:
                    html = html.replace(f["token"], f"<mark>{f['token']}</mark>")
                st.markdown(html, unsafe_allow_html=True)
        except Exception as e:
            st.caption(f"Explainability unavailable: {e}")

st.divider()
st.caption("Tip: run ingestion + training first: `python data/scripts/build_dataset.py --synthetic` then `python src/train.py --no-cv` or baseline.")
