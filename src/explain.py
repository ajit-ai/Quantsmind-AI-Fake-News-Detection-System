"""Explainability: Captum Integrated Gradients or SHAP fallback.

Exposes `explain_prediction` callable from API and evaluation.
Returns top-k tokens/phrases with attribution scores for the 'fake' class.
"""

from __future__ import annotations

import torch


def _integrated_gradients(
    model,
    tokenizer,
    text: str,
    target: int = 1,
    n_steps: int = 25,
    internal_batch_size: int = 1,
) -> list[dict]:
    """Captum IG on token embeddings. Returns token -> score."""
    try:
        from captum.attr import LayerIntegratedGradients  # type: ignore
    except ImportError as e:
        raise ImportError("captum not installed; cannot use IG") from e

    model.eval()
    # Find embedding layer (works for RoBERTa/DeBERTa/BERT)
    embeddings = None
    for name in ["roberta.embeddings", "deberta.embeddings", "bert.embeddings", "embeddings"]:
        try:
            obj = model
            for part in name.split("."):
                obj = getattr(obj, part)
            embeddings = obj
            break
        except AttributeError:
            continue
    if embeddings is None:
        # fallback: first param that looks like embeddings
        embeddings = list(model.modules())[1]

    lig = LayerIntegratedGradients(lambda inputs, mask: model(inputs, attention_mask=mask).logits, embeddings)

    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    # baseline: pad token ids
    baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id or 0)

    attributions, delta = lig.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        additional_forward_args=(attention_mask,),
        target=target,
        n_steps=n_steps,
        internal_batch_size=internal_batch_size,
        return_convergence_delta=True,
    )
    # sum over embedding dim
    attr = attributions.sum(dim=-1).squeeze(0)  # [seq]
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())
    scores = attr.detach().cpu().tolist()
    # filter special tokens
    items = []
    for tok, sc in zip(tokens, scores):
        if tok in (tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token, "<s>", "</s>", "[CLS]", "[SEP]", "[PAD]"):
            continue
        # clean roberta leading Ġ
        clean = tok.replace("Ġ", "").replace("▁", "")
        if not clean.strip():
            continue
        items.append({"token": clean, "score": float(sc)})
    # sort by absolute attribution descending
    items.sort(key=lambda x: abs(x["score"]), reverse=True)
    return items


def _shap_fallback(model, tokenizer, text: str, target: int = 1) -> list[dict]:
    """SHAP fallback using shap.Explainer on pipeline wrapper."""
    try:
        import numpy as np  # type: ignore
        import shap  # type: ignore
    except ImportError as e:
        raise ImportError("shap not installed") from e

    def _predict(texts: list[str]):
        import torch

        model.eval()
        probs = []
        with torch.no_grad():
            for t in texts:
                enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=512, padding=True)
                logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
                p = torch.softmax(logits, dim=-1)[0, target].item()
                probs.append(p)
        return np.array(probs)

    # SHAP not ideal for HF, use simple token occlusion as lightweight fallback
    tokens = text.split()
    baseline_p = _predict([text])[0]
    items: list[dict] = []
    for tok in tokens:
        masked = text.replace(tok, "[MASK]", 1)
        p = _predict([masked])[0]
        score = float(baseline_p - p)  # drop when token removed
        items.append({"token": tok, "score": score})
    items.sort(key=lambda x: abs(x["score"]), reverse=True)
    return items


def explain_prediction(
    model,
    tokenizer,
    title: str,
    body: str,
    top_k: int = 8,
    method: str = "ig",
) -> list[dict]:
    """Return top-k influential tokens for fake prediction.

    Args:
        model: HF sequence classifier (eval mode)
        tokenizer: HF tokenizer
        title, body: article fields
        top_k: number of tokens to return
        method: "ig" (Captum) or "shap"/"occlusion"
    """
    from src.utils import truncate_for_model

    text = truncate_for_model(title, body, tokenizer, max_length=512)
    if not text.strip():
        return []

    # try IG, fallback to occlusion
    try:
        if method == "ig":
            items = _integrated_gradients(model, tokenizer, text, target=1)
        else:
            raise ValueError("force fallback")
    except Exception as e:
        # fallback: token occlusion / shap-lite
        try:
            items = _shap_fallback(model, tokenizer, text, target=1)
        except Exception as e2:
            # ultimate fallback: heuristic sensational-word scoring
            print(f"[warn] explain fallbacks failed: {e} / {e2}; using heuristic")
            from src.utils import SENSATIONAL_WORDS

            lower = text.lower()
            items = [{"token": w, "score": 1.0} for w in SENSATIONAL_WORDS if w in lower][:top_k]
            if not items:
                items = [{"token": t, "score": 0.0} for t in text.split()[:top_k]]

    return items[:top_k]


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="models/checkpoints/fold_0/best")
    p.add_argument("--title", default="SHOCK: Miracle cure hidden by pharma")
    p.add_argument("--body", default="Insiders claim the truth has been hidden for decades and share before deleted.")
    p.add_argument("--top-k", type=int, default=8)
    args = p.parse_args()

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_path)
    mod = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    print(explain_prediction(mod, tok, args.title, args.body, top_k=args.top_k))
