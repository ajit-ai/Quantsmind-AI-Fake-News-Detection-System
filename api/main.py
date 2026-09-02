"""FastAPI serving layer: POST /predict, GET /health.

- Loads model once at startup (HF or baseline fallback)
- Supports batched requests (single item for now; batch via list extension)
- Optional ONNX export when ONNX_EXPORT=1
- Explainability via src.explain
- English validation at request time
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException

from api.schemas import HealthResponse, PredictRequest, PredictResponse, TokenScore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fake News Detector", version="1.0.0")

# Global state loaded at startup
_state: dict = {"model": None, "tokenizer": None, "baseline": None, "model_name": None, "use_baseline": False}


def _load_model() -> None:
    model_path = os.getenv("MODEL_PATH", "models/checkpoints/fold_0/best")
    model_name = os.getenv("MODEL_NAME", "roberta-base")
    baseline_path = os.getenv("BASELINE_PATH", "models/baseline.joblib")
    _state["model_name"] = model_name

    # Try HF model
    if Path(model_path).exists() and any(Path(model_path).glob("*")):
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(model_path)
            mod = AutoModelForSequenceClassification.from_pretrained(model_path)
            mod.eval()
            _state["model"] = mod
            _state["tokenizer"] = tok
            logger.info(f"Loaded HF model from {model_path}")
            # ONNX export if requested
            if os.getenv("ONNX_EXPORT", "0") == "1":
                try:
                    from src.model import export_onnx

                    export_onnx(mod, tok, "models/model.onnx")
                except Exception as e:
                    logger.warning(f"ONNX export failed: {e}")
            return
        except Exception as e:
            logger.warning(f"HF load failed ({e}), trying baseline")

    # Try baseline
    if Path(baseline_path).exists():
        try:
            import joblib

            _state["baseline"] = joblib.load(baseline_path)
            _state["use_baseline"] = True
            logger.info(f"Loaded baseline from {baseline_path}")
            return
        except Exception as e:
            logger.warning(f"Baseline load failed: {e}")

    # Try loading from HF hub directly (no local checkpoint)
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_name)
        mod = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        mod.eval()
        _state["model"] = mod
        _state["tokenizer"] = tok
        logger.info(f"Loaded HF model from hub: {model_name} (untuned)")
        return
    except Exception as e:
        logger.warning(f"Hub load failed: {e}")

    logger.warning("No model loaded — /predict will return 503 until a model is available")


@app.on_event("startup")
async def startup() -> None:
    _load_model()


@app.get("/health", response_model=HealthResponse)
async def health():
    loaded = _state["model"] is not None or _state["baseline"] is not None
    return HealthResponse(status="ok" if loaded else "no_model", model_loaded=loaded, model_name=_state.get("model_name"))


def _predict_with_hf(title: str, body: str) -> tuple[str, float, list]:
    import torch

    from src.utils import truncate_for_model

    model = _state["model"]
    tokenizer = _state["tokenizer"]
    text = truncate_for_model(title, body, tokenizer, max_length=512)
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
        probs = torch.softmax(logits, dim=-1)
        p_fake = float(probs[0, 1].item())
        label = "fake" if p_fake >= 0.5 else "real"

    # explain
    top_features: list[dict] = []
    try:
        from src.explain import explain_prediction

        feats = explain_prediction(model, tokenizer, title, body, top_k=8)
        top_features = feats
    except Exception as e:
        logger.debug(f"explain failed: {e}")
        top_features = []

    return label, p_fake, top_features


def _predict_with_baseline(title: str, body: str) -> tuple[str, float, list]:
    from src.baseline import predict_baseline
    from src.utils import SENSATIONAL_WORDS

    pipe = _state["baseline"]
    label, p_fake = predict_baseline(pipe, title, body)
    # heuristic features for baseline
    text = f"{title} {body}".lower()
    feats = [{"token": w, "score": 1.0} for w in SENSATIONAL_WORDS if w in text][:5]
    return label, p_fake, feats


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    # validate not both empty
    if not req.title.strip() and not req.body.strip():
        raise HTTPException(status_code=400, detail="Both title and body are empty")
    # english check
    try:
        req.validate_english()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if _state["model"] is not None and _state["tokenizer"] is not None:
        label, p_fake, feats = _predict_with_hf(req.title, req.body)
    elif _state["baseline"] is not None:
        label, p_fake, feats = _predict_with_baseline(req.title, req.body)
    else:
        raise HTTPException(status_code=503, detail="Model not loaded")

    top = [TokenScore(token=f["token"], score=float(f["score"])) for f in feats]
    return PredictResponse(label=label, probability_fake=p_fake, top_features=top)


# For tests: allow injecting a mock model
def _set_mock_model(model, tokenizer) -> None:
    _state["model"] = model
    _state["tokenizer"] = tokenizer
    _state["use_baseline"] = False
