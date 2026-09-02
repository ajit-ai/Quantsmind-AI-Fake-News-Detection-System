# Quantsmind AI — Fake News Detection System

Production-ready, end-to-end English-language fake news detection system.
Transformer (RoBERTa/DeBERTa) fine-tuned as binary classifier (real=0 / fake=1), TF-IDF baseline, explainability (Captum IG / SHAP), FastAPI serving with ONNX, Streamlit demo, and Docker.

> **Scope note:** On curated benchmarks (WELFake, ISOT, FakeNewsNet) a fine-tuned RoBERTa/DeBERTa reliably clears 90%+ accuracy *within* dataset style. Open-domain adversarial misinformation is lower — this system is a **style/pattern triage filter** for human / fact-check review, not a ground-truth arbiter.

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INGESTION LAYER                              │
│  RSS/News-API pullers + custom scrapers (requests/httpx + trafilatura│
│  for boilerplate-free extraction) → dedup (URL hash + simhash on     │
│  body text) → language filter (langdetect, keep en only)             │
└───────────────────────────────┬───────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       CLEANING / NORMALIZATION                       │
│  HTML strip, boilerplate removal, unicode normalization, sentence    │
│  splitting, metadata extraction (title, author, publish_date, source)│
│  → stored as Parquet/Postgres rows: id, url, source, title, body,    │
│    published_at, scraped_at, lang, raw_html_ref                      │
└───────────────────────────────┬───────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        FEATURE / DATASET LAYER                       │
│  Training corpus = curated public datasets (see §2) + optional       │
│  domain-credibility features (source reputation score, sensational-  │
│  ism lexicon hits, claim density) as auxiliary signals               │
└───────────────────────────────┬───────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          MODEL LAYER                                 │
│  Transformer encoder (RoBERTa-base or DeBERTa-v3-base) fine-tuned    │
│  as binary sequence classifier (real=0 / fake=1), title+body concat  │
│  with [SEP], max_len 512 w/ sliding-window or truncation strategy.   │
│  Optional: gradient-boosted stacking on encoder embedding +          │
│  handcrafted features for a small accuracy bump.                     │
│  Explainability: Captum Integrated Gradients or SHAP on token        │
│  embeddings → top contributing tokens/phrases returned with score.   │
└───────────────────────────────┬───────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       EVALUATION / REPORTING                         │
│  Stratified train/val/test split (70/15/15), 5-fold CV on train for  │
│  model selection, held-out test only touched once. Report: accuracy, │
│  precision, recall, F1 (macro + per-class), confusion matrix, ROC-AUC│
└───────────────────────────────┬───────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          SERVING LAYER                                │
│  FastAPI: POST /predict {title, body} → {label, p_fake, top_features}│
│  Batching + async inference, ONNX export for latency, optional       │
│  Streamlit/Gradio demo UI on top of same inference module            │
└───────────────────────────────┬───────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        DEPLOYMENT / OPS                              │
│  Dockerfile (multi-stage), docker-compose (api + optional Postgres), │
│  model registry via MLflow or plain versioned S3/local artifact dir, │
│  CI (pytest + lint) via GitHub Actions, structured logging, basic    │
│  drift monitoring (input length/vocab shift, prediction distribution)│
└─────────────────────────────────────────────────────────────────────┘
```

**Compact flow:**
```
RSS/NewsAPI → dedup (URL hash + simhash) → lang filter → Parquet
→ cleaning (HTML/unicode/whitespace) → dataset merge (WELFake + FakeNewsNet) → 70/15/15 stratified split
→ Transformer (title+ [SEP] + body, max 512, title-preserving truncation) → Trainer + 5-fold CV
→ evaluate (accuracy, precision, recall, F1 macro/per-class, confusion matrix, ROC-AUC) → report
→ FastAPI POST /predict {title,body} → {label, p_fake, top_features} + ONNX + Streamlit demo
→ Docker / CI / MLflow logging
```

### Key Design Decisions and Why
- **Model choice**: RoBERTa-base / DeBERTa-v3-base are strong, well-documented baselines for text classification and reliably clear 90%+ accuracy on curated fake-news benchmarks (e.g., LIAR, FakeNewsNet, ISOT, WELFake) without exotic tuning — good balance of accuracy vs. training cost. DeBERTa-v3 tends to edge out RoBERTa by 1–3 points if compute allows.
- **Title+body concatenation**: fake news signals are often strongest in headlines (clickbait, sensationalism) — concatenating and truncating from the body (not the title) preserves that signal (`src/utils.py:28` `truncate_for_model`).
- **Explainability via Integrated Gradients/SHAP** (`src/explain.py:22`) rather than raw attention weights: attention is a weaker proxy for "why" the model decided; gradient-based attribution is more defensible for transparency.
- **Held-out test set touched exactly once** (`src/evaluate.py:1`): prevents metric inflation from repeated peeking.
- **Source-credibility auxiliary feature is optional**: using domain reputation can leak label information if training set is domain-correlated. Text-only is primary, credibility as optional add-on.
- **ONNX export** (`src/model.py:58`): keeps inference latency reasonable for CPU demo/API without GPU.

### Recommended Datasets (English, publicly available)
- **WELFake** (~72k articles, merges 4 datasets, good class balance) — primary training corpus (`data/scripts/build_dataset.py:22`)
- **FakeNewsNet** (PolitiFact + GossipCop) — secondary held-out generalization test
- **LIAR** (short statements, auxiliary claims model)
- **ISOT Fake News Dataset**
> Train primarily on WELFake for scale/balance, use FakeNewsNet as secondary generalization check.

### Repo Structure
```
fake-news-detector/
├── README.md
├── requirements.txt
├── environment.yml
├── Dockerfile
├── docker-compose.yml
├── .github/workflows/ci.yml
├── configs/
│   └── train_config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── scripts/
│       ├── ingest.py
│       ├── clean.py
│       └── build_dataset.py
├── src/
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── explain.py
│   ├── baseline.py
│   └── utils.py
├── api/
│   ├── main.py
│   └── schemas.py
├── demo/
│   └── app.py
├── tests/
│   ├── test_clean.py
│   ├── test_model.py
│   └── test_api.py
├── reports/
│   └── evaluation_report.md
└── models/  (trained weights or download script)
```

### Milestones (as implemented)
1. Ingestion + cleaning pipeline, sample stored dataset (1–2 days) — `data/scripts/ingest.py:110`, `data/scripts/clean.py:19` ✅
2. Dataset build + baseline (TF-IDF + LogReg) (`src/baseline.py:14`) ✅
3. Transformer fine-tuning + 5-fold CV, target ≥90% (`src/train.py:34`, `configs/train_config.yaml:1`) — code ready, GPU run pending
4. Explainability wired into inference (`src/explain.py:22`, `api/main.py:118`) ✅
5. FastAPI + Dockerfile + demo UI (`api/main.py:87`, `Dockerfile:1`, `demo/app.py:13`) ✅
6. Evaluation report, README, CI (`src/evaluate.py:29`, `.github/workflows/ci.yml:1`) ✅

## Repo Structure

```
configs/train_config.yaml
data/scripts/ingest.py, clean.py, build_dataset.py
src/model.py, train.py, evaluate.py, explain.py, baseline.py, utils.py
api/main.py, schemas.py
demo/app.py
tests/test_clean.py, test_model.py, test_api.py
reports/evaluation_report.md
Dockerfile, docker-compose.yml, .github/workflows/ci.yml
```

## Setup

```bash
# pip
pip install -r requirements.txt

# conda
conda env create -f environment.yml
conda activate fakenews
```

## 1. Ingestion

```bash
python data/scripts/ingest.py --out data/raw/ingested.parquet
# or custom feeds
python data/scripts/ingest.py --feeds https://feeds.bbci.co.uk/news/rss.xml --out data/raw/ingested.parquet
```

## 2. Build Dataset

```bash
# synthetic (no download, for CI/demo)
python data/scripts/build_dataset.py --synthetic --n 2000 --out-dir data/processed

# with WELFake CSV (download from Kaggle: WELFake_Dataset.csv)
python data/scripts/build_dataset.py --welfake data/raw/WELFake_Dataset.csv --out-dir data/processed
```

Outputs `data/processed/{train,val,test}.parquet` with columns `title, body, label, source` and `split_meta.json`. Class balance logged per split. Seed fixed for reproducibility.

## 3. Training

```bash
# baseline (fast, no GPU)
python src/baseline.py --train data/processed/train.parquet --val data/processed/val.parquet

# transformer with 5-fold CV (requires HF model download)
python src/train.py --config configs/train_config.yaml

# single fold (faster)
python src/train.py --config configs/train_config.yaml --no-cv
```

Config: `configs/train_config.yaml` — `model_name` (`roberta-base` default, or `microsoft/deberta-v3-base`), `learning_rate`, `batch_size`, `epochs`, `warmup_ratio`, `weight_decay`, etc. If HF download fails (offline CI), training falls back to baseline automatically.

## 4. Evaluation

```bash
# HF model
python src/evaluate.py --test data/processed/test.parquet --model-path models/checkpoints/fold_0/best --report reports/evaluation_report.md

# baseline
python src/evaluate.py --test data/processed/test.parquet --use-baseline --baseline-path models/baseline.joblib --report reports/evaluation_report.md

# with secondary generalization check
python src/evaluate.py --test data/processed/test.parquet --secondary data/processed/fakenewsnet.parquet --report reports/evaluation_report.md
```

Held-out test is touched **exactly once**. Reports `accuracy, precision, recall, F1 (macro + per-class), confusion matrix, ROC-AUC`. Secondary FakeNewsNet eval reported separately.

Target: **≥90% accuracy** on curated held-out test; if not met, retry hyperparams and document final config.

## 5. Serving

```bash
uvicorn api.main:app --reload --port 8000
# env vars: MODEL_PATH, MODEL_NAME, BASELINE_PATH, ONNX_EXPORT=1

# ONNX export
python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; from src.model import export_onnx; tok=AutoTokenizer.from_pretrained('models/checkpoints/fold_0/best'); m=AutoModelForSequenceClassification.from_pretrained('models/checkpoints/fold_0/best'); export_onnx(m, tok, 'models/model.onnx')"
```

API:
- `GET /health` → `{status, model_loaded, model_name}`
- `POST /predict` body `{title: str, body: str}` → `{label: "real"|"fake", probability_fake: float, top_features: [{token, score}]}`

Non-English input rejected (langdetect). Batched inference supported.

## 6. Demo

```bash
streamlit run demo/app.py
# or Gradio alternative if preferred
```

## Docker

```bash
docker build -t fakenews .
docker run -p 8000:8000 -v ./models:/app/models:ro fakenews

# compose (api + optional postgres)
docker-compose up --build
```

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs `ruff check .` and `pytest -q` on push/PR.

## Tests

```bash
pytest -q
```

Covers cleaning, model forward (mocked), and API `/predict` (mocked model).

## Reproducibility

- All scripts fix seed 42 (configurable via `configs/train_config.yaml`).
- Stratified 70/15/15 split.
- 5-fold CV on train for model selection; best by val F1.
- Structured logging + MLflow optional.

## Final Metrics (after real training — placeholder until first run)

| Split | Accuracy | F1 macro | F1 real | F1 fake | ROC-AUC |
|---|---|---|---|---|---|
| Held-out test | — | — | — | — | — |
| FakeNewsNet (secondary) | — | — | — | — | — |

Regenerate with `python src/evaluate.py` to fill.

## License

MIT
