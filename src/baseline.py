"""TF-IDF + Logistic Regression baseline for sanity check."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

from src.utils import concat_title_body


def build_pipeline(max_features: int = 50000, ngram_range: tuple[int, int] = (1, 2)) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, stop_words="english")),
            ("clf", LogisticRegression(max_iter=1000, n_jobs=-1, class_weight="balanced")),
        ]
    )


def train_baseline(train_path: str | Path, val_path: str | Path, out_path: str | Path = "models/baseline.joblib") -> Pipeline:
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    def _texts(df: pd.DataFrame):
        return [concat_title_body(str(t), str(b)) for t, b in zip(df["title"], df["body"])]

    X_train = _texts(train_df)
    y_train = train_df["label"].astype(int).tolist()
    X_val = _texts(val_df)
    y_val = val_df["label"].astype(int).tolist()

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_val)
    acc = accuracy_score(y_val, preds)
    print(f"[baseline] val accuracy: {acc:.4f}")
    print(classification_report(y_val, preds, target_names=["real", "fake"]))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_path)
    print(f"Saved baseline -> {out_path}")
    return pipe


def predict_baseline(pipe: Pipeline, title: str, body: str) -> tuple[str, float]:
    text = concat_title_body(title, body)
    proba = pipe.predict_proba([text])[0]
    # proba order matches classes sorted; find index of 1
    classes = list(pipe.named_steps["clf"].classes_)
    idx_fake = classes.index(1) if 1 in classes else 1
    p_fake = float(proba[idx_fake])
    label = "fake" if p_fake >= 0.5 else "real"
    return label, p_fake


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--train", default="data/processed/train.parquet")
    p.add_argument("--val", default="data/processed/val.parquet")
    p.add_argument("--out", default="models/baseline.joblib")
    args = p.parse_args()
    train_baseline(args.train, args.val, args.out)
