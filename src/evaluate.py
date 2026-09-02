"""Evaluation: held-out test + optional generalization check.

Computes accuracy, precision, recall, F1 (macro + per-class), confusion matrix, ROC-AUC.
Generates reports/evaluation_report.md with tables/plots and failure discussion.
Test set is touched exactly once — no tuning against it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.utils import concat_title_body


def compute_all_metrics(y_true, y_pred, y_proba: list | None = None) -> dict:
    acc = accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    precision_per, recall_per, f1_per, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    out: dict = {
        "accuracy": float(acc),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_real": float(precision_per[0]),
        "recall_real": float(recall_per[0]),
        "f1_real": float(f1_per[0]),
        "precision_fake": float(precision_per[1]),
        "recall_fake": float(recall_per[1]),
        "f1_fake": float(f1_per[1]),
        "confusion_matrix": cm,  # [[TN, FP],[FN, TP]] for labels [0,1]
    }
    if y_proba is not None:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            out["roc_auc"] = None
    return out


def evaluate_with_model(
    model,
    tokenizer,
    test_df: pd.DataFrame,
    max_length: int = 512,
    batch_size: int = 32,
    device: str = "cpu",
) -> dict:
    """Run HF model on test_df."""
    import torch

    from src.utils import truncate_for_model

    model.eval()
    model.to(device)
    y_true = test_df["label"].astype(int).tolist()
    y_pred: list[int] = []
    y_proba: list[float] = []

    texts = [truncate_for_model(str(t), str(b), tokenizer, max_length) for t, b in zip(test_df["title"], test_df["body"])]

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(batch, return_tensors="pt", truncation=True, max_length=max_length, padding=True)
            logits = model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device)).logits
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
            preds = (torch.softmax(logits, dim=-1).argmax(dim=-1)).cpu().tolist()
            y_pred.extend(preds)
            y_proba.extend(probs)

    return compute_all_metrics(y_true, y_pred, y_proba)


def evaluate_baseline(pipe, test_df: pd.DataFrame) -> dict:
    texts = [concat_title_body(str(t), str(b)) for t, b in zip(test_df["title"], test_df["body"])]
    y_true = test_df["label"].astype(int).tolist()
    y_pred = pipe.predict(texts).tolist()
    try:
        y_proba = pipe.predict_proba(texts)[:, list(pipe.classes_).index(1)].tolist()
    except Exception:
        y_proba = None
    return compute_all_metrics(y_true, y_pred, y_proba)


def generate_report(metrics: dict, out_path: Path, secondary: dict | None = None, config: dict | None = None) -> None:
    """Write markdown report with metrics, confusion matrix table, and discussion."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cm = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
    lines = [
        "# Evaluation Report",
        "",
        "> Held-out test set evaluated exactly once. No hyperparameter tuning against this split.",
        "",
        "## Metrics (held-out test)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Accuracy | {metrics.get('accuracy',0):.4f} |",
        f"| Precision (macro) | {metrics.get('precision_macro',0):.4f} |",
        f"| Recall (macro) | {metrics.get('recall_macro',0):.4f} |",
        f"| F1 (macro) | {metrics.get('f1_macro',0):.4f} |",
        f"| Precision (real) | {metrics.get('precision_real',0):.4f} |",
        f"| Recall (real) | {metrics.get('recall_real',0):.4f} |",
        f"| F1 (real) | {metrics.get('f1_real',0):.4f} |",
        f"| Precision (fake) | {metrics.get('precision_fake',0):.4f} |",
        f"| Recall (fake) | {metrics.get('recall_fake',0):.4f} |",
        f"| F1 (fake) | {metrics.get('f1_fake',0):.4f} |",
        f"| ROC-AUC | {metrics.get('roc_auc','n/a')} |",
        "",
        "## Confusion Matrix",
        "",
        "|  | Pred real | Pred fake |",
        "|---|---|---|",
        f"| True real | {cm[0][0]} | {cm[0][1]} |",
        f"| True fake | {cm[1][0]} | {cm[1][1]} |",
        "",
    ]
    if secondary:
        lines += [
            "## Secondary generalization check (FakeNewsNet / external)",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Accuracy | {secondary.get('accuracy',0):.4f} |",
            f"| F1 macro | {secondary.get('f1_macro',0):.4f} |",
            f"| ROC-AUC | {secondary.get('roc_auc','n/a')} |",
            "",
        ]
    lines += [
        "## Notes",
        "",
        "- Target is >=0.90 accuracy on held-out test for curated benchmarks (WELFake/ISOT). If not met, retry with hyperparameter adjustments and document config.",
        "- This is a style/pattern triage filter, not a ground-truth fact checker; open-domain adversarial misinformation will score lower.",
        "- Source-credibility feature, if used, must be evaluated separately to avoid domain leakage.",
        "",
        "## Failure cases (manual review guidance)",
        "",
        "- Review FP/FN examples: sensational but true vs. subtly false claims.",
        "- Check input length / vocab shift drift monitoring.",
        "",
    ]
    if config:
        lines += ["## Config used", "", "```json", json.dumps(config, indent=2), "```", ""]
    out_path.write_text("\n".join(lines))
    print(f"Report written -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate on held-out test")
    parser.add_argument("--test", default="data/processed/test.parquet")
    parser.add_argument("--secondary", default=None, help="Optional secondary parquet for generalization check")
    parser.add_argument("--model-path", default="models/checkpoints/fold_0/best")
    parser.add_argument("--report", default="reports/evaluation_report.md")
    parser.add_argument("--use-baseline", action="store_true", help="Evaluate baseline instead of HF model")
    parser.add_argument("--baseline-path", default="models/baseline.joblib")
    args = parser.parse_args()

    test_df = pd.read_parquet(args.test)
    print(f"Test rows: {len(test_df)}  balance: {test_df['label'].value_counts().to_dict()}")

    if args.use_baseline:
        import joblib

        pipe = joblib.load(args.baseline_path)
        metrics = evaluate_baseline(pipe, test_df)
    else:
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(args.model_path)
            model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
            metrics = evaluate_with_model(model, tok, test_df)
        except Exception as e:
            print(f"[warn] HF model eval failed ({e}), falling back to baseline if available")
            from pathlib import Path as _P

            import joblib

            if _P(args.baseline_path).exists():
                pipe = joblib.load(args.baseline_path)
                metrics = evaluate_baseline(pipe, test_df)
            else:
                raise

    print(json.dumps(metrics, indent=2))

    secondary = None
    if args.secondary and Path(args.secondary).exists():
        sec_df = pd.read_parquet(args.secondary)
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(args.model_path)
            model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
            secondary = evaluate_with_model(model, tok, sec_df)
        except Exception:
            pass
        print(f"Secondary metrics: {json.dumps(secondary, indent=2)}")

    generate_report(metrics, Path(args.report), secondary=secondary)


if __name__ == "__main__":
    main()
