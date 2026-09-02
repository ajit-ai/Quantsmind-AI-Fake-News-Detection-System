"""Build training dataset: merge public corpora + stratified split.

- Downloads/merges WELFake and optionally FakeNewsNet / ISOT if available
- Harmonizes to schema: title, body, label (0=real, 1=fake), source
- Stratified 70/15/15 split with fixed seed
- Logs class balance per split
- Falls back to synthetic dummy data if no public file found (for CI/demo)

Usage:
    python data/scripts/build_dataset.py --welfake data/raw/WELFake_Dataset.csv --out-dir data/processed
    python data/scripts/build_dataset.py --synthetic --n 1000 --out-dir data/processed
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from clean import clean_text  # when run as script
except ImportError:
    try:
        from data.scripts.clean import clean_text
    except ImportError:
        def clean_text(x: str, is_html: bool = False) -> str:  # type: ignore
            return x


RANDOM_SEED = 42


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Try to map common column names to title/body/label."""
    col_lower = {c.lower(): c for c in df.columns}
    # title
    title_col = None
    for cand in ["title", "headline", "statement"]:
        if cand in col_lower:
            title_col = col_lower[cand]
            break
    # text/body
    body_col = None
    for cand in ["text", "body", "content", "article", "news"]:
        if cand in col_lower:
            body_col = col_lower[cand]
            break
    # label
    label_col = None
    for cand in ["label", "class", "target", "fake"]:
        if cand in col_lower:
            label_col = col_lower[cand]
            break

    if label_col is None:
        raise ValueError(f"Could not find label column in {list(df.columns)}")
    if body_col is None and title_col is None:
        raise ValueError(f"Could not find text column in {list(df.columns)}")

    out = pd.DataFrame()
    out["title"] = df[title_col].astype(str) if title_col else ""
    out["body"] = df[body_col].astype(str) if body_col else df[title_col].astype(str)
    # normalize label to 0/1 (real=0, fake=1)
    raw_labels = df[label_col]
    # handle string labels
    def _map_label(v) -> int:
        s = str(v).strip().lower()
        if s in ("1", "fake", "false", "unreliable", "1.0"):
            return 1
        if s in ("0", "real", "true", "reliable", "0.0"):
            return 0
        try:
            return int(float(s))
        except Exception:
            return 1 if "fake" in s else 0

    out["label"] = raw_labels.map(_map_label).astype(int)
    out["source"] = "welfake"
    return out[["title", "body", "label", "source"]]


def load_welfake(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    print(f"Loading WELFake from {path}")
    # try csv
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_parquet(path)
    df = _normalize_columns(df)
    # optional cleaning
    df["title"] = df["title"].apply(lambda x: clean_text(str(x)))
    df["body"] = df["body"].apply(lambda x: clean_text(str(x)))
    return df


def make_synthetic(n: int = 1000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate synthetic balanced dataset for CI / demo without downloading."""
    random.seed(seed)
    titles_real = [
        "Government announces new infrastructure plan",
        "Study shows benefits of regular exercise",
        "Local university hosts science conference",
    ]
    titles_fake = [
        "SHOCK: Scientists hide alien discovery!!!",
        "You won't believe what this politician secretly did",
        "BREAKING: Miracle cure suppressed by big pharma",
    ]
    bodies_real = [
        "According to official reports, the initiative will focus on sustainable development and public consultation over the next fiscal year.",
        "Researchers at the institute published peer-reviewed findings indicating moderate improvements in cardiovascular health.",
    ]
    bodies_fake = [
        "Insiders claim the truth has been hidden for decades and only this leaked document reveals what really happened!!!",
        "Share this before it gets deleted — the mainstream media doesn't want you to know this secret method.",
    ]
    rows = []
    for i in range(n):
        label = i % 2  # balanced
        if label == 0:
            rows.append(
                {
                    "title": random.choice(titles_real),
                    "body": random.choice(bodies_real) + f" Reference id {i}.",
                    "label": 0,
                    "source": "synthetic",
                }
            )
        else:
            rows.append(
                {
                    "title": random.choice(titles_fake),
                    "body": random.choice(bodies_fake) + f" Reference id {i}.",
                    "label": 1,
                    "source": "synthetic",
                }
            )
    df = pd.DataFrame(rows)
    # shuffle deterministically
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def log_balance(name: str, df: pd.DataFrame) -> None:
    counts = df["label"].value_counts().sort_index().to_dict()
    total = len(df)
    print(f"[{name}] n={total}  label counts: {counts}  fake_rate={counts.get(1,0)/total:.3f}")


def stratified_split(df: pd.DataFrame, seed: int = RANDOM_SEED, train_ratio: float = 0.70, val_ratio: float = 0.15):
    """70/15/15 stratified split."""
    train, temp = train_test_split(df, test_size=1 - train_ratio, random_state=seed, stratify=df["label"])
    val_size = val_ratio / (1 - train_ratio)
    val, test = train_test_split(temp, test_size=1 - val_size, random_state=seed, stratify=temp["label"])
    return train, val, test


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dataset with stratified splits")
    parser.add_argument("--welfake", type=str, default="data/raw/WELFake_Dataset.csv", help="Path to WELFake csv/parquet")
    parser.add_argument("--fakenewsnet", type=str, default=None, help="Optional FakeNewsNet path for extra eval")
    parser.add_argument("--out-dir", type=str, default="data/processed")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic data instead of loading files")
    parser.add_argument("--n", type=int, default=2000, help="Synthetic n")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        df = make_synthetic(n=args.n, seed=args.seed)
        print(f"Generated synthetic dataset n={len(df)}")
    else:
        df = load_welfake(Path(args.welfake))
        if df is None:
            print(f"[warn] WELFake not found at {args.welfake}, falling back to synthetic data")
            df = make_synthetic(n=args.n, seed=args.seed)
        # optionally merge FakeNewsNet as held-out generalization (not for training)
        if args.fakenewsnet and Path(args.fakenewsnet).exists():
            print(f"[info] FakeNewsNet found at {args.fakenewsnet} — will be used as secondary eval only, not merged into train.")

    # drop empty
    df = df[(df["title"].str.strip() != "") | (df["body"].str.strip() != "")].copy()
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    train, val, test = stratified_split(df, seed=args.seed)

    for name, split in [("train", train), ("val", val), ("test", test)]:
        log_balance(name, split)
        out = out_dir / f"{name}.parquet"
        split.to_parquet(out, index=False)
        print(f"Saved {out} ({len(split)} rows)")

    # also log overall
    log_balance("overall", df)
    # write a small meta file
    meta = {
        "seed": args.seed,
        "total": len(df),
        "train": len(train),
        "val": len(val),
        "test": len(test),
    }
    import json

    (out_dir / "split_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Done. Meta: {meta}")


if __name__ == "__main__":
    main()
