"""Cleaning / normalization utilities.

HTML stripping, unicode normalization, whitespace cleanup, sentence splitting,
and metadata normalization. Pure functions for testability.
"""

from __future__ import annotations

import html as html_lib
import re
import unicodedata

try:
    import trafilatura  # type: ignore
except ImportError:
    trafilatura = None  # type: ignore

try:
    from bs4 import BeautifulSoup  # type: ignore
except ImportError:
    BeautifulSoup = None  # type: ignore


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def strip_html(raw_html: str) -> str:
    """Remove HTML tags and return visible text.

    Prefers ``trafilatura.extract`` when available (better boilerplate
    removal), falls back to BeautifulSoup / regex.
    """
    if not raw_html:
        return ""
    # try trafilatura first
    if trafilatura is not None:
        try:
            extracted = trafilatura.extract(raw_html, include_comments=False)
            if extracted:
                return extracted.strip()
        except Exception:
            pass
    if BeautifulSoup is not None:
        soup = BeautifulSoup(raw_html, "html.parser")
        # remove script/style
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)
    # regex fallback
    text = re.sub(r"<[^>]+>", " ", raw_html)
    return html_lib.unescape(text).strip()


def normalize_unicode(text: str) -> str:
    """Apply NFKC normalization and strip control chars."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    # remove zero-width / control chars except newline/tab
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or not unicodedata.category(ch).startswith("C"))
    return text


def normalize_whitespace(text: str) -> str:
    """Collapse whitespace, trim, normalize newlines."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # collapse multiple spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)
    # collapse >2 newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # trim each line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    return text.strip()


def clean_text(raw: str, is_html: bool = False) -> str:
    """Full pipeline: html strip (optional) → unescape → unicode → whitespace."""
    if is_html:
        raw = strip_html(raw)
    raw = html_lib.unescape(raw)
    raw = normalize_unicode(raw)
    raw = normalize_whitespace(raw)
    return raw


def split_sentences(text: str) -> list[str]:
    """Naive sentence splitter (no heavy dep). Splits on [.!?] + space + capital."""
    if not text:
        return []
    # normalize
    text = text.strip()
    # split keeping delimiter
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)
    return [p.strip() for p in parts if p.strip()]


def clean_record(title: str, body: str, title_is_html: bool = False, body_is_html: bool = True) -> dict:
    """Clean title + body and return dict with cleaned fields + sentence lists."""
    clean_title = clean_text(title or "", is_html=title_is_html)
    clean_body = clean_text(body or "", is_html=body_is_html)
    return {
        "title": clean_title,
        "body": clean_body,
        "title_sentences": split_sentences(clean_title),
        "body_sentences": split_sentences(clean_body),
    }


# ---------------------------------------------------------------------------
# CLI for standalone cleaning of parquet/csv
# ---------------------------------------------------------------------------

def _cli() -> None:
    import argparse
    from pathlib import Path

    import pandas as pd

    p = argparse.ArgumentParser(description="Clean a parquet/csv file")
    p.add_argument("--input", required=True, help="Input parquet/csv")
    p.add_argument("--output", required=True, help="Output parquet")
    p.add_argument("--title-col", default="title")
    p.add_argument("--body-col", default="body")
    args = p.parse_args()

    inp = Path(args.input)
    if inp.suffix == ".parquet":
        df = pd.read_parquet(inp)
    else:
        df = pd.read_csv(inp)

    cleaned_titles = []
    cleaned_bodies = []
    for _, row in df.iterrows():
        rec = clean_record(str(row.get(args.title_col, "")), str(row.get(args.body_col, "")))
        cleaned_titles.append(rec["title"])
        cleaned_bodies.append(rec["body"])
    df[args.title_col] = cleaned_titles
    df[args.body_col] = cleaned_bodies
    df.to_parquet(args.output, index=False)
    print(f"Cleaned {len(df)} rows -> {args.output}")


if __name__ == "__main__":
    _cli()
