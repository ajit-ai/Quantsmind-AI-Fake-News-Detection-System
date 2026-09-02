"""Shared utilities: seeding, concatenation, text prep, metrics."""

from __future__ import annotations

import random
import re

import numpy as np


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def concat_title_body(title: str, body: str, max_chars: int | None = None) -> str:
    """Concatenate title + [SEP] + body. Title is preserved; body is truncated."""
    title = (title or "").strip()
    body = (body or "").strip()
    if not title and not body:
        return ""
    if not title:
        text = body
    elif not body:
        text = title
    else:
        text = f"{title} [SEP] {body}"
    if max_chars and len(text) > max_chars:
        # keep title, truncate body tail
        if " [SEP] " in text:
            prefix, _ = text.split(" [SEP] ", 1)
            keep_body = max_chars - len(prefix) - len(" [SEP] ")
            text = f"{prefix} [SEP] {body[:keep_body]}"
        else:
            text = text[:max_chars]
    return text


def truncate_for_model(title: str, body: str, tokenizer, max_length: int = 512) -> str:
    """Truncate from body side so title signal is preserved."""
    title = (title or "").strip()
    body = (body or "").strip()
    if not title:
        return body
    if not body:
        return title
    # encode separately to count
    title_ids = tokenizer.encode(title, add_special_tokens=False)
    sep_ids = tokenizer.encode(" [SEP] ", add_special_tokens=False)
    body_ids = tokenizer.encode(body, add_special_tokens=False)
    # account for [CLS] + [SEP] added by tokenizer
    budget = max_length - 2  # CLS + SEP
    if len(title_ids) + len(sep_ids) + len(body_ids) <= budget:
        return f"{title} [SEP] {body}"
    # truncate body
    keep_body = max(0, budget - len(title_ids) - len(sep_ids))
    body_ids = body_ids[:keep_body]
    body_trunc = tokenizer.decode(body_ids, skip_special_tokens=True)
    return f"{title} [SEP] {body_trunc}"


# ---------------------------------------------------------------------------
# Lightweight credibility / handcrafted signals (optional auxiliary)
# ---------------------------------------------------------------------------

SENSATIONAL_WORDS = {
    "shocking",
    "unbelievable",
    "secret",
    "exposed",
    "breaking",
    "miracle",
    "hidden",
    "leaked",
    "conspiracy",
    "cover-up",
    "you won't believe",
    "before it's deleted",
}

CLAIM_PATTERNS = re.compile(r"\b(according to|experts say|studies show|reportedly|allegedly)\b", re.IGNORECASE)


def handcrafted_features(text: str) -> dict:
    lower = text.lower()
    return {
        "length": len(text),
        "exclamation_count": text.count("!"),
        "caps_ratio": sum(1 for c in text if c.isupper()) / max(1, len(text)),
        "sensational_hits": sum(1 for w in SENSATIONAL_WORDS if w in lower),
        "claim_density": len(CLAIM_PATTERNS.findall(text)) / max(1, len(text.split())),
    }
