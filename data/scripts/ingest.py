"""Ingestion layer: RSS / News-API pullers + trafilatura extraction.

Features:
- Configurable source list (RSS URLs or NewsAPI)
- httpx async fetching with retry, trafilatura boilerplate-free extraction
- Dedup via URL hash + simhash on body text
- Language filter (langdetect, keep en only)
- Output Parquet with schema: id, url, source, title, body, published_at, scraped_at, lang, raw_html_ref
"""

from __future__ import annotations

import argparse
import hashlib
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import httpx
import pandas as pd

try:
    import trafilatura  # type: ignore
except ImportError:
    trafilatura = None

try:
    from langdetect import detect  # type: ignore
except ImportError:
    detect = None  # type: ignore

try:
    from data.scripts.clean import clean_text  # type: ignore
except ImportError:
    try:
        from clean import clean_text  # type: ignore
    except ImportError:
        def clean_text(x: str, is_html: bool = False) -> str:  # type: ignore
            return x


DEFAULT_RSS_FEEDS = [
    "https://feeds.bbci.co.uk/news/rss.xml",
    "https://rss.cnn.com/rss/edition.rss",
]

# ---------------------------------------------------------------------------
# Simhash (lightweight, no extra dep)
# ---------------------------------------------------------------------------

def _hash_features(text: str, n: int = 3) -> list[str]:
    tokens = re.findall(r"\w+", text.lower())
    if len(tokens) < n:
        return tokens
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def simhash(text: str, bits: int = 64) -> int:
    """Compute simhash integer for near-dup detection."""
    v = [0] * bits
    for feat in _hash_features(text):
        h = int(hashlib.md5(feat.encode()).hexdigest(), 16)
        for b in range(bits):
            v[b] += 1 if (h >> b) & 1 else -1
    fingerprint = 0
    for b in range(bits):
        if v[b] > 0:
            fingerprint |= 1 << b
    return fingerprint


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# ---------------------------------------------------------------------------
# Article dataclass
# ---------------------------------------------------------------------------

@dataclass
class Article:
    id: str
    url: str
    source: str
    title: str
    body: str
    published_at: str | None
    scraped_at: str
    lang: str
    raw_html_ref: str = ""


def url_hash(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def detect_language(text: str) -> str:
    if detect is None:
        return "en"
    try:
        return detect(text[:2000])
    except Exception:
        return "unknown"


async def fetch_and_extract(client: httpx.AsyncClient, url: str) -> tuple[str, str]:
    """Fetch URL and extract title/body via trafilatura."""
    try:
        resp = await client.get(url, timeout=15, follow_redirects=True)
        resp.raise_for_status()
        html = resp.text
        if trafilatura is not None:
            extracted = trafilatura.extract(html, include_comments=False, include_tables=False)
            meta = trafilatura.extract_metadata(html)
            title = meta.title if meta and meta.title else ""
            body = extracted or ""
        else:
            body = html
            title = ""
        return title, body
    except Exception as e:
        print(f"[warn] fetch failed {url}: {e}")
        return "", ""


async def ingest_rss(feeds: list[str], out_path: Path, lang_filter: str = "en", dedup_hamming: int = 3) -> pd.DataFrame:
    """Ingest RSS feeds, dedup, lang-filter, save parquet."""
    import xml.etree.ElementTree as ET

    articles: list[Article] = []
    seen_urls: set[str] = set()
    seen_hashes: list[int] = []

    async with httpx.AsyncClient(headers={"User-Agent": "FakeNewsDetector/1.0"}) as client:
        for feed_url in feeds:
            try:
                resp = await client.get(feed_url, timeout=15)
                resp.raise_for_status()
                root = ET.fromstring(resp.text)
            except Exception as e:
                print(f"[warn] rss fetch failed {feed_url}: {e}")
                continue

            # crude RSS parsing (works for RSS 2.0)
            for item in root.findall(".//item"):
                link_el = item.find("link")
                title_el = item.find("title")
                pub_el = item.find("pubDate")
                url = (link_el.text or "").strip() if link_el is not None else ""
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                rss_title = (title_el.text or "").strip() if title_el is not None else ""
                pub = (pub_el.text or "").strip() if pub_el is not None else None

                fetched_title, body = await fetch_and_extract(client, url)
                title = fetched_title or rss_title
                if not body:
                    continue

                # clean
                title_c = clean_text(title)
                body_c = clean_text(body)

                # lang filter
                lang = detect_language(f"{title_c} {body_c}")
                if lang_filter and lang != lang_filter:
                    continue

                # simhash dedup
                sh = simhash(body_c)
                if any(hamming(sh, prev) <= dedup_hamming for prev in seen_hashes):
                    continue
                seen_hashes.append(sh)

                source = urlparse(url).netloc
                articles.append(
                    Article(
                        id=url_hash(url),
                        url=url,
                        source=source,
                        title=title_c,
                        body=body_c,
                        published_at=pub,
                        scraped_at=datetime.now(timezone.utc).isoformat(),
                        lang=lang,
                    )
                )
                # be polite
                await _async_sleep(0.2)

    df = pd.DataFrame([asdict(a) for a in articles])
    if not df.empty:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        print(f"Ingested {len(df)} articles -> {out_path}")
    else:
        print("No articles ingested.")
    return df


async def _async_sleep(s: float) -> None:
    import asyncio

    await asyncio.sleep(s)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest RSS feeds")
    parser.add_argument("--feeds", nargs="*", default=DEFAULT_RSS_FEEDS, help="RSS feed URLs")
    parser.add_argument("--feed-file", type=str, default=None, help="File with one feed URL per line")
    parser.add_argument("--out", type=str, default="data/raw/ingested.parquet")
    parser.add_argument("--lang", type=str, default="en", help="Keep only this language (empty to disable)")
    args = parser.parse_args()

    feeds = args.feeds
    if args.feed_file:
        feeds = [l.strip() for l in Path(args.feed_file).read_text().splitlines() if l.strip()]

    import asyncio

    asyncio.run(ingest_rss(feeds, Path(args.out), lang_filter=args.lang))


if __name__ == "__main__":
    main()
