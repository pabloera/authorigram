"""
Module A — Domain Taxonomy classifier.

Drop-in replacement for the URL-extraction-and-count step of digiNEV
Stage 15 (Domain Analysis).

Usage from a stage script:
    from src.modules.module_a_domain_taxonomy.domain_taxonomy import classify_domains
    df = classify_domains(df, url_column="urls_extracted")

The function adds four columns:
    domain_class
    domain_subclass
    domain_country
    domain_taxonomy_version

If a single message contains multiple URLs, the columns hold
semicolon-joined lists in the order URLs appear in the source.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import tldextract

# ---------------------------------------------------------------------------

TAXONOMY_PATH = Path(__file__).with_name("taxonomy_v1.0.json")
URL_REGEX = re.compile(r"https?://[^\s\"'<>]+", flags=re.IGNORECASE)


def load_taxonomy(path: Path = TAXONOMY_PATH) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def extract_urls(text: Optional[str]) -> list[str]:
    """Pull every absolute URL out of a free-text message."""
    if not isinstance(text, str) or not text:
        return []
    return URL_REGEX.findall(text)


def normalize_domain(url: str) -> str:
    """Return a canonical 'domain.tld' (or 'sub.domain.tld' for known subs)."""
    parts = tldextract.extract(url)
    if not parts.domain or not parts.suffix:
        return ""
    base = f"{parts.domain}.{parts.suffix}"
    if parts.subdomain and parts.subdomain not in {"www", "m", "mobile"}:
        return f"{parts.subdomain}.{base}".lower()
    return base.lower()


def classify_one(domain: str, taxonomy: dict) -> dict:
    """Classify a single domain. Falls back to base domain match if subdomain unknown."""
    domains = taxonomy["domains"]
    if not domain:
        return {"class": "other", "subclass": "unknown", "country": "INT"}
    if domain in domains:
        return domains[domain]
    # try the base domain (drop leftmost subdomain)
    if "." in domain:
        base = domain.split(".", 1)[1]
        if base in domains:
            return domains[base]
    return {"class": "unknown", "subclass": "unknown", "country": "INT"}


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def classify_domains(
    df: pd.DataFrame,
    url_column: str = "urls",
    text_column: str = "text",
    llm_triage_fn=None,
) -> pd.DataFrame:
    """
    Add domain-class columns to a digiNEV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Pipeline DataFrame (one row per message).
    url_column : str
        If present, treated as a list/semicolon-joined string of URLs already
        extracted upstream. If absent or empty, URLs are extracted from text.
    text_column : str
        Fallback source for URL extraction.
    llm_triage_fn : callable | None
        Optional callable `fn(domain: str) -> dict` for unknown domains.
        Pass `None` to skip the LLM step (recommended for the first run).

    Returns
    -------
    pd.DataFrame
        The same DataFrame with four new columns.
    """
    taxonomy = load_taxonomy()
    version = taxonomy["_meta"]["version"]
    cache: dict[str, dict] = {}

    classes, subclasses, countries = [], [], []

    for _, row in df.iterrows():
        urls = _resolve_urls(row, url_column, text_column)
        domains = [normalize_domain(u) for u in urls]
        labels = []
        for d in domains:
            if not d:
                labels.append({"class": "other", "subclass": "unknown", "country": "INT"})
                continue
            if d in cache:
                labels.append(cache[d])
                continue
            label = classify_one(d, taxonomy)
            if label["class"] == "unknown" and llm_triage_fn is not None:
                triaged = llm_triage_fn(d)
                if triaged:
                    label = triaged
            cache[d] = label
            labels.append(label)
        classes.append(";".join(l["class"] for l in labels))
        subclasses.append(";".join(l["subclass"] for l in labels))
        countries.append(";".join(l["country"] for l in labels))

    df = df.copy()
    df["domain_class"] = classes
    df["domain_subclass"] = subclasses
    df["domain_country"] = countries
    df["domain_taxonomy_version"] = version
    return df


def _resolve_urls(row: pd.Series, url_column: str, text_column: str) -> list[str]:
    raw = row.get(url_column)
    if isinstance(raw, list):
        return [u for u in raw if isinstance(u, str)]
    if isinstance(raw, str) and raw.strip():
        return [u for u in raw.split(";") if u.strip()]
    return extract_urls(row.get(text_column))


# ---------------------------------------------------------------------------
# Aggregation helpers (used by the article's empirical analysis)
# ---------------------------------------------------------------------------

def domain_class_share(df: pd.DataFrame, group_col: Optional[str] = None) -> pd.DataFrame:
    """
    Returns the share of each domain class in the corpus, optionally grouped
    (e.g. by channel, community, or month).

    Empty domain_class strings are ignored. Multi-URL rows are exploded so each
    URL contributes equally to the share.
    """
    work = df[["domain_class"] + ([group_col] if group_col else [])].copy()
    work = work[work["domain_class"].astype(bool)]
    work["domain_class"] = work["domain_class"].str.split(";")
    work = work.explode("domain_class")
    if group_col:
        return (
            work.groupby([group_col, "domain_class"]).size()
                .groupby(level=0)
                .apply(lambda s: s / s.sum())
                .unstack(fill_value=0.0)
        )
    return (work["domain_class"].value_counts(normalize=True)).rename("share").to_frame()
