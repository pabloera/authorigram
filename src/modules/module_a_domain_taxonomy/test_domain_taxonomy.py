"""Minimal smoke tests for module_a_domain_taxonomy."""

from __future__ import annotations

import pandas as pd

from src.modules.module_a_domain_taxonomy.domain_taxonomy import (
    classify_domains,
    classify_one,
    domain_class_share,
    extract_urls,
    load_taxonomy,
    normalize_domain,
)


def test_extract_urls_basic():
    text = "Veja https://g1.globo.com/politica e tambem https://t.me/canal/123"
    urls = extract_urls(text)
    assert len(urls) == 2
    assert urls[0].startswith("https://g1.globo.com")


def test_normalize_domain_strips_www_and_keeps_known_subdomain():
    assert normalize_domain("https://www.globo.com/x") == "globo.com"
    assert normalize_domain("https://g1.globo.com/y") == "g1.globo.com"
    assert normalize_domain("https://noticias.uol.com.br/z") == "noticias.uol.com.br"


def test_classify_one_known_domain():
    tax = load_taxonomy()
    label = classify_one("g1.globo.com", tax)
    assert label["class"] == "mainstream"
    assert label["country"] == "BR"


def test_classify_one_falls_back_to_base():
    tax = load_taxonomy()
    # not in dict at this exact subdomain, but globo.com is
    label = classify_one("esporte.globo.com", tax)
    assert label["class"] == "mainstream"


def test_classify_one_unknown_returns_unknown():
    tax = load_taxonomy()
    label = classify_one("totalmente-novo-xyz.example", tax)
    assert label["class"] == "unknown"


def test_classify_domains_dataframe_smoke():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "text": [
                "Confira https://g1.globo.com/x",
                "Olha https://youtube.com/watch?v=abc e https://gettr.com/p/y",
                "Sem URL aqui.",
            ],
        }
    )
    out = classify_domains(df, url_column="urls", text_column="text")
    assert "domain_class" in out.columns
    assert out.loc[0, "domain_class"] == "mainstream"
    assert out.loc[1, "domain_class"] == "platform;platform"
    assert out.loc[2, "domain_class"] == ""


def test_domain_class_share_global():
    df = pd.DataFrame(
        {
            "domain_class": ["mainstream", "platform;platform", "alt-media", ""],
        }
    )
    share = domain_class_share(df)
    assert abs(share["share"].sum() - 1.0) < 1e-9
    assert share.loc["platform", "share"] == 0.5  # 2 of 4 url-events
