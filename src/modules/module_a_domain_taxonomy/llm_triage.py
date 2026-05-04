"""
LLM-based fallback triage for unknown domains. Use when classify_one()
returns class='unknown' and you want the Anthropic API to attempt a
classification before sending the domain to manual review.

This is intentionally a thin wrapper: the digiNEV pipeline already owns
its Anthropic client (used by Stages 06, 08, 11, 12, 16, 17). Pass that
client in instead of importing a new one.
"""

from __future__ import annotations

from typing import Callable, Optional

# Allowed values mirror the schema in taxonomy_v1.0.json (_meta.schema)
ALLOWED_CLASS = {"mainstream", "alt-media", "platform", "government", "international", "social", "other"}
ALLOWED_COUNTRY = {"BR", "US", "INT"}

PROMPT_TEMPLATE = """You are classifying a single web domain into a fixed taxonomy
used by a Brazilian Telegram political-discourse research project.

Domain: {domain}

Return STRICT JSON with exactly three keys:
  "class":     one of {classes}
  "subclass":  one of [journalism-professional, aggregator, blog, video, messaging,
               social-network, audio, image-host, encyclopedia, official-gov,
               court, legislative, executive, search, store, unknown]
  "country":   one of {countries}

Definitions:
- mainstream:    professional Brazilian journalism (Globo, Folha, Estadao, etc.)
- alt-media:     ideologically aligned outlets, aggregators, and blogs that operate
                 outside mainstream gatekeeping (Brazilian or international diaspora)
- platform:      a hosting platform for user content (YouTube, Twitter, Telegram, etc.)
- government:    Brazilian official government domain (gov.br, jus.br, leg.br)
- international: non-Brazilian professional news outlets
- social:        general utility (Wikipedia, Google, retail)
- other:         anything that does not fit above

If you are not confident, return class="other" and subclass="unknown".

JSON:"""


def make_triage_fn(anthropic_client, model: str = "claude-haiku-4-5-20251001") -> Callable[[str], Optional[dict]]:
    """
    Build a `triage_fn(domain) -> {class, subclass, country}` closure.

    The Anthropic client must already be the same one digiNEV uses for the
    other LLM-augmented stages. Haiku is preferred here for cost reasons —
    domain classification is a low-complexity task.
    """
    import json

    def triage(domain: str) -> Optional[dict]:
        prompt = PROMPT_TEMPLATE.format(
            domain=domain,
            classes=sorted(ALLOWED_CLASS),
            countries=sorted(ALLOWED_COUNTRY),
        )
        try:
            resp = anthropic_client.messages.create(
                model=model,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text.strip()
            # Pull the first {...} block in case the model added prose.
            start, end = text.find("{"), text.rfind("}")
            if start == -1 or end == -1:
                return None
            data = json.loads(text[start : end + 1])
            if data.get("class") not in ALLOWED_CLASS:
                return None
            if data.get("country") not in ALLOWED_COUNTRY:
                data["country"] = "INT"
            return data
        except Exception:
            return None

    return triage
