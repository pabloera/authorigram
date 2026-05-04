"""
Shared schema definitions for digiNEV Modules A–D.

All four modules add columns to the same message-level DataFrame produced
by the existing 17-stage pipeline. This file enumerates every new column,
its type, and the module/stage that produces it. Use the `NEW_COLUMNS`
constant to validate output schemas in tests.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    dtype: str
    module: str            # "A" | "B" | "C" | "D"
    target_stage: int      # digiNEV stage number where this column lands
    description: str
    nullable: bool = True


NEW_COLUMNS: tuple[ColumnSpec, ...] = (
    # --- Module A — Domain Taxonomy (Stage 15) -------------------------------
    ColumnSpec("domain_class",            "str",  "A", 15, "Top-level domain class (mainstream/alt-media/platform/...)"),
    ColumnSpec("domain_subclass",         "str",  "A", 15, "Fine-grained subclass (journalism-professional, video, ...)"),
    ColumnSpec("domain_country",          "str",  "A", 15, "BR / US / INT"),
    ColumnSpec("domain_taxonomy_version", "str",  "A", 15, "Version stamp of taxonomy_v*.json", nullable=False),

    # --- Module B — Flow-Stability Network (Stage 14) ------------------------
    ColumnSpec("community_id",       "int",   "B", 14, "Channel community label (per snapshot)"),
    ColumnSpec("role",               "str",   "B", 14, "upstream / core / downstream"),
    ColumnSpec("pagerank",           "float", "B", 14, "Weighted PageRank (alpha=0.85)"),
    ColumnSpec("in_degree_z",        "float", "B", 14, "Z-score of weighted in-degree"),
    ColumnSpec("out_degree_z",       "float", "B", 14, "Z-score of weighted out-degree"),
    ColumnSpec("betweenness",        "float", "B", 14, "Betweenness centrality (sampled k=200)"),
    ColumnSpec("flow_snapshot_id",   "str",   "B", 14, "YYYY-MM identifier of the snapshot used"),

    # --- Module C — Burst Detection (Stages 13+16) ---------------------------
    ColumnSpec("in_burst",  "bool", "C", 13, "True if message timestamp falls inside a detected burst", nullable=False),
    ColumnSpec("burst_id",  "str",  "C", 13, "Identifier of the burst the message belongs to"),

    # --- Module D — Frame Classifier (Stages 11+12) --------------------------
    ColumnSpec("frame_primary",        "str",   "D", 11, "Primary discursive frame (1 of 8 keys, or 'none')", nullable=False),
    ColumnSpec("frame_secondary",      "str",   "D", 11, "Secondary frame or 'none'", nullable=False),
    ColumnSpec("frame_confidence",     "float", "D", 11, "0.0-1.0 (lexicon=0.6, LLM up to 1.0)", nullable=False),
    ColumnSpec("frame_lexicon_hits",   "str",   "D", 11, "Semicolon-joined frame keys with lexicon hits"),
)


def list_module(module_letter: str) -> list[ColumnSpec]:
    return [c for c in NEW_COLUMNS if c.module == module_letter]


def required_columns(modules: tuple[str, ...] = ("A", "B", "C", "D")) -> set[str]:
    return {c.name for c in NEW_COLUMNS if c.module in modules}


# ---------------------------------------------------------------------------
# Side artifacts (written outside the message DataFrame)
# ---------------------------------------------------------------------------

SIDE_ARTIFACTS = (
    {"module": "B", "path": "outputs/network/community_snapshots/snapshot_YYYY-MM.parquet",
     "description": "Channel-level table per monthly snapshot (community, role, centralities)"},
    {"module": "B", "path": "outputs/network/community_evolution.csv",
     "description": "Long-format channel x month community/role evolution"},
    {"module": "C", "path": "outputs/temporal/bursts.csv",
     "description": "All detected bursts with start/peak/end/intensity + nearest event"},
    {"module": "C", "path": "outputs/temporal/event_windows.csv",
     "description": "Pre/post feature deltas around each Stage-16 event"},
    {"module": "C", "path": "outputs/temporal/permutation_test.json",
     "description": "Burst-event alignment permutation test result"},
    {"module": "D", "path": "outputs/frames/frame_prevalence.csv",
     "description": "Long-format frame prevalence per channel x month"},
    {"module": "D", "path": "outputs/frames/gold_set_v1.0.csv",
     "description": "Manually coded gold set (1,500-2,000 messages)"},
)
