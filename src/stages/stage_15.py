#!/usr/bin/env python3
"""
digiNEV Pipeline — stage_15.py
Auto-extracted from analyzer.py (TAREFA 11 modularização)
Enhanced with Module A — Domain Taxonomy (P1, P4)
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse


def _stage_15_domain_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 15: Análise de domínios.

    Analisa domínios e URLs para identificar padrões de mídia.
    Module A adds: domain_class, domain_subclass, domain_country,
    domain_taxonomy_version (propositions P1 and P4).
    """
    try:
        ctx.logger.info("🔄 Stage 15: Análise de domínios")

        # Backward-compatible ad-hoc domain analysis (existing columns)
        if 'domain' in df.columns:
            df['domain_type'] = df['domain'].apply(_classify_domain_type)
            df['domain_trust_score'] = df['domain'].apply(_calculate_domain_trust_score)

            domain_counts = df['domain'].value_counts()
            df['domain_frequency'] = df['domain'].map(domain_counts)

            mainstream_types = ['mainstream_news', 'government']
            df['is_mainstream_media'] = df['domain_type'].isin(mainstream_types)
        else:
            df['domain_type'] = 'unknown'
            df['domain_trust_score'] = 0.0
            df['domain_frequency'] = 0
            df['is_mainstream_media'] = False

        if 'urls_extracted' in df.columns:
            df['url_count'] = df['urls_extracted'].apply(
                lambda x: len(eval(x)) if isinstance(x, str) and x.startswith('[') else (1 if x else 0)
            )
            df['has_external_links'] = df['url_count'] > 0
        else:
            df['url_count'] = 0
            df['has_external_links'] = False

        # --- Module A: versioned domain taxonomy (P1, P4) ---
        try:
            from src.modules.module_a_domain_taxonomy.domain_taxonomy import classify_domains

            text_col = 'normalized_text' if 'normalized_text' in df.columns else 'body'
            url_col = 'urls_extracted' if 'urls_extracted' in df.columns else 'urls'
            df = classify_domains(
                df,
                url_column=url_col,
                text_column=text_col,
                llm_triage_fn=None,
            )
            ctx.logger.info(
                "✅ Module A — Domain Taxonomy aplicado "
                f"(domain_class: {df['domain_class'].astype(bool).sum()} rows com URL)"
            )
        except ImportError as exc:
            ctx.logger.warning(f"⚠️ Module A não encontrado ({exc}) — colunas taxonomy ausentes")
            for col in ('domain_class', 'domain_subclass', 'domain_country', 'domain_taxonomy_version'):
                df[col] = ''
        # --- fim Module A ---

        ctx.stats['stages_completed'] += 1
        ctx.stats['features_extracted'] += 9

        ctx.logger.info(f"✅ Stage 15 concluído: {len(df)} registros processados")
        return df

    except Exception as e:
        ctx.logger.error(f"❌ Erro Stage 15: {e}")
        ctx.stats['processing_errors'] += 1
        return df
