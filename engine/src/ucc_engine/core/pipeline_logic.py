
# =============================================================================
# DAGSTER ASSETS
# =============================================================================
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from pathlib import Path
from ..compat import asset, AssetExecutionContext, Output, MetadataValue, MaterializeResult
import gc
import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'  # macOS Accelerate framework
os.environ['NUMEXPR_NUM_THREADS'] = '1'
from datetime import datetime
from collections import defaultdict
from .utils import (
    soundex,
    detect_entity_type,
    normalize_name,
    generate_ngrams,
    generate_ngrams_frozen,
    get_meaningful_words,
    _get_meaningful_words_frozen,
    _long_tokens,
    ngram_similarity,
    tfidf_ngram_similarity,
    has_shared_meaningful_identifier,
    extract_entity_core,
    extract_acronym,
    is_acronym_match,
    compute_match_score,
    compute_match_score_fast,
    combined_fuzzy_score,
    min_token_overlap,
    entity_types_compatible,
    should_merge_clusters,
    build_tfidf_vec,
    ENTITY_TYPE_WORDS,
)
from .models.data_structures import NGramIndex, UnionFind, TFIDFNGramIndex
from ..config import PipelineConfig


STATE_COLUMNS = ["DEBT_STATE", "SECP_STATE"]
RAW_MATCH_COLUMNS = STATE_COLUMNS + ["SECP_NAME"]
STATE_MISSING_SENTINEL = "__MISSING_STATE__"
NAME_MISSING_SENTINEL = "__MISSING_NAME__"
NO_MASTER_STATE_PAIR = "NO_MASTER_STATE_PAIR"


def _standardize_state_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize state columns so state-aware joins use stable keys."""
    df = df.copy()
    for col in STATE_COLUMNS:
        if col in df.columns:
            state = df[col].astype("string").str.upper().str.strip()
            df[col] = state.mask(state.eq(""))
    return df


def _clean_key_component(series: pd.Series, *, uppercase: bool = True) -> pd.Series:
    values = series.astype("string").str.strip()
    if uppercase:
        values = values.str.upper()
    return values.mask(values.eq("")).fillna(NAME_MISSING_SENTINEL)


def _state_pair_keys(df: pd.DataFrame) -> pd.Series:
    debt = _clean_key_component(df["DEBT_STATE"], uppercase=True).replace(
        NAME_MISSING_SENTINEL, STATE_MISSING_SENTINEL
    )
    secp = _clean_key_component(df["SECP_STATE"], uppercase=True).replace(
        NAME_MISSING_SENTINEL, STATE_MISSING_SENTINEL
    )
    return pd.Series(list(zip(debt, secp)), index=df.index)


def _state_name_keys(
    df: pd.DataFrame,
    name_col: str,
    *,
    normalized: bool = False,
) -> pd.Series:
    state_pairs = _state_pair_keys(df)
    if normalized:
        names = df[name_col].apply(normalize_name).astype("string")
    else:
        names = _clean_key_component(df[name_col], uppercase=True)
    names = names.mask(names.str.strip().eq("")).fillna(NAME_MISSING_SENTINEL)
    return pd.Series(
        [(debt, secp, name) for (debt, secp), name in zip(state_pairs, names)],
        index=df.index,
    )


def _unique_candidates(series: pd.Series) -> List[str]:
    candidates = []
    for value in series:
        if value is None or pd.isna(value):
            continue
        candidate = str(value).strip()
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _preview_markdown(df: pd.DataFrame, rows: int = 5) -> str:
    """Render Dagster preview metadata without exposing tabulate to pandas NA."""
    return df.head(rows).astype("object").where(pd.notna(df.head(rows)), "").to_markdown()


def _detect_delimiter(file_path: str) -> str:
    """Detect comma vs tab-delimited input from the header row."""
    with open(file_path, "r", encoding="latin-1") as f:
        header = f.readline()
    return "\t" if header.count("\t") > header.count(",") else ","


def _pending_match_mask(df: pd.DataFrame) -> pd.Series:
    return (
        df['SECP_NAME_MAPPED'].isna()
        & df['SECP_NAME_NORMALIZED'].notna()
        & (df['SECP_NAME_NORMALIZED'].str.len() > 0)
        & (df['confidence'].fillna("") != NO_MASTER_STATE_PAIR)
    )


@asset(
    description="Load raw UCC filing data from CSV",
    compute_kind="pandas",
    group_name="ingestion",
    owners=["team:data-engineering"],
    tags={"domain": "ucc-mapping", "stage": "ingestion"},
)
def ucc_raw_data(context: AssetExecutionContext, config: PipelineConfig) -> Output[pd.DataFrame]:
    """Load and deduplicate UCC filing data."""
    df = pd.read_csv(
        config.ucc_file,
        sep=_detect_delimiter(config.ucc_file),
        encoding="latin-1",
        usecols=RAW_MATCH_COLUMNS,
    ).drop_duplicates()
    df = _standardize_state_columns(df)
    
    df = df[df['SECP_NAME'].notna() & (df['SECP_NAME'].str.strip() != '')]
    df = df.reset_index(drop=True)
    
    context.log.info(f"Loaded {len(df):,} unique UCC records")
    
    # Return Output with metadata facets
    return Output(
        value=df,
        metadata={
            "row_count": MetadataValue.int(len(df)),
            "preview": MetadataValue.md(_preview_markdown(df)),
            "columns": MetadataValue.json(list(df.columns)),
            "source_file": MetadataValue.text(config.ucc_file),
        }
    )


@asset(
    description="Load master mapping file with SECP_NAME to SECP_NAME_MAPPED",
    compute_kind="pandas",
    group_name="ingestion",
    tags={"domain": "ucc-mapping", "stage": "ingestion"},
)
def master_mapping_data(context: AssetExecutionContext, config: PipelineConfig) -> Output[pd.DataFrame]:
    """Load master mapping file."""
    df = pd.read_csv(
        config.master_file,
        sep=_detect_delimiter(config.master_file),
        encoding='latin-1',
        usecols=RAW_MATCH_COLUMNS + ['SECP_NAME_MAPPED', 'Firm ID'],
        dtype={'Firm ID': 'Int64'},
    ).drop_duplicates()
    df = _standardize_state_columns(df)
    
    df = df[df['SECP_NAME'].notna() & (df['SECP_NAME'].str.strip() != '')]
    df = df.reset_index(drop=True)
    
    context.log.info(f"Loaded {len(df):,} master mapping records")
    
    return Output(
        value=df,
        metadata={
            "row_count": MetadataValue.int(len(df)),
            "preview": MetadataValue.md(_preview_markdown(df)),
            "source_file": MetadataValue.text(config.master_file),
        }
    )


@asset(
    description="Precompute normalized master data for faster matching",
    compute_kind="python",
    group_name="ingestion",
    tags={"domain": "ucc-mapping", "stage": "ingestion"},
)
def normalized_master_data(
    context: AssetExecutionContext, 
    master_mapping_data: pd.DataFrame
) -> dict:
    """Precompute normalized master lookups."""
    master_df = master_mapping_data.copy()
    master_df['SECP_NAME_NORMALIZED'] = master_df['SECP_NAME'].apply(normalize_name)
    master_df = master_df[master_df['SECP_NAME_NORMALIZED'].str.len() > 0]
    master_df['_RAW_STATE_KEY'] = _state_name_keys(master_df, 'SECP_NAME')
    master_df['_NORM_STATE_KEY'] = _state_name_keys(master_df, 'SECP_NAME_NORMALIZED', normalized=False)
    master_df['_STATE_PAIR_KEY'] = _state_pair_keys(master_df)

    raw_candidates = (
        master_df.groupby('_RAW_STATE_KEY')['SECP_NAME_MAPPED']
        .apply(_unique_candidates)
        .to_dict()
    )
    norm_candidates = (
        master_df.groupby('_NORM_STATE_KEY')['SECP_NAME_MAPPED']
        .apply(_unique_candidates)
        .to_dict()
    )
    ambiguous_raw_keys = {key for key, values in raw_candidates.items() if len(values) > 1}
    ambiguous_norm_keys = {key for key, values in norm_candidates.items() if len(values) > 1}

    # Keep first occurrence for each normalized name within its debtor/secured-party state pair.
    raw_dedup = master_df.drop_duplicates('_RAW_STATE_KEY', keep='first')
    master_dedup = master_df.drop_duplicates('_NORM_STATE_KEY', keep='first').reset_index(drop=True)
    
    # Pre-compute entity types for all master names (avoids recomputation in matching)
    normalized_list = master_dedup['SECP_NAME_NORMALIZED'].tolist()
    entity_types = [detect_entity_type(name) for name in normalized_list]
    state_pair_list = master_dedup['_STATE_PAIR_KEY'].tolist()
    state_pair_to_indices = {}
    for idx, state_pair in enumerate(state_pair_list):
        state_pair_to_indices.setdefault(state_pair, []).append(idx)
    
    # Stage 2b lookup: normalised name -> the one bank it maps to across ALL state
    # pairs (names that map to several banks nationally - PINNACLE BANK, FIRST BANK -
    # are left out; they are state-dependent and must not match across states).
    global_groups = master_df.groupby('SECP_NAME_NORMALIZED')
    global_norm_to_mapped = {}
    for norm, grp in global_groups:
        banks = _unique_candidates(grp['SECP_NAME_MAPPED'])
        if len(banks) == 1:
            first = grp.iloc[0]
            global_norm_to_mapped[norm] = (banks[0], first['SECP_NAME'], first['Firm ID'])

    # Create lookups
    result = {
        'global_norm_to_mapped': global_norm_to_mapped,
        'raw_key_to_mapped': dict(zip(raw_dedup['_RAW_STATE_KEY'], raw_dedup['SECP_NAME_MAPPED'])),
        'raw_key_to_raw': dict(zip(raw_dedup['_RAW_STATE_KEY'], raw_dedup['SECP_NAME'])),
        'raw_key_to_firm_id': dict(zip(raw_dedup['_RAW_STATE_KEY'], raw_dedup['Firm ID'])),
        'raw_key_to_mapped_candidates': raw_candidates,
        'ambiguous_raw_keys': ambiguous_raw_keys,
        'norm_to_mapped': dict(zip(master_dedup['_NORM_STATE_KEY'], master_dedup['SECP_NAME_MAPPED'])),
        'norm_to_raw': dict(zip(master_dedup['_NORM_STATE_KEY'], master_dedup['SECP_NAME'])),
        'norm_to_firm_id': dict(zip(master_dedup['_NORM_STATE_KEY'], master_dedup['Firm ID'])),
        'normalized_list': normalized_list,
        'raw_list': master_dedup['SECP_NAME'].tolist(),
        'mapped_list': master_dedup['SECP_NAME_MAPPED'].tolist(),
        'firm_id_list': master_dedup['Firm ID'].tolist(),
        'state_pair_list': state_pair_list,
        'state_pair_to_indices': state_pair_to_indices,
        'entity_types': entity_types,  # Pre-computed entity types
        'ambiguous_norms': ambiguous_norm_keys,
        'norm_to_mapped_candidates': norm_candidates,
    }
    
    context.log.info(f"Prepared {len(result['normalized_list']):,} normalized master records")
    context.log.info(f"Found {len(ambiguous_raw_keys):,} ambiguous raw state/name keys")
    context.log.info(f"Found {len(ambiguous_norm_keys):,} ambiguous normalized state/name keys")
    context.log.info(f"Pre-computed entity types for all master records")
    
    return result


@asset(
    description="Pre-compute master embeddings (CPU OPTIMIZED)",
    compute_kind="ml",
    group_name="preprocessing",
    tags={"domain": "ucc-mapping", "stage": "preprocessing"},
)
def master_embeddings_cache(
        context: AssetExecutionContext,
        config: PipelineConfig,
        normalized_master_data: dict,
) -> dict:
    """Pre-compute master embeddings (CPU OPTIMIZED)."""
    from ..embeddings import load_embedding_model
    import faiss
    import os
    from concurrent.futures import ThreadPoolExecutor

    # Optimize CPU usage
    if config.cpu_threads > 0:
        os.environ['OMP_NUM_THREADS'] = str(config.cpu_threads)
        os.environ['OPENBLAS_NUM_THREADS'] = str(config.cpu_threads)
        os.environ['MKL_NUM_THREADS'] = str(config.cpu_threads)

    master_normalized_list = normalized_master_data['normalized_list']

    if not master_normalized_list:
        context.log.info("No master data to embed")
        return {'embeddings': None, 'faiss_index': None, 'ngram_index': None}

    context.log.info("Loading embedding model for CPU...")
    model = load_embedding_model(config, context)

    # CPU-optimized encoding
    context.log.info(f"Encoding {len(master_normalized_list):,} master mappings (CPU)...")
    master_embeddings = model.encode(
        master_normalized_list,
        batch_size=config.embed_batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        device='cpu',
        convert_to_numpy=True
    )
    master_embeddings = np.ascontiguousarray(master_embeddings, dtype="float32")

    # Optimized FAISS index for CPU
    context.log.info("Building CPU-optimized FAISS index...")
    faiss.omp_set_num_threads(config.cpu_threads if config.cpu_threads > 0 else -1)

    # For CPU, stick with IndexFlatIP for most cases (it's faster than HNSW on CPU for moderate sizes)
    if len(master_normalized_list) > 100000:
        # Only use HNSW for very large datasets
        faiss_index = faiss.IndexHNSWFlat(master_embeddings.shape[1], 16)  # Smaller M for CPU
        faiss_index.hnsw.efSearch = 64  # Reduced for speed
        faiss_index.hnsw.efConstruction = 100
    else:
        faiss_index = faiss.IndexFlatIP(master_embeddings.shape[1])

    faiss_index.add(master_embeddings)

    # Parallel auxiliary computations
    context.log.info("Computing auxiliary data in parallel...")

    def compute_ngram_batch(args):
        start, end, names = args
        local_ngrams = {}
        for i in range(start, end):
            local_ngrams[i] = generate_ngrams(names[i], 3)
        return local_ngrams

    def compute_soundex_batch(names):
        return [soundex(name) for name in names]

    # Build n-gram index in parallel
    ngram_index = NGramIndex(n=3)
    chunk_size = max(1000, len(master_normalized_list) // config.embedding_workers)
    chunks = [
        (i, min(i + chunk_size, len(master_normalized_list)), master_normalized_list)
        for i in range(0, len(master_normalized_list), chunk_size)
    ]

    with ThreadPoolExecutor(max_workers=config.embedding_workers) as executor:
        # Compute n-grams in parallel
        ngram_results = executor.map(compute_ngram_batch, chunks)
        for ngram_dict in ngram_results:
            for idx, ngrams in ngram_dict.items():
                ngram_index.add_precomputed(idx, master_normalized_list[idx], ngrams)

        # Compute soundex in parallel
        master_soundex = compute_soundex_batch(master_normalized_list)

    # Switch n-gram index to vectorised numpy query mode (big speedup in Stage 4)
    ngram_index.freeze()

    # Build TF-IDF weighted n-gram index for better n-gram similarity
    context.log.info("Building TF-IDF n-gram index...")
    tfidf_index = TFIDFNGramIndex(n=3)
    for i, name in enumerate(master_normalized_list):
        tfidf_index.add(i, name)
    tfidf_index.freeze()
    # Pre-compute per-master TF-IDF vectors so Stage 4 doesn't re-tokenise.
    master_tfidf_vecs, master_tfidf_norms = tfidf_index.build_vectors()
    context.log.info(
        f"  TF-IDF index: {len(tfidf_index._idf):,} unique n-grams weighted, "
        f"{len(master_tfidf_vecs):,} master vectors pre-built"
    )

    # --- Master-side derived features ----------------------------------------
    # Building these once at cache time turns Stage 3/4/6 inner loops from
    # Python-heavy into pure dict/array lookups.
    context.log.info("Pre-computing master-side derived features...")
    master_long_tokens = [_long_tokens(n) for n in master_normalized_list]
    master_meaningful_words = [_get_meaningful_words_frozen(n) for n in master_normalized_list]
    master_entity_cores = [extract_entity_core(n) for n in master_normalized_list]

    # Token → master_id inverted index for fast token-based blocking in Stage 3/4.
    # Uses meaningful tokens only (drops generic noise like BANK/INC).
    token_to_master_ids: Dict[str, np.ndarray] = {}
    token_to_ids_build: Dict[str, list] = defaultdict(list)
    for idx, toks in enumerate(master_meaningful_words):
        for tok in toks:
            token_to_ids_build[tok].append(idx)
    for tok, ids in token_to_ids_build.items():
        token_to_master_ids[tok] = np.asarray(ids, dtype=np.int64)
    context.log.info(
        f"  Token block index: {len(token_to_master_ids):,} unique tokens"
    )

    context.log.info(f"Master embeddings cache ready: {master_embeddings.shape}")

    # -------------------------------------------------------------------------
    # Per-state-pair FAISS indices (Stage 4 perf upgrade).
    # Each state pair gets its own IndexFlatIP over the rows in that state.
    # Stage 4 looks up the index for the query's state pair and runs a single
    # FAISS search instead of a per-batch np.matmul. Mathematically identical
    # to the matmul path (both use exact inner product on normalised vectors).
    # Skipped for tiny groups (overhead > savings).
    # -------------------------------------------------------------------------
    per_state_faiss: Dict[str, dict] = {}
    if getattr(config, 'advanced_use_per_state_faiss', False):
        min_state_size = getattr(config, 'advanced_per_state_min_size', 64)
        state_pair_to_indices_local = normalized_master_data['state_pair_to_indices']
        n_built = 0
        n_rows_indexed = 0
        for state_pair, idx_list in state_pair_to_indices_local.items():
            if len(idx_list) < min_state_size:
                continue  # tiny groups stay on matmul path
            idx_arr = np.asarray(idx_list, dtype=np.int64)
            sub_emb = np.ascontiguousarray(master_embeddings[idx_arr], dtype="float32")
            sub_index = faiss.IndexFlatIP(sub_emb.shape[1])
            sub_index.add(sub_emb)
            per_state_faiss[state_pair] = {
                'index': sub_index,
                'master_indices': idx_arr,  # row -> global master idx
            }
            n_built += 1
            n_rows_indexed += len(idx_arr)
        context.log.info(
            f"  Built {n_built:,} per-state FAISS indices covering "
            f"{n_rows_indexed:,}/{len(master_normalized_list):,} master rows "
            f"(state-pairs with < {min_state_size} rows stay on matmul path)"
        )

    return {
        'embeddings': master_embeddings,
        'faiss_index': faiss_index,
        'ngram_index': ngram_index,
        'tfidf_idf_weights': tfidf_index._idf,  # pass IDF weights for compute_match_score
        'master_tfidf_vecs': master_tfidf_vecs,
        'master_tfidf_norms': master_tfidf_norms,
        'master_long_tokens': master_long_tokens,
        'master_meaningful_words': master_meaningful_words,
        'master_entity_cores': master_entity_cores,
        'token_to_master_ids': token_to_master_ids,
        'per_state_faiss': per_state_faiss,
        'soundex': master_soundex,
        'model': model,
    }


@asset(
    description="Stage 1: Exact match on DEBT_STATE, SECP_STATE, and raw SECP_NAME",
    compute_kind="python",
    group_name="matching",
    tags={"domain": "ucc-mapping", "stage": "matching"},
)
def exact_match_raw_results(
    context: AssetExecutionContext,
    ucc_raw_data: pd.DataFrame,
    normalized_master_data: dict,
) -> pd.DataFrame:
    """Stage 1: Exact match on raw SECP_NAME within the same state pair."""
    df = ucc_raw_data.copy()
    raw_keys = _state_name_keys(df, 'SECP_NAME')
    master_lookup = normalized_master_data['raw_key_to_mapped']
    raw_to_master = normalized_master_data['raw_key_to_raw']
    raw_to_firm_id = normalized_master_data['raw_key_to_firm_id']
    ambiguous_raw_keys = normalized_master_data['ambiguous_raw_keys']
    raw_candidates = normalized_master_data['raw_key_to_mapped_candidates']
    
    df['SECP_NAME_MAPPED'] = raw_keys.map(master_lookup)
    df['matched_master_raw'] = raw_keys.map(raw_to_master)
    df['Firm_ID'] = raw_keys.map(raw_to_firm_id)
    df['match_method'] = np.where(df['SECP_NAME_MAPPED'].notna(), 'Exact Match (Raw)', None)
    df['match_score'] = np.where(df['SECP_NAME_MAPPED'].notna(), 1.0, np.nan)
    df['confidence'] = np.where(df['SECP_NAME_MAPPED'].notna(), 'AUTO', None)
    df['ambiguous_master_candidates'] = None

    ambiguous_mask = raw_keys.isin(ambiguous_raw_keys) & df['SECP_NAME_MAPPED'].notna()
    if ambiguous_mask.any():
        df.loc[ambiguous_mask, 'match_method'] = 'Exact Match (Raw - State Ambiguous)'
        df.loc[ambiguous_mask, 'confidence'] = 'REVIEW_HIGH'
        df.loc[ambiguous_mask, 'ambiguous_master_candidates'] = raw_keys[ambiguous_mask].map(
            lambda key: "; ".join(raw_candidates.get(key, []))
        )
    
    matched = df['SECP_NAME_MAPPED'].notna().sum()
    ambiguous_count = int(ambiguous_mask.sum())
    context.log.info(
        f"Stage 1 - Exact Match (Raw + states): "
        f"{matched - ambiguous_count:,} auto + {ambiguous_count:,} ambiguous"
    )
    
    return df


@asset(
    description="Stage 2: Exact match on normalized names",
    compute_kind="python",
    group_name="matching",
    tags={"domain": "ucc-mapping", "stage": "matching"},
)
def exact_match_normalized_results(
    context: AssetExecutionContext,
    exact_match_raw_results: pd.DataFrame,
    normalized_master_data: dict,
) -> pd.DataFrame:
    """Stage 2: Exact match on normalized names."""
    df = exact_match_raw_results.copy()
    
    unmatched_mask = df['SECP_NAME_MAPPED'].isna()
    if not unmatched_mask.any():
        context.log.info("Stage 2 - Exact Match (Normalized): 0 matches (none to process)")
        df['SECP_NAME_NORMALIZED'] = df['SECP_NAME'].apply(normalize_name)
        return df
    
    df['SECP_NAME_NORMALIZED'] = df['SECP_NAME'].apply(normalize_name)
    
    norm_keys = _state_name_keys(df, 'SECP_NAME_NORMALIZED', normalized=False)
    norm_to_mapped = normalized_master_data['norm_to_mapped']
    norm_to_raw = normalized_master_data['norm_to_raw']
    norm_to_firm_id = normalized_master_data['norm_to_firm_id']
    ambiguous_norms = normalized_master_data['ambiguous_norms']
    norm_to_mapped_candidates = normalized_master_data['norm_to_mapped_candidates']
    state_pair_to_indices = normalized_master_data['state_pair_to_indices']
    
    matches = norm_keys[unmatched_mask].map(norm_to_mapped)
    matched_mask = unmatched_mask & matches.notna().reindex(df.index, fill_value=False)
    
    # Wrap RHS in list() so pandas does element-wise dtype coercion;
    # otherwise int-valued Firm_IDs raise LossySetitemError against the
    # float64 column (pandas >= 2.2 strict casting).
    df.loc[matched_mask, 'SECP_NAME_MAPPED'] = list(norm_keys[matched_mask].map(norm_to_mapped))
    df.loc[matched_mask, 'matched_master_raw'] = list(norm_keys[matched_mask].map(norm_to_raw))
    df.loc[matched_mask, 'Firm_ID'] = list(norm_keys[matched_mask].map(norm_to_firm_id))
    df.loc[matched_mask, 'match_score'] = 1.0
    
    matched_norm_keys = norm_keys[matched_mask]
    ambiguous_mask = matched_norm_keys.isin(ambiguous_norms)
    matched_indices = matched_norm_keys.index
    
    df.loc[matched_indices, 'match_method'] = 'Exact Match (Normalized)'
    df.loc[matched_indices, 'confidence'] = 'AUTO'
    
    if ambiguous_mask.any():
        amb_idx = matched_norm_keys[ambiguous_mask].index
        df.loc[amb_idx, 'match_method'] = 'Exact Match (Normalized - Ambiguous)'
        df.loc[amb_idx, 'confidence'] = 'REVIEW_HIGH'
        df.loc[amb_idx, 'ambiguous_master_candidates'] = [
            "; ".join(norm_to_mapped_candidates.get(norm_keys.at[i], []))
            for i in amb_idx
        ]
    
    matched = matched_mask.sum()
    ambiguous_count = ambiguous_mask.sum()
    context.log.info(f"Stage 2 - Exact Match (Normalized): {matched - ambiguous_count:,} auto + {ambiguous_count:,} ambiguous")

    # Stage 2b: the same normalised name is in the master under ANOTHER state pair and
    # maps to exactly one bank nationally (US BANK TRUST in AR->IL when the master only
    # has it for other pairs). A suggestion for review, never auto: a name that maps to
    # one bank today may just be the only one of that name seen so far.
    global_norm = normalized_master_data.get('global_norm_to_mapped') or {}
    if global_norm:
        still = df['SECP_NAME_MAPPED'].isna()
        hits = df.loc[still, 'SECP_NAME_NORMALIZED'].map(global_norm)
        hits = hits[hits.notna()]
        if len(hits):
            idx = hits.index
            df.loc[idx, 'SECP_NAME_MAPPED'] = [h[0] for h in hits]
            df.loc[idx, 'matched_master_raw'] = [h[1] for h in hits]
            df.loc[idx, 'Firm_ID'] = [h[2] for h in hits]
            df.loc[idx, 'match_score'] = 1.0
            df.loc[idx, 'match_method'] = 'Exact Match (Normalized - Other State)'
            df.loc[idx, 'confidence'] = 'REVIEW_HIGH'
            context.log.info(
                f"Stage 2b - Exact Match (Normalized, other state pair): {len(hits):,} review suggestions "
                f"({hits.map(lambda h: h[0]).nunique():,} banks)"
            )

    remaining_unmatched = df['SECP_NAME_MAPPED'].isna()
    state_pairs = _state_pair_keys(df)
    no_state_pair_mask = remaining_unmatched & ~state_pairs.isin(state_pair_to_indices.keys())
    if no_state_pair_mask.any():
        df.loc[no_state_pair_mask, 'match_method'] = 'No Master State Pair'
        df.loc[no_state_pair_mask, 'confidence'] = NO_MASTER_STATE_PAIR
        context.log.info(
            f"Stage 2 - Skipped {int(no_state_pair_mask.sum()):,} records "
            "with state pairs absent from the Master Mapping file"
        )
    
    return df


@asset(
    description="Stage 3: Fuzzy matching with enhanced false positive prevention",
    compute_kind="python",
    group_name="matching",
    tags={"domain": "ucc-mapping", "stage": "matching"},
)
def fuzzy_match_results(
        context: AssetExecutionContext,
        config: PipelineConfig,
        exact_match_normalized_results: pd.DataFrame,
        normalized_master_data: dict,
        master_embeddings_cache: dict,
) -> pd.DataFrame:
    """Stage 3: Fuzzy matching with enhanced false-positive prevention.

    Performance:
      * Token-based blocking via ``token_to_master_ids`` collapses the
        ``Q x M_state`` matrix into ``Q x C`` where C is typically 5-50x smaller.
      * ``process.cdist`` runs with ``dtype=np.uint8`` and ``workers=-1``
        (rapidfuzz releases the GIL).
      * All ``df.loc`` writes are deferred and applied in a single vectorised
        pass at the end - eliminates ~7 Pandas indexer calls per matched row.
    """
    from rapidfuzz import process, fuzz

    df = exact_match_normalized_results.copy()

    unmatched_mask = _pending_match_mask(df)
    if not unmatched_mask.any():
        context.log.info("Stage 3 - Fuzzy Match: 0 matches (none to process)")
        return df

    master_normalized_list = normalized_master_data['normalized_list']
    master_entity_types = normalized_master_data['entity_types']
    norm_to_mapped = normalized_master_data['norm_to_mapped']
    norm_to_raw = normalized_master_data['norm_to_raw']
    norm_to_firm_id = normalized_master_data['norm_to_firm_id']
    ambiguous_norms = normalized_master_data['ambiguous_norms']
    norm_to_mapped_candidates = normalized_master_data['norm_to_mapped_candidates']
    state_pair_to_indices = normalized_master_data['state_pair_to_indices']

    # Cached master-side features (built in master_embeddings_cache asset)
    master_meaningful_words = master_embeddings_cache.get('master_meaningful_words')
    master_entity_cores = master_embeddings_cache.get('master_entity_cores')
    token_to_master_ids = master_embeddings_cache.get('token_to_master_ids')

    pending = df.loc[unmatched_mask, STATE_COLUMNS + ['SECP_NAME_NORMALIZED']].copy()
    pending['_match_key'] = _state_name_keys(pending, 'SECP_NAME_NORMALIZED', normalized=False)

    # Fast key -> indices map (avoids groupby.apply which is slow for many groups)
    key_to_indices: Dict[Any, list] = defaultdict(list)
    for row_idx, key in zip(pending.index.to_numpy(), pending['_match_key'].to_numpy()):
        key_to_indices[key].append(row_idx)

    unique_pending = pending.drop_duplicates('_match_key', keep='first')

    queries = unique_pending['SECP_NAME_NORMALIZED'].tolist()
    query_keys = unique_pending['_match_key'].tolist()
    query_state_pairs = _state_pair_keys(unique_pending).tolist()

    deduped_count = int(unmatched_mask.sum()) - len(queries)
    if deduped_count:
        context.log.info(
            f"Stage 3 - Fuzzy Match: deduped {deduped_count:,} repeated normalized state/name records"
        )

    # Pre-compute query metadata once (cached at utils level too)
    query_entity_types = [detect_entity_type(q) for q in queries]
    query_meaningful_words = [_get_meaningful_words_frozen(q) for q in queries]
    query_entity_cores = [extract_entity_core(q) for q in queries]

    # --- Stage scratch arrays (numpy-backed, no df.loc churn in hot loop) ---
    n_rows = len(df)
    pending_pos = df.index.get_indexer(pending.index)  # positions of pending rows in df
    # Build write buffers keyed by *df position*; flushed at the end.
    write_mapped = np.empty(n_rows, dtype=object); write_mapped.fill(None)
    write_raw = np.empty(n_rows, dtype=object); write_raw.fill(None)
    write_firm = np.empty(n_rows, dtype=object); write_firm.fill(None)
    write_score = np.full(n_rows, np.nan, dtype=float)
    write_method = np.empty(n_rows, dtype=object); write_method.fill(None)
    write_conf = np.empty(n_rows, dtype=object); write_conf.fill(None)
    write_ambig = np.empty(n_rows, dtype=object); write_ambig.fill(None)
    pos_index = df.index.get_indexer(df.index)  # identity; used below as np.array fast lookup
    df_index_to_pos = {idx: pos for pos, idx in enumerate(df.index.tolist())}

    FUZZY_COMBINED_MIN = 85.0
    auto_count = 0
    review_count = 0
    rejected_count = 0

    fuzzy_workers = config.fuzzy_workers if config.fuzzy_workers else -1
    batch_size = max(1, config.fuzzy_batch_size)
    review_threshold = config.fuzzy_review_threshold
    auto_threshold = config.fuzzy_auto_threshold

    for start in range(0, len(queries), batch_size):
        end = min(start + batch_size, len(queries))
        batch_queries = queries[start:end]
        batch_keys = query_keys[start:end]
        batch_query_types = query_entity_types[start:end]
        batch_state_pairs = query_state_pairs[start:end]
        batch_query_words = query_meaningful_words[start:end]
        batch_query_cores = query_entity_cores[start:end]

        # Group positions by state pair, then within each state group additionally
        # block by token overlap to slash candidate counts.
        positions_by_state_pair: Dict[Any, list] = defaultdict(list)
        for pos, state_pair in enumerate(batch_state_pairs):
            positions_by_state_pair[state_pair].append(pos)

        for state_pair, positions in positions_by_state_pair.items():
            state_candidate_arr = state_pair_to_indices.get(state_pair, [])
            if len(state_candidate_arr) == 0:
                continue
            state_candidate_set = set(state_candidate_arr)

            # ---- Per-query token blocking ------------------------------------
            # For each query, compute its candidate subset within this state.
            # Then group queries that share the *same* candidate set into one
            # cdist call - dramatically cheaper than per-query process.extract.
            per_query_candidates: list = []
            for pos in positions:
                qwords = batch_query_words[pos]
                if not qwords or token_to_master_ids is None:
                    cand_ids = state_candidate_set
                else:
                    cand_ids = set()
                    for tok in qwords:
                        arr = token_to_master_ids.get(tok)
                        if arr is not None:
                            cand_ids.update(arr.tolist())
                    cand_ids &= state_candidate_set
                    if not cand_ids:
                        # No token-block hit - fall back to whole state pool
                        # (preserves recall vs old code)
                        cand_ids = state_candidate_set
                per_query_candidates.append(cand_ids)

            # Group queries by identical candidate sets (heavy dedup expected
            # because many queries share the same meaningful tokens)
            grouped: Dict[frozenset, list] = defaultdict(list)
            for local_pos, cand in zip(positions, per_query_candidates):
                grouped[frozenset(cand)].append(local_pos)

            for cand_fset, group_positions in grouped.items():
                if not cand_fset:
                    continue
                cand_list = list(cand_fset)
                candidate_names = [master_normalized_list[i] for i in cand_list]
                sub_queries = [batch_queries[p] for p in group_positions]

                scores = process.cdist(
                    sub_queries,
                    candidate_names,
                    scorer=fuzz.token_sort_ratio,
                    score_cutoff=review_threshold,
                    workers=fuzzy_workers,
                    dtype=np.uint8,
                )
                if scores.size == 0:
                    continue
                # uint8 scores 0..100; quick max gate
                row_max = scores.max(axis=1)
                if row_max.max() < review_threshold:
                    continue

                best_idx = scores.argmax(axis=1)

                for sub_i, pos in enumerate(group_positions):
                    score = int(row_max[sub_i])
                    if score < review_threshold:
                        continue

                    match_local = int(best_idx[sub_i])
                    match_idx = cand_list[match_local]
                    match_norm = master_normalized_list[match_idx]
                    query_name = batch_queries[pos]

                    # Entity-type gating
                    query_type = batch_query_types[pos]
                    match_type = master_entity_types[match_idx]
                    if not entity_types_compatible(query_type, match_type):
                        rejected_count += 1
                        continue

                    # Meaningful-identifier overlap (uses frozensets)
                    if not has_shared_meaningful_identifier(query_name, match_norm):
                        rejected_count += 1
                        continue

                    # Combined fuzzy score (cached)
                    combined_score = combined_fuzzy_score(query_name, match_norm)
                    if combined_score < FUZZY_COMBINED_MIN:
                        rejected_count += 1
                        continue

                    # Entity-core mismatch guard
                    query_core = batch_query_cores[pos]
                    match_core = master_entity_cores[match_idx] if master_entity_cores else extract_entity_core(match_norm)
                    if query_core != match_core and len(query_core) > 2 and len(match_core) > 2:
                        if fuzz.ratio(query_core, match_core) < 70:
                            if not (is_acronym_match(query_core, match_core)
                                    or is_acronym_match(match_core, query_core)):
                                rejected_count += 1
                                continue

                    final_score = combined_score
                    match_key = (state_pair[0], state_pair[1], match_norm)
                    is_ambiguous = match_key in ambiguous_norms

                    update_indices = key_to_indices[batch_keys[pos]]
                    update_positions = np.fromiter(
                        (df_index_to_pos[i] for i in update_indices),
                        dtype=np.int64,
                        count=len(update_indices),
                    )

                    mapped_val = norm_to_mapped.get(match_key)
                    raw_val = norm_to_raw.get(match_key)
                    firm_val = norm_to_firm_id.get(match_key)

                    write_mapped[update_positions] = mapped_val
                    write_raw[update_positions] = raw_val
                    write_firm[update_positions] = firm_val
                    write_score[update_positions] = final_score / 100.0

                    if is_ambiguous:
                        write_method[update_positions] = 'Fuzzy Match (Ambiguous)'
                        write_conf[update_positions] = 'REVIEW_HIGH'

                        candidates_raw = norm_to_mapped_candidates.get(match_key, [])
                        if candidates_raw:
                            valid_candidates = [
                                str(c) for c in candidates_raw
                                if c is not None and not (isinstance(c, float) and pd.isna(c))
                            ]
                            write_ambig[update_positions] = "; ".join(valid_candidates)
                        else:
                            write_ambig[update_positions] = ""
                        review_count += len(update_indices)
                    elif final_score >= auto_threshold:
                        write_method[update_positions] = 'Fuzzy Match'
                        write_conf[update_positions] = 'AUTO'
                        auto_count += len(update_indices)
                    else:
                        write_method[update_positions] = 'Fuzzy Match'
                        write_conf[update_positions] = 'REVIEW_HIGH'
                        review_count += len(update_indices)

    # --- Single vectorised flush of all writes to the dataframe ---
    # We convert numpy slices -> plain Python lists so pandas does element-wise
    # dtype coercion (mirroring the original per-row scalar assignment).
    # Without this, a float64 column (e.g. Firm_ID with NaN) refuses an np.int64
    # array with a LossySetitemError on newer pandas.
    written = write_method != None  # noqa: E711 - object-dtype None compare
    if written.any():
        idx_to_write = df.index[written]
        df.loc[idx_to_write, 'SECP_NAME_MAPPED'] = list(write_mapped[written])
        df.loc[idx_to_write, 'matched_master_raw'] = list(write_raw[written])
        df.loc[idx_to_write, 'Firm_ID'] = list(write_firm[written])
        df.loc[idx_to_write, 'match_score'] = list(write_score[written])
        df.loc[idx_to_write, 'match_method'] = list(write_method[written])
        df.loc[idx_to_write, 'confidence'] = list(write_conf[written])
        ambig_mask = write_ambig != None  # noqa: E711
        if ambig_mask.any():
            df.loc[df.index[ambig_mask], 'ambiguous_master_candidates'] = list(write_ambig[ambig_mask])

    context.log.info(
        f"Stage 3 - Fuzzy Match: {auto_count:,} auto + {review_count:,} review "
        f"(rejected {rejected_count:,} false positives)"
    )

    return df


@asset(
    description="Stage 4: Advanced multi-signal embedding match",
    compute_kind="ml",
    group_name="matching",
    tags={"domain": "ucc-mapping", "stage": "matching"},
)
def advanced_embedding_results(
    context: AssetExecutionContext,
    config: PipelineConfig,
    fuzzy_match_results: pd.DataFrame,
    normalized_master_data: dict,
    master_embeddings_cache: dict,
) -> pd.DataFrame:
    """Stage 4: Advanced multi-signal matching (optimized).

    Optimisations applied:
    - Reuses precomputed master features from cache (soundex, TF-IDF vecs, entity types).
    - Precomputes per-query TF-IDF vec + soundex once.
    - Uses `compute_match_score_fast` to avoid rebuilding vectors per candidate.
    - Defers all per-row `df.loc` writes via numpy buffers; single vectorised flush at end.
    - Bumps gc cadence from every 500 batches to every 5,000 queries.
    """
    import os
    from concurrent.futures import ThreadPoolExecutor

    # Optimize CPU usage
    if config.cpu_threads > 0:
        os.environ['OMP_NUM_THREADS'] = str(config.cpu_threads)
        os.environ['OPENBLAS_NUM_THREADS'] = str(config.cpu_threads)
        os.environ['MKL_NUM_THREADS'] = str(config.cpu_threads)

    df = fuzzy_match_results.copy()

    unmatched_mask = _pending_match_mask(df)
    if not unmatched_mask.any():
        context.log.info("Stage 4 - Advanced Match: 0 matches (none to process)")
        return df

    master_normalized_list = normalized_master_data['normalized_list']
    master_mapped_list = normalized_master_data['mapped_list']
    master_raw_list = normalized_master_data['raw_list']
    master_firm_id_list = normalized_master_data['firm_id_list']
    master_entity_types = normalized_master_data['entity_types']
    master_state_pair_list = normalized_master_data['state_pair_list']
    state_pair_to_indices = normalized_master_data['state_pair_to_indices']

    if not master_normalized_list:
        context.log.info("Stage 4 - Advanced Match: 0 matches (no master data)")
        return df

    # Pull pre-computed artefacts from cache (single source of truth)
    master_embeddings = master_embeddings_cache.get('embeddings')
    ngram_index = master_embeddings_cache.get('ngram_index')
    master_soundex = master_embeddings_cache.get('soundex')
    model = master_embeddings_cache.get('model')
    tfidf_idf_weights = master_embeddings_cache.get('tfidf_idf_weights')
    master_tfidf_vecs = master_embeddings_cache.get('master_tfidf_vecs')
    master_tfidf_norms = master_embeddings_cache.get('master_tfidf_norms')
    per_state_faiss = master_embeddings_cache.get('per_state_faiss') or {}

    # Fallback: model reload
    if model is None:
        from ..embeddings import load_embedding_model
        context.log.warning("Model not found in cache, reloading...")
        model = load_embedding_model(config, context)

    if master_embeddings is None:
        context.log.error("Critical: Master embeddings cache is invalid!")
        return df

    if ngram_index is None:
        ngram_index = NGramIndex(n=3)
        for i, name in enumerate(master_normalized_list):
            ngram_index.add(i, name)
        ngram_index.freeze()

    if master_soundex is None:
        master_soundex = [soundex(name) for name in master_normalized_list]

    pending = df.loc[unmatched_mask, STATE_COLUMNS + ['SECP_NAME_NORMALIZED']].copy()
    pending['_match_key'] = _state_name_keys(pending, 'SECP_NAME_NORMALIZED', normalized=False)
    key_to_indices = pending.groupby('_match_key').apply(lambda g: g.index.tolist()).to_dict()
    unique_pending = pending.drop_duplicates('_match_key', keep='first')

    queries = unique_pending['SECP_NAME_NORMALIZED'].tolist()
    indices = unique_pending.index.tolist()
    query_keys = unique_pending['_match_key'].tolist()
    query_state_pairs = _state_pair_keys(unique_pending).tolist()

    deduped_count = int(unmatched_mask.sum()) - len(queries)
    if deduped_count:
        context.log.info(
            f"Stage 4 - Advanced Match: deduped {deduped_count:,} repeated normalized state/name records"
        )

    context.log.info(f"Matching {len(queries):,} records with CPU-optimized processing...")

    # Pre-compute all query metadata in parallel (soundex, type, ngram cands, tfidf vec)
    context.log.info("Pre-computing query metadata in parallel...")

    def compute_query_metadata(query):
        q_vec, q_norm = (build_tfidf_vec(query, tfidf_idf_weights)
                         if tfidf_idf_weights else ({}, 0.0))
        return {
            'soundex': soundex(query),
            'entity_type': detect_entity_type(query),
            'ngram_candidates': set(ngram_index.query(query, min_overlap=2)),
            'tfidf_vec': q_vec,
            'tfidf_norm': q_norm,
        }

    with ThreadPoolExecutor(max_workers=config.embedding_workers) as executor:
        query_metadata = list(executor.map(compute_query_metadata, queries))

    auto_count = high_count = medium_count = 0
    batch_size = config.embed_batch_size
    top_k = max(1, min(config.embed_top_k, len(master_normalized_list)))

    # Build df.index positional lookup once for fast numpy-buffer flushing.
    df_index = df.index
    df_pos_lookup = {orig_idx: pos for pos, orig_idx in enumerate(df_index)}
    n_rows = len(df)

    # Deferred write buffers (one entry per row). Use object dtype where strings are possible.
    write_mapped = np.empty(n_rows, dtype=object)
    write_raw = np.empty(n_rows, dtype=object)
    write_firm = np.empty(n_rows, dtype=object)
    write_score = np.full(n_rows, np.nan, dtype=np.float64)
    write_method = np.empty(n_rows, dtype=object)
    write_conf = np.empty(n_rows, dtype=object)
    has_write = np.zeros(n_rows, dtype=bool)

    # Sidecar table for the optional cross-encoder rerank pass.
    # Populated only for queries that landed in REVIEW_MEDIUM; one entry per
    # *query* (not per row) so multiple rows sharing a normalized name are
    # reranked once.
    rerank_ce_top_n = getattr(config, 'advanced_cross_encoder_top_n', 10)
    review_medium_queue: list = []  # list of dicts: query_name, candidates, best_match_idx, update_indices

    # -------------------------------------------------------------------------
    # Persistent query-embedding cache (across-run warm-restart optimisation).
    # Encoding queries on CPU is the most expensive single step in Stage 4.
    # If we've already encoded a query under the same model id we reload its
    # vector from disk instead of re-running the transformer.
    # -------------------------------------------------------------------------
    embed_cache_obj = None
    if getattr(config, 'query_embed_cache_enabled', False):
        try:
            from .embedding_cache import QueryEmbeddingCache
            embed_dim = int(master_embeddings.shape[1])
            embed_cache_obj = QueryEmbeddingCache(
                cache_dir=config.query_embed_cache_path,
                model_id=str(getattr(model, 'model_id', config.embedding_model)),
                dim=embed_dim,
            )
            context.log.info(
                f"Query embedding cache: {len(embed_cache_obj):,} entries loaded "
                f"from {config.query_embed_cache_path}"
            )
        except Exception as e:
            context.log.warning(f"Query embedding cache disabled (load failed): {e!r}")
            embed_cache_obj = None

    # Process in batches for CPU efficiency
    for start in range(0, len(queries), batch_size):
        end = min(start + batch_size, len(queries))
        batch_queries = queries[start:end]
        batch_indices = indices[start:end]
        batch_keys = query_keys[start:end]
        batch_metadata = query_metadata[start:end]
        batch_state_pairs = query_state_pairs[start:end]

        # ----- Encode batch (cache-aware) -----
        # Try to satisfy from the persistent cache first; encode only the misses.
        # Result is the same float32 (n, dim) array as before; order matches
        # batch_queries.
        if embed_cache_obj is not None:
            n_batch = len(batch_queries)
            embed_dim = int(master_embeddings.shape[1])
            query_embeddings = np.empty((n_batch, embed_dim), dtype=np.float32)
            miss_positions: list = []
            miss_queries: list = []
            for pos, q in enumerate(batch_queries):
                cached = embed_cache_obj.get(q)
                if cached is not None:
                    query_embeddings[pos] = cached
                else:
                    miss_positions.append(pos)
                    miss_queries.append(q)

            if miss_queries:
                # Dedup misses so we don't re-encode duplicates within a batch.
                unique_miss_queries: list = []
                seen: set = set()
                for q in miss_queries:
                    if q not in seen:
                        seen.add(q)
                        unique_miss_queries.append(q)
                miss_embeds = model.encode(
                    unique_miss_queries,
                    batch_size=128,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    device='cpu',
                )
                miss_embeds = np.ascontiguousarray(miss_embeds, dtype="float32")
                q_to_emb = {q: miss_embeds[i] for i, q in enumerate(unique_miss_queries)}
                for pos, q in zip(miss_positions, miss_queries):
                    query_embeddings[pos] = q_to_emb[q]
                try:
                    embed_cache_obj.add_many(unique_miss_queries, miss_embeds)
                except Exception as e:
                    context.log.warning(f"Embedding cache write failed: {e!r}")
        else:
            query_embeddings = model.encode(
                batch_queries,
                batch_size=128,
                normalize_embeddings=True,
                show_progress_bar=False,
                device='cpu'
            )
            query_embeddings = np.ascontiguousarray(query_embeddings, dtype="float32")

        positions_by_state_pair = {}
        for pos, state_pair in enumerate(batch_state_pairs):
            positions_by_state_pair.setdefault(state_pair, []).append(pos)

        for state_pair, positions in positions_by_state_pair.items():
            candidate_indices = state_pair_to_indices.get(state_pair, [])
            if not candidate_indices:
                continue

            candidate_array = np.asarray(candidate_indices, dtype=np.int64)
            state_query_embeddings = query_embeddings[positions]
            group_top_k = min(top_k, len(candidate_array))

            # Candidate retrieval: per-state FAISS if available (skip matmul),
            # otherwise fall back to the original matmul path. Both paths produce
            # mathematically identical scores (exact inner product on L2-
            # normalised vectors).
            state_index_info = per_state_faiss.get(state_pair)
            if state_index_info is not None:
                sub_index = state_index_info['index']
                # FAISS returns (similarities desc, sub-index ids) -- the same
                # top_local + sims[top_local] we need below.
                _D, _I = sub_index.search(
                    np.ascontiguousarray(state_query_embeddings, dtype="float32"),
                    group_top_k,
                )
                faiss_top_local_per_row = _I    # (n_positions, group_top_k)
                faiss_top_sims_per_row = _D     # (n_positions, group_top_k)
                similarity_matrix = None
            else:
                candidate_embeddings = master_embeddings[candidate_array]
                similarity_matrix = np.matmul(state_query_embeddings, candidate_embeddings.T)
                faiss_top_local_per_row = None
                faiss_top_sims_per_row = None

            for row_pos, pos in enumerate(positions):
                query_name = batch_queries[pos]
                update_indices = key_to_indices[batch_keys[pos]]
                metadata = batch_metadata[pos]
                query_soundex = metadata['soundex']
                query_type = metadata['entity_type']
                ngram_candidates = metadata['ngram_candidates']
                q_tfidf_vec = metadata['tfidf_vec']
                q_tfidf_norm = metadata['tfidf_norm']

                if faiss_top_local_per_row is not None:
                    # FAISS path: top-K is already sorted desc by similarity.
                    # -1 sentinels appear when the sub-index has fewer than
                    # group_top_k items; filter them out.
                    raw_top = faiss_top_local_per_row[row_pos]
                    raw_sims = faiss_top_sims_per_row[row_pos]
                    valid = raw_top >= 0
                    top_local = raw_top[valid].astype(np.int64, copy=False)
                    top_sims = raw_sims[valid].astype(np.float64, copy=False)
                else:
                    sims = similarity_matrix[row_pos]
                    if len(candidate_array) > group_top_k:
                        top_local = np.argpartition(sims, -group_top_k)[-group_top_k:]
                        top_local = top_local[np.argsort(sims[top_local])[::-1]]
                    else:
                        top_local = np.argsort(sims)[::-1]
                    top_sims = sims[top_local]

                top_embed_candidates = candidate_array[top_local].tolist()
                state_candidates = set(candidate_indices)
                all_candidates = set(top_embed_candidates) | (ngram_candidates & state_candidates)
                if len(all_candidates) > config.max_candidates_per_query:
                    remaining_slots = config.max_candidates_per_query - len(top_embed_candidates)
                    ngram_only = list((ngram_candidates & state_candidates) - set(top_embed_candidates))[:remaining_slots]
                    all_candidates = set(top_embed_candidates + ngram_only)

                # embed_sim_map is identical in both paths: cand_idx -> sim score.
                embed_sim_map = {
                    int(candidate_array[local_idx]): float(top_sims[j])
                    for j, local_idx in enumerate(top_local)
                }
                for cand_idx in all_candidates:
                    if cand_idx not in embed_sim_map:
                        embed_sim_map[cand_idx] = float(
                            np.dot(query_embeddings[pos], master_embeddings[cand_idx])
                        )

                sorted_candidates = sorted(
                    all_candidates,
                    key=lambda x: embed_sim_map[x],
                    reverse=True
                )

                best_score = 0.0
                best_match_idx = -1

                for cand_idx in sorted_candidates:
                    embed_sim = embed_sim_map[cand_idx]
                    if embed_sim < 0.70:
                        break
                    if best_score >= config.early_stop_threshold:
                        break

                    master_name = master_normalized_list[cand_idx]
                    master_type = master_entity_types[cand_idx]
                    if master_state_pair_list[cand_idx] != state_pair:
                        continue

                    if not entity_types_compatible(query_type, master_type):
                        continue

                    if embed_sim < 0.85 and not min_token_overlap(query_name, master_name, min_overlap=1):
                        continue

                    # Use precomputed master tfidf vec/norm when available (skip rebuilding).
                    if (master_tfidf_vecs is not None
                            and master_tfidf_norms is not None
                            and tfidf_idf_weights is not None):
                        total_score = compute_match_score_fast(
                            query_name, master_name, embed_sim,
                            soundex1=query_soundex,
                            soundex2=master_soundex[cand_idx],
                            query_tfidf_vec=q_tfidf_vec,
                            query_tfidf_norm=q_tfidf_norm,
                            cand_tfidf_vec=master_tfidf_vecs[cand_idx],
                            cand_tfidf_norm=master_tfidf_norms[cand_idx],
                            phonetic_weight=0.08,
                            ngram_weight=0.12,
                            fuzzy_weight=0.20,
                            embed_weight=0.50,
                            acronym_weight=0.10,
                        )
                    else:
                        total_score, _ = compute_match_score(
                            query_name, master_name, embed_sim,
                            phonetic_weight=0.08,
                            ngram_weight=0.12,
                            fuzzy_weight=0.20,
                            embed_weight=0.50,
                            acronym_weight=0.10,
                            idf_weights=tfidf_idf_weights,
                        )

                    if query_soundex == master_soundex[cand_idx]:
                        total_score += 0.05
                    if query_type == master_type and query_type != 'OTHER':
                        total_score += 0.03

                    if total_score > best_score:
                        best_score = total_score
                        best_match_idx = cand_idx

                # Stage write into numpy buffers (no df.loc here)
                if best_match_idx >= 0 and best_score >= config.embed_medium_threshold:
                    mapped_val = master_mapped_list[best_match_idx]
                    raw_val = master_raw_list[best_match_idx]
                    firm_val = master_firm_id_list[best_match_idx]
                    if best_score >= config.embed_auto_threshold:
                        conf_val = 'AUTO'
                        auto_count += len(update_indices)
                    elif best_score >= config.embed_high_threshold:
                        conf_val = 'REVIEW_HIGH'
                        high_count += len(update_indices)
                    else:
                        conf_val = 'REVIEW_MEDIUM'
                        medium_count += len(update_indices)
                        # Capture top candidates for the cross-encoder pass.
                        # Take the top-N by embed_sim so the reranker sees the
                        # most plausible alternatives, not just the chosen one.
                        top_for_rerank = sorted(
                            all_candidates,
                            key=lambda x: embed_sim_map.get(x, 0.0),
                            reverse=True,
                        )[:rerank_ce_top_n]
                        review_medium_queue.append({
                            'query_name': query_name,
                            'candidates': top_for_rerank,
                            'chosen_idx': best_match_idx,
                            'chosen_score': best_score,
                            'update_indices': list(update_indices),
                        })

                    for orig_idx in update_indices:
                        p = df_pos_lookup[orig_idx]
                        write_mapped[p] = mapped_val
                        write_raw[p] = raw_val
                        write_firm[p] = firm_val
                        write_score[p] = best_score
                        write_method[p] = 'Advanced Match'
                        write_conf[p] = conf_val
                        has_write[p] = True

        # Memory management — much less aggressive than the original
        if start > 0 and start % 5000 == 0:
            gc.collect()
            context.log.info(f"    Processed {start + len(batch_queries):,} / {len(queries):,}")

    # -------------------------------------------------------------------------
    # Cross-encoder rerank pass for REVIEW_MEDIUM rows.
    # Never demotes: only promotes to REVIEW_HIGH or AUTO when the cross-
    # encoder is strongly confident. If a different candidate beats the
    # original pick by a wide margin AND clears the promote-to-high threshold,
    # we swap to that candidate -- strictly an improvement since we're moving
    # from a low-confidence match to a high-confidence one.
    # -------------------------------------------------------------------------
    if (getattr(config, 'advanced_cross_encoder_enabled', False)
            and review_medium_queue):
        from ..rerank import (
            try_load_reranker, score_pairs, promote_confidence,
        )
        ce_model_name = getattr(
            config, 'advanced_cross_encoder_model',
            'cross-encoder/ms-marco-MiniLM-L-6-v2',
        )
        promote_auto = getattr(config, 'advanced_cross_encoder_promote_to_auto', 0.92)
        promote_high = getattr(config, 'advanced_cross_encoder_promote_to_high', 0.80)

        context.log.info(
            f"Cross-encoder rerank: loading {ce_model_name} for "
            f"{len(review_medium_queue):,} REVIEW_MEDIUM queries..."
        )
        ce_model = try_load_reranker(ce_model_name, config=config)
        if ce_model is None:
            context.log.warning(
                "Cross-encoder rerank skipped: model failed to load (likely "
                "not available locally and no internet). Downstream behaviour "
                "is unchanged."
            )
        else:
            # Build flat pairs list with index back-pointers.
            pairs: list = []
            pair_owners: list = []  # (queue_idx, cand_idx)
            for q_idx, entry in enumerate(review_medium_queue):
                qn = entry['query_name']
                for cand_idx in entry['candidates']:
                    pairs.append([qn, master_normalized_list[cand_idx]])
                    pair_owners.append((q_idx, cand_idx))

            ce_scores = score_pairs(ce_model, pairs)
            if len(ce_scores) != len(pairs):
                context.log.warning(
                    "Cross-encoder returned unexpected number of scores; "
                    "skipping rerank."
                )
            else:
                # Group scores by query.
                per_query_scores: dict = {}
                for (q_idx, cand_idx), score in zip(pair_owners, ce_scores):
                    per_query_scores.setdefault(q_idx, {})[cand_idx] = score

                promoted_auto = promoted_high = swapped = 0
                for q_idx, entry in enumerate(review_medium_queue):
                    cand_scores = per_query_scores.get(q_idx, {})
                    if not cand_scores:
                        continue
                    chosen_idx = entry['chosen_idx']
                    chosen_score = cand_scores.get(chosen_idx, 0.0)

                    ce_best_idx, ce_best_score = max(
                        cand_scores.items(), key=lambda kv: kv[1]
                    )

                    # Only swap if CE prefers a different candidate by a wide
                    # margin AND that candidate clears the promote-to-high bar.
                    use_idx = chosen_idx
                    use_ce_score = chosen_score
                    if (ce_best_idx != chosen_idx
                            and ce_best_score >= promote_high
                            and ce_best_score > chosen_score + 0.10):
                        use_idx = ce_best_idx
                        use_ce_score = ce_best_score
                        swapped += 1

                    new_band = promote_confidence(
                        'REVIEW_MEDIUM', use_ce_score, promote_auto, promote_high
                    )
                    if new_band is None and use_idx == chosen_idx:
                        continue  # leave row alone

                    mapped_val = master_mapped_list[use_idx]
                    raw_val = master_raw_list[use_idx]
                    firm_val = master_firm_id_list[use_idx]
                    new_score = entry['chosen_score'] if use_idx == chosen_idx else max(
                        entry['chosen_score'], use_ce_score
                    )
                    new_method = 'Advanced Match (Cross-Encoder Rerank)'
                    final_band = new_band or 'REVIEW_MEDIUM'
                    if final_band == 'AUTO':
                        promoted_auto += len(entry['update_indices'])
                    elif final_band == 'REVIEW_HIGH':
                        promoted_high += len(entry['update_indices'])

                    for orig_idx in entry['update_indices']:
                        p = df_pos_lookup[orig_idx]
                        write_mapped[p] = mapped_val
                        write_raw[p] = raw_val
                        write_firm[p] = firm_val
                        write_score[p] = new_score
                        write_method[p] = new_method
                        write_conf[p] = final_band

                # Reconcile reporting counters.
                medium_count -= (promoted_auto + promoted_high)
                auto_count += promoted_auto
                high_count += promoted_high
                context.log.info(
                    f"Cross-encoder rerank: promoted {promoted_auto:,} -> AUTO, "
                    f"{promoted_high:,} -> REVIEW_HIGH, swapped {swapped:,} picks"
                )

    # Persist the query embedding cache (best-effort).
    if embed_cache_obj is not None:
        try:
            embed_cache_obj.save()
            context.log.info(
                f"Query embedding cache: persisted ({len(embed_cache_obj):,} total entries)"
            )
        except Exception as e:
            context.log.warning(f"Embedding cache save failed: {e!r}")

    # Single vectorised flush
    # Convert numpy slices -> plain Python lists so pandas does element-wise
    # dtype coercion (mirrors original per-row scalar behaviour and avoids
    # LossySetitemError when Firm_ID is float64 with NaN).
    if has_write.any():
        mask = has_write
        idx_to_write = df_index[mask]
        df.loc[idx_to_write, 'SECP_NAME_MAPPED'] = list(write_mapped[mask])
        df.loc[idx_to_write, 'matched_master_raw'] = list(write_raw[mask])
        df.loc[idx_to_write, 'Firm_ID'] = list(write_firm[mask])
        df.loc[idx_to_write, 'match_score'] = list(write_score[mask])
        df.loc[idx_to_write, 'match_method'] = list(write_method[mask])
        df.loc[idx_to_write, 'confidence'] = list(write_conf[mask])

    context.log.info(f"Stage 4 - Advanced Match: {auto_count:,} auto, {high_count:,} high, {medium_count:,} medium")

    df.attrs['embedding_model'] = model
    gc.collect()

    return df


@asset(
    description="Stage 5: Cluster remaining unmatched records",
    compute_kind="ml",
    group_name="clustering",
    tags={"domain": "ucc-mapping", "stage": "clustering"},
)
def clustered_results(
    context: AssetExecutionContext,
    config: PipelineConfig,
    advanced_embedding_results: pd.DataFrame,
    master_embeddings_cache: dict,
) -> pd.DataFrame:
    """Stage 5: Cluster remaining unmatched records."""
    from rapidfuzz import fuzz
    import faiss
    
    df = advanced_embedding_results.copy()
    
    unmatched_mask = _pending_match_mask(df)
    
    if not unmatched_mask.any():
        context.log.info("Stage 5 - Clustering: 0 records to cluster")
        df['cluster_id'] = None
        df['cluster_label'] = None
        df['cluster_size'] = None
        return df
    
    sub_df = df.loc[unmatched_mask, ['SECP_NAME_NORMALIZED']].copy()
    names = sub_df['SECP_NAME_NORMALIZED'].tolist()
    original_indices = sub_df.index.tolist()
    n = len(names)
    
    context.log.info(f"Stage 5 - Clustering {n:,} unmatched records...")
    
    # NOTE: OMP/MKL thread env vars are set at module level (top of file)
    # Setting them here would have no effect since numpy/FAISS are already loaded.
    
    # Use model from cache (no reloading)
    model = master_embeddings_cache.get('model')
    if model is None:
        from ..embeddings import load_embedding_model
        model = load_embedding_model(config, context)
    else:
        context.log.info("Using model from cache (no reload)")
    
    context.log.info("Encoding unmatched names...")
    embed_dim = int(getattr(model, "get_sentence_embedding_dimension", lambda: 384)() or 384)
    embeddings = np.empty((n, embed_dim), dtype="float32")
    
    # Use larger batch size for better throughput
    encode_bs = 256
    for start in range(0, n, encode_bs):
        end = min(n, start + encode_bs)
        batch = names[start:end]
        batch_embeddings = model.encode(
            batch,
            batch_size=encode_bs,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        embeddings[start:end] = np.asarray(batch_embeddings, dtype="float32")
        
        # Progress logging for large datasets
        if start > 0 and start % 5000 == 0:
            context.log.info(f"  Encoded {start:,} / {n:,} names")
    
    context.log.info("Building search index...")
    dim = embeddings.shape[1]
    if n >= config.cluster_hnsw_threshold:
        index = faiss.IndexHNSWFlat(dim, config.cluster_hnsw_m)
        index.hnsw.efSearch = config.cluster_hnsw_ef_search
        index.hnsw.efConstruction = config.cluster_hnsw_ef_construction
    else:
        index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    context.log.info("Finding clusters...")
    uf = UnionFind(n)
    k = min(config.cluster_k_neighbors, n)
    
    search_batch_size = min(2048, config.search_batch_size)
    
    for start in range(0, n, search_batch_size):
        end = min(n, start + search_batch_size)
        query_batch = embeddings[start:end]
        
        scores, indices_batch = index.search(query_batch, k=k)
        
        for qi in range(end - start):
            i = start + qi
            for j in range(1, k):
                neighbor_idx = int(indices_batch[qi, j])
                sim = float(scores[qi, j])
                
                if sim >= config.cluster_sim_threshold:
                    fuzzy_score = fuzz.token_sort_ratio(names[i], names[neighbor_idx])
                    if fuzzy_score >= config.cluster_fuzzy_guard:
                        uf.union(i, neighbor_idx)
        
        if start > 0 and start % 10000 == 0:
            gc.collect()
    
    context.log.info("Assigning cluster labels...")
    roots = np.array([uf.find(i) for i in range(n)], dtype=np.int32)
    unique_roots, cluster_ids = np.unique(roots, return_inverse=True)
    
    cluster_id_str = [f"C{cid:06d}" for cid in cluster_ids]
    cluster_sizes = pd.Series(cluster_id_str).value_counts().to_dict()
    cluster_size_arr = [cluster_sizes[cid] for cid in cluster_id_str]
    
    cluster_labels = {}
    cluster_to_indices = {}
    for i, cid in enumerate(cluster_id_str):
        cluster_to_indices.setdefault(cid, []).append(i)
    
    for cid, idx_list in cluster_to_indices.items():
        if len(idx_list) == 1:
            cluster_labels[cid] = names[idx_list[0]]
        elif len(idx_list) > 50:
            cluster_labels[cid] = max((names[i] for i in idx_list), key=len)
        else:
            cluster_embeddings = np.asarray(embeddings[idx_list], dtype=np.float64)
            centroid = cluster_embeddings.mean(axis=0, keepdims=True)
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            medoid_idx = int(np.argmin(distances))
            cluster_labels[cid] = names[idx_list[medoid_idx]]
    
    cluster_label_arr = [cluster_labels[cid] for cid in cluster_id_str]
    
    df['cluster_id'] = None
    df['cluster_label'] = None
    df['cluster_size'] = None
    
    df.loc[original_indices, 'cluster_id'] = cluster_id_str
    df.loc[original_indices, 'cluster_label'] = cluster_label_arr
    df.loc[original_indices, 'cluster_size'] = cluster_size_arr
    df.loc[original_indices, 'match_method'] = 'Unmatched (Clustered)'
    df.loc[original_indices, 'confidence'] = 'NEW_ENTITY'
    
    n_clusters = len(unique_roots)
    singleton_clusters = sum(1 for s in cluster_sizes.values() if s == 1)
    context.log.info(f"Stage 5 - Clustering: {n_clusters:,} clusters ({singleton_clusters:,} singletons)")
    
    df.attrs['embedding_model'] = model
    df.attrs['cluster_embeddings'] = embeddings
    df.attrs['cluster_names'] = names
    df.attrs['cluster_indices'] = original_indices
    
    del index
    gc.collect()
    
    return df


@asset(
    description="Stage 6: Hierarchical merging of clusters",
    compute_kind="ml",
    group_name="clustering",
    tags={"domain": "ucc-mapping", "stage": "clustering"},
)
def merged_cluster_results(
    context: AssetExecutionContext,
    config: PipelineConfig,
    clustered_results: pd.DataFrame,
    master_embeddings_cache: dict,
) -> pd.DataFrame:
    """Stage 6: Hierarchical merging of clusters."""
    import faiss
    
    df = clustered_results.copy()
    
    if 'cluster_id' not in df.columns:
        return df
    
    clustered_mask = df['cluster_id'].notna()
    if not clustered_mask.any():
        return df
    
    # Use model from cache (no reloading)
    model = master_embeddings_cache.get('model')
    if model is None:
        from ..embeddings import load_embedding_model
        model = load_embedding_model(config, context)
    else:
        context.log.info("Using model from cache (no reload)")
    
    n_iterations = config.cluster_merge_iterations
    merge_k = config.cluster_merge_k
    max_cluster_size = config.cluster_max_size
    
    for iteration in range(n_iterations):
        cluster_info = df.loc[clustered_mask, ['cluster_id', 'cluster_label']].drop_duplicates('cluster_id')
        cluster_sizes = df.loc[clustered_mask, 'cluster_id'].value_counts().to_dict()
        
        if len(cluster_info) <= 1:
            break
        
        n_clusters = len(cluster_info)
        
        if n_clusters > 100000:
            context.log.info(f"Iteration {iteration + 1}: Skipping ({n_clusters:,} clusters exceeds 100k limit)")
            break
        
        context.log.info(f"Iteration {iteration + 1}: Processing {n_clusters:,} clusters (max size cap: {max_cluster_size})...")
        
        labels = cluster_info['cluster_label'].tolist()
        cluster_ids = cluster_info['cluster_id'].tolist()
        
        # Track cluster sizes for merge decisions
        cluster_idx_to_size = {i: cluster_sizes.get(cid, 1) for i, cid in enumerate(cluster_ids)}
        
        label_embeddings = model.encode(
            labels,
            batch_size=256,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        label_embeddings = np.asarray(label_embeddings, dtype="float32")
        
        n = len(labels)
        uf = UnionFind(n)
        
        # Track merged cluster sizes to prevent mega-clusters
        merged_sizes = {i: cluster_idx_to_size[i] for i in range(n)}
        
        index = faiss.IndexFlatIP(label_embeddings.shape[1])
        index.add(label_embeddings)
        
        # Use strict threshold - NO relaxation on later iterations
        base_threshold = config.cluster_merge_threshold
        
        k = min(merge_k, n)
        scores, indices = index.search(label_embeddings, k=k)
        
        merges_made = 0
        merges_blocked_by_size = 0
        for i in range(n):
            for j_idx in range(1, k):
                j = int(indices[i, j_idx])
                sim = float(scores[i, j_idx])
                
                if should_merge_clusters(labels[i], labels[j], sim):
                    root_i = uf.find(i)
                    root_j = uf.find(j)
                    if root_i != root_j:
                        # Check if combined size would exceed max
                        combined_size = merged_sizes[root_i] + merged_sizes[root_j]
                        if combined_size <= max_cluster_size:
                            uf.union(i, j)
                            # Update size tracking
                            new_root = uf.find(i)
                            merged_sizes[new_root] = combined_size
                            merges_made += 1
                            # Log merge decisions for debugging (first 10 merges)
                            if merges_made <= 10:
                                from rapidfuzz import fuzz
                                fuzzy_score = fuzz.token_set_ratio(labels[i].lower(), labels[j].lower())
                                context.log.debug(
                                    f"  Merging: '{labels[i][:40]}...' <-> '{labels[j][:40]}...' "
                                    f"(embed={sim:.3f}, fuzzy={fuzzy_score})"
                                )
                        else:
                            merges_blocked_by_size += 1
        
        if merges_made == 0:
            if merges_blocked_by_size > 0:
                context.log.info(f"No merges made ({merges_blocked_by_size:,} blocked by size cap), stopping")
            else:
                context.log.info("No merges possible, stopping")
            del index, label_embeddings
            gc.collect()
            break
        
        if merges_blocked_by_size > 0:
            context.log.info(f"  {merges_blocked_by_size:,} potential merges blocked by size cap")
        
        roots = [uf.find(i) for i in range(n)]
        unique_roots, new_cluster_ids = np.unique(roots, return_inverse=True)
        
        old_to_new = {}
        new_label_candidates = {}
        
        for i, old_cid in enumerate(cluster_ids):
            new_cid = f"H{iteration}{new_cluster_ids[i]:06d}"
            old_to_new[old_cid] = new_cid
            new_label_candidates.setdefault(new_cid, []).append(labels[i])
        
        # Pick best label for merged cluster
        # Strategy: Prefer concise, complete names over very long ones
        new_labels = {}
        for new_cid, candidates in new_label_candidates.items():
            if len(candidates) == 1:
                new_labels[new_cid] = candidates[0]
            else:
                # Filter out very long names (likely truncated or less representative)
                reasonable_length = [c for c in candidates if len(c) <= 100]
                if reasonable_length:
                    candidates = reasonable_length
                
                # Prefer multi-word names (more descriptive)
                multi_word = [c for c in candidates if ' ' in c.strip()]
                if multi_word:
                    # Among multi-word, prefer moderate length (not too short, not too long)
                    # This avoids picking very long names like "UNITED STATES OF AMERICA ACTING THROUGH RURAL"
                    # when a shorter, more representative name exists
                    def score_label(label):
                        word_count = len(label.split())
                        length = len(label)
                        # Prefer 2-5 word names of moderate length
                        if 2 <= word_count <= 5 and 20 <= length <= 80:
                            return (0, -word_count, length)  # Best: moderate word count, moderate length
                        elif word_count >= 2:
                            return (1, -word_count, length)  # Good: multi-word
                        else:
                            return (2, length)  # Less preferred: single word
                    
                    new_labels[new_cid] = min(multi_word, key=score_label)
                else:
                    # Single-word candidates - pick shortest (most concise)
                    new_labels[new_cid] = min(candidates, key=len)
        
        df.loc[clustered_mask, 'cluster_id'] = df.loc[clustered_mask, 'cluster_id'].map(old_to_new)
        df.loc[clustered_mask, 'cluster_label'] = df.loc[clustered_mask, 'cluster_id'].map(new_labels)
        
        n_merged = len(unique_roots)
        context.log.info(f"{n_clusters:,} -> {n_merged:,} clusters (merged {n_clusters - n_merged:,})")
        
        del index, label_embeddings
        gc.collect()
        
        if n_merged < 100 or (n_clusters - n_merged) < 50:
            break
    
    new_sizes = df.loc[clustered_mask, 'cluster_id'].value_counts().to_dict()
    df.loc[clustered_mask, 'cluster_size'] = df.loc[clustered_mask, 'cluster_id'].map(new_sizes)
    
    final_clusters = df.loc[clustered_mask, 'cluster_id'].nunique()
    context.log.info(f"Stage 6 - Final: {final_clusters:,} clusters")
    
    return df


@asset(
    description="Stage 7: Suggest mappings for clusters",
    compute_kind="ml",
    group_name="clustering",
    tags={"domain": "ucc-mapping", "stage": "clustering"},
)
def suggested_mapping_results(
    context: AssetExecutionContext,
    config: PipelineConfig,
    merged_cluster_results: pd.DataFrame,
    normalized_master_data: dict,
    master_embeddings_cache: dict,
) -> pd.DataFrame:
    """Stage 7: Suggest mappings for clusters."""
    from rapidfuzz import fuzz
    import faiss

    df = merged_cluster_results.copy()

    if 'cluster_id' not in df.columns:
        return df

    # CRITICAL: Only suggest mappings for unmatched clusters (NEW_ENTITY)
    clustered_mask = (df['cluster_id'].notna()) & (df['confidence'] == 'NEW_ENTITY')
    if not clustered_mask.any():
        context.log.info("Stage 7 - No unmatched clusters to suggest mappings for")
        return df

    # Use pre-computed master data
    master_mapped = normalized_master_data['mapped_list']
    master_mapped_normalized = normalized_master_data['normalized_list']
    master_state_pairs = normalized_master_data['state_pair_list']

    if not master_mapped:
        context.log.info("Stage 7 - No master mappings to suggest from")
        return df

    cluster_info = df.loc[
        clustered_mask,
        ['cluster_id', 'cluster_label', 'DEBT_STATE', 'SECP_STATE']
    ].drop_duplicates('cluster_id')
    cluster_labels = cluster_info['cluster_label'].tolist()
    cluster_ids = cluster_info['cluster_id'].tolist()
    cluster_state_pairs = _state_pair_keys(cluster_info).tolist()

    context.log.info(f"Stage 7 - Suggesting mappings for {len(cluster_ids):,} clusters...")
    context.log.info(f"  Master mappings available: {len(master_mapped):,}")

    # Use pre-computed master embeddings and index
    master_embeddings = master_embeddings_cache.get('embeddings')
    index = master_embeddings_cache.get('faiss_index')
    model = master_embeddings_cache.get('model')

    # Fallback handling
    if model is None:
        from ..embeddings import load_embedding_model
        context.log.warning("Model not found in cache, reloading...")
        model = load_embedding_model(config, context)

    if master_embeddings is None or index is None:
        import faiss
        context.log.warning("Embeddings/index not found in cache, rebuilding...")
        master_embeddings = model.encode(
            master_mapped_normalized,
            batch_size=256,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        master_embeddings = np.ascontiguousarray(master_embeddings, dtype="float32")
        faiss.omp_set_num_threads(1)
        index = faiss.IndexFlatIP(master_embeddings.shape[1])
        index.add(master_embeddings)

    context.log.info("Using pre-computed master embeddings")

    # Pre-normalize all labels
    context.log.info("Normalizing cluster labels...")
    label_norms = [normalize_name(l) for l in cluster_labels]

    # Filter out empty normalized labels
    valid_indices = [i for i, norm in enumerate(label_norms) if norm.strip()]
    if not valid_indices:
        context.log.warning("No valid normalized cluster labels")
        df['suggested_mapping'] = None
        df['suggestion_score'] = None
        df['suggestion_confidence'] = None
        df['alternative_suggestions'] = None
        return df

    # Filter to valid labels only
    valid_cluster_ids = [cluster_ids[i] for i in valid_indices]
    valid_cluster_labels = [cluster_labels[i] for i in valid_indices]
    valid_label_norms = [label_norms[i] for i in valid_indices]
    valid_state_pairs = [cluster_state_pairs[i] for i in valid_indices]

    context.log.info(f"Processing {len(valid_cluster_ids):,} valid clusters")

    # Encode cluster labels
    context.log.info("Encoding cluster labels...")
    label_embeddings = model.encode(
        valid_label_norms,
        batch_size=512,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    label_embeddings = np.asarray(label_embeddings, dtype="float32")
    context.log.info(f"  Encoded {len(label_embeddings):,} cluster labels")

    k = min(max(3, config.embed_top_k), len(master_mapped))
    scores, indices = index.search(label_embeddings, k=k)
    context.log.info(f"  Searched for top-{k} matches for each cluster")

    SUGGEST_HIGH_THRESHOLD = 0.80
    SUGGEST_MEDIUM_THRESHOLD = 0.70
    SUGGEST_LOW_THRESHOLD = 0.60

    # Build results dictionaries
    cluster_suggestions = {}
    suggestion_scores = {}
    suggestion_confidence = {}
    suggestion_alternatives = {}
    no_suggest_scores = []

    for i, cid in enumerate(valid_cluster_ids):
        state_pair = valid_state_pairs[i]
        scored_candidates = []
        for rank in range(indices.shape[1]):
            candidate_idx = int(indices[i, rank])
            if candidate_idx >= len(master_mapped) or master_state_pairs[candidate_idx] != state_pair:
                continue

            label_norm = valid_label_norms[i]
            match_norm = master_mapped_normalized[candidate_idx]
            fuzzy_score = fuzz.token_set_ratio(label_norm.lower(), match_norm.lower()) / 100.0
            score = (float(scores[i, rank]) * 0.6) + (fuzzy_score * 0.4)
            if label_norm.lower() in match_norm.lower() or match_norm.lower() in label_norm.lower():
                score += 0.1
            scored_candidates.append((score, candidate_idx, float(scores[i, rank])))

        if not scored_candidates:
            continue

        scored_candidates.sort(key=lambda candidate: candidate[0], reverse=True)
        score, best_idx, _ = scored_candidates[0]

        # Determine confidence level
        if score >= SUGGEST_HIGH_THRESHOLD:
            confidence = 'SUGGEST_HIGH'
        elif score >= SUGGEST_MEDIUM_THRESHOLD:
            confidence = 'SUGGEST_MEDIUM'
        elif score >= SUGGEST_LOW_THRESHOLD:
            confidence = 'SUGGEST_LOW'
        else:
            confidence = None

        if confidence is not None:
            best_match = master_mapped[best_idx]

            cluster_suggestions[cid] = best_match
            suggestion_scores[cid] = score
            suggestion_confidence[cid] = confidence

            # Get alternatives with proper validation
            alts = []
            for _, alt_idx, alt_embed_score in scored_candidates[1:]:
                if alt_embed_score >= SUGGEST_LOW_THRESHOLD * 0.9 and alt_idx < len(master_mapped):
                    alt_suggestion = master_mapped[alt_idx]

                    # Validate the alternative suggestion
                    if alt_suggestion is not None:
                        if isinstance(alt_suggestion, float) and pd.isna(alt_suggestion):
                            continue  # Skip NaN values

                        # Convert to string and validate
                        alt_str = str(alt_suggestion).strip()
                        if alt_str and alt_str.lower() != 'nan':
                            alts.append(alt_str)

            # Join alternatives safely
            suggestion_alternatives[cid] = "; ".join(alts) if alts else None
        else:
            no_suggest_scores.append(score)

    # Initialize columns
    df['suggested_mapping'] = None
    df['suggestion_score'] = None
    df['suggestion_confidence'] = None
    df['alternative_suggestions'] = None

    # Vectorized assignment
    if cluster_suggestions:
        cluster_to_suggestion = pd.Series(cluster_suggestions)
        cluster_to_score = pd.Series(suggestion_scores)
        cluster_to_confidence = pd.Series(suggestion_confidence)
        cluster_to_alternatives = pd.Series(suggestion_alternatives)

        mask = df['cluster_id'].isin(cluster_to_suggestion.index)
        df.loc[mask, 'suggested_mapping'] = df.loc[mask, 'cluster_id'].map(cluster_to_suggestion)
        df.loc[mask, 'suggestion_score'] = df.loc[mask, 'cluster_id'].map(cluster_to_score)
        df.loc[mask, 'suggestion_confidence'] = df.loc[mask, 'cluster_id'].map(cluster_to_confidence)
        df.loc[mask, 'alternative_suggestions'] = df.loc[mask, 'cluster_id'].map(cluster_to_alternatives)

    # Summary statistics
    high_count = sum(1 for v in suggestion_confidence.values() if v == 'SUGGEST_HIGH')
    medium_count = sum(1 for v in suggestion_confidence.values() if v == 'SUGGEST_MEDIUM')
    low_count = sum(1 for v in suggestion_confidence.values() if v == 'SUGGEST_LOW')
    no_suggest = len(valid_cluster_ids) - len(cluster_suggestions)

    # Diagnostic logging
    if no_suggest_scores:
        no_suggest_scores = np.array(no_suggest_scores)
        context.log.info(
            f"  Score distribution for unmatched clusters: "
            f"min={no_suggest_scores.min():.3f}, max={no_suggest_scores.max():.3f}, "
            f"mean={no_suggest_scores.mean():.3f}, median={np.median(no_suggest_scores):.3f}"
        )

    context.log.info(
        f"Stage 7 - Suggestions: {high_count:,} high, {medium_count:,} medium, {low_count:,} low, {no_suggest:,} no match")

    # Cleanup
    del label_embeddings
    gc.collect()

    return df


@asset(
    description="Load tracked banks file mapping bank names to Firm_ID",
    compute_kind="pandas",
    group_name="ingestion",
    tags={"domain": "ucc-mapping", "stage": "ingestion"},
)
def our_universe_data(
    context: AssetExecutionContext,
    config: PipelineConfig,
) -> Output[pd.DataFrame]:
    """Load tracked banks and normalize them to the pipeline lookup schema."""
    if not Path(config.universe_file).exists():
        context.log.warning(
            f"Tracked banks file not found at {config.universe_file}; "
            "continuing with Firm_ID values from the master mapping file only"
        )
        df = pd.DataFrame(columns=["SECP_NAME", "Firm_ID"])
        return Output(
            value=df,
            metadata={
                "row_count": MetadataValue.int(0),
                "unique_firm_ids": MetadataValue.int(0),
                "preview": MetadataValue.md("No tracked banks file found."),
                "source_file": MetadataValue.text(config.universe_file),
            },
        )

    # "Firm ID" (Universe.csv) and "Firm_ID" (tracked_banks.csv) are the same column.
    df = pd.read_csv(
        config.universe_file,
        sep=_detect_delimiter(config.universe_file),
        encoding="latin-1",
        usecols=lambda c: c in ("Bank Name", "Firm ID", "Firm_ID"),
    ).drop_duplicates()
    df = df.rename(columns={"Bank Name": "SECP_NAME", "Firm ID": "Firm_ID"})
    df["Firm_ID"] = pd.to_numeric(df["Firm_ID"], errors="coerce").astype("Int64")

    df = df[df["SECP_NAME"].notna() & (df["SECP_NAME"].str.strip() != "")]
    df = df.reset_index(drop=True)

    context.log.info(
        f"Loaded {len(df):,} tracked bank rows | "
        f"{df['Firm_ID'].notna().sum():,} with Firm_ID"
    )

    return Output(
        value=df,
        metadata={
            "row_count": MetadataValue.int(len(df)),
            "unique_firm_ids": MetadataValue.int(int(df["Firm_ID"].nunique())),
            "preview": MetadataValue.md(_preview_markdown(df)),
            "source_file": MetadataValue.text(config.universe_file),
        },
    )


@asset(
    description="Enrich pipeline results with Firm_ID from tracked banks",
    compute_kind="pandas",
    group_name="reporting",
    tags={"domain": "ucc-mapping", "stage": "reporting"},
)
def universe_enriched_results(
    context: AssetExecutionContext,
    suggested_mapping_results: pd.DataFrame,
    our_universe_data: pd.DataFrame,
) -> pd.DataFrame:
    """Join tracked-bank Firm_ID values onto the matched results.

    Matching logic (applied in priority order):
      1. Exact match on SECP_NAME_MAPPED → tracked Bank Name → Firm_ID.
      2. Exact match on raw SECP_NAME → tracked Bank Name → Firm_ID, only where
         the mapped-name lookup did not produce a Firm_ID.
    """
    df = suggested_mapping_results.copy()
    universe = our_universe_data.copy()
    existing_firm_id = (
        df["Firm_ID"].copy()
        if "Firm_ID" in df.columns
        else pd.Series(pd.NA, index=df.index, dtype="object")
    )
    if universe.empty:
        df["Firm_ID"] = existing_firm_id
        context.log.info(
            f"Tracked-bank enrichment skipped: {int(df['Firm_ID'].notna().sum()):,} "
            "rows keep Firm_ID from master mapping"
        )
        return df

    # ------------------------------------------------------------------
    # Build case-insensitive lookup tables from the tracked banks file.
    # All keys are stored upper-cased so lookups are case-insensitive.
    # When multiple tracked-bank rows map to the same upper-cased key we keep
    # the one whose Firm_ID appears most frequently (majority vote), which
    # prevents a rare mis-keyed row from overriding the correct Firm_ID.
    # ------------------------------------------------------------------

    def _majority_firm(sub: pd.DataFrame) -> pd.Series:
        """Return the most common Firm_ID for a tracked bank name."""
        if sub["Firm_ID"].notna().any():
            firm = sub["Firm_ID"].dropna().mode().iloc[0]
        else:
            firm = pd.NA
        return pd.Series({"Firm_ID": firm})

    universe["_key"] = universe["SECP_NAME"].str.upper().str.strip()
    univ_by_secp = (
        universe[universe["_key"].notna()]
        .groupby("_key", as_index=False)
        .apply(_majority_firm, include_groups=False)
    )
    secp_to_firm_ci = univ_by_secp.set_index("_key")["Firm_ID"]

    # ------------------------------------------------------------------
    # Initialise output columns
    # ------------------------------------------------------------------
    df["Firm_ID"] = existing_firm_id

    # ------------------------------------------------------------------
    # Pass 1 – SECP_NAME_MAPPED is the authoritative resolved identity.
    # Always resolve Firm_ID from the mapped name first so that, e.g.,
    # "FIRST SECURITY BANK" mapped to "Glacier Bank" gets Firm_ID 5959
    # (Glacier Bancorp) rather than 1354 (First Security Bancorp).
    # ------------------------------------------------------------------
    if "SECP_NAME_MAPPED" in df.columns:
        mapped_upper = df["SECP_NAME_MAPPED"].str.upper().str.strip()

        mapped_firm_secp = mapped_upper.map(secp_to_firm_ci)

        has_mapped_firm = mapped_firm_secp.notna()

        # Wrap RHS in list() so pandas does element-wise dtype coercion;
        # otherwise a numpy int array triggers LossySetitemError against the
        # float64 column when NaNs are present (pandas >= 2.2 strict casting).
        df.loc[has_mapped_firm, "Firm_ID"] = list(mapped_firm_secp[has_mapped_firm])

        pass1_firm = int(has_mapped_firm.sum())
        context.log.info(
            f"Tracked-bank join - Pass 1 (SECP_NAME_MAPPED, case-insensitive): "
            f"{pass1_firm:,} Firm_IDs matched"
        )

    # ------------------------------------------------------------------
    # Pass 2 – fall back to raw SECP_NAME only where SECP_NAME_MAPPED
    # produced no result (i.e. unmatched / new-entity rows).
    # ------------------------------------------------------------------
    secp_upper = df["SECP_NAME"].str.upper().str.strip()

    raw_firm = secp_upper.map(secp_to_firm_ci)

    fill_raw_firm = df["Firm_ID"].isna() & raw_firm.notna()

    # Wrap RHS in list() so pandas does element-wise dtype coercion against
    # the float64 column (see note in Pass 1).
    df.loc[fill_raw_firm, "Firm_ID"] = list(raw_firm[fill_raw_firm])

    pass2_firm = int(fill_raw_firm.sum())
    context.log.info(
        f"Tracked-bank join - Pass 2 (raw SECP_NAME fallback, case-insensitive): "
        f"{pass2_firm:,} additional Firm_IDs"
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_with_firm = int(df["Firm_ID"].notna().sum())
    total = len(df)
    context.log.info(
        f"Tracked-bank enrichment complete: "
        f"{total_with_firm:,}/{total:,} rows have Firm_ID "
        f"({total_with_firm / total * 100:.1f}%)"
    )

    return df


# =============================================================================
# Updated master mapping + merged raw file
# =============================================================================
#
# Goal
# ----
# Produce a single "updated master" mapping file that extends the original
# Master Mapping CSV with the new high-confidence resolutions the pipeline
# generated this run. Then join that updated mapping back onto the raw UCC
# file to produce a row-per-filing merged artefact that downstream analyses
# (notably the future DUNS-aware lead-bank work) can consume.
#
# Why these are separate assets
# -----------------------------
# * ``updated_master_data`` is the human-verifiable artefact: "here is what
#   we knew before + what the pipeline learned this run". It contains the
#   same five columns as the original Master Mapping so it can be reviewed
#   in Excel and, if accepted, swapped in as the next run's master.
# * ``merged_raw_with_firm_id`` is the operational artefact: every raw UCC
#   row enriched with ``SECP_NAME_MAPPED`` / ``Firm_ID`` / ``match_method``.
#   Re-reads the raw file fresh (not via ``ucc_raw_data``) so that when the
#   user upgrades to the full schema (DUNS, F_STATEMENT_TYPE_CODE, etc.)
#   those columns flow through automatically without code changes.
#
# Eligible match methods (per 2026-06-16 design discussion)
# ----------------------------------------------------------
# Stage 1-3 results only. Stage 4 (Advanced Match) is deliberately excluded
# pending separate review. Both deterministic and "Ambiguous" variants are
# included; ambiguous rows are not 1:1 mappings, but the user wants them
# visible in the updated master for review purposes.
# =============================================================================

# Rows the pipeline may add to the updated master on its own: anything it accepted
# (confidence AUTO, whatever the method). REVIEW_* rows are suggestions and reach the
# master only through an approved review decision; NEW_ENTITY rows never do.
# (Until 2026-10 this was a list of match methods, which let unreviewed
# "Fuzzy Match (Ambiguous)" rows in and kept AUTO "Advanced Match" rows out.)
_UPDATED_MASTER_CONFIDENCE: tuple = ("AUTO",)

# Columns kept in the updated master file. Mirrors the original Master
# Mapping CSV schema so the file is drop-in usable as the next run's master.
_UPDATED_MASTER_COLUMNS: tuple = (
    "DEBT_STATE",
    "SECP_STATE",
    "SECP_NAME",
    "SECP_NAME_MAPPED",
    "Firm ID",
)


@asset(
    description=(
        "Updated master mapping: original Master Mapping rows + newly resolved "
        "(SECP_NAME -> SECP_NAME_MAPPED, Firm_ID) rows from Stage 1-3 of this "
        "run. Stage 4 (Advanced Match) is intentionally excluded."
    ),
    compute_kind="pandas",
    group_name="reporting",
    tags={"domain": "ucc-mapping", "stage": "reporting"},
)
def updated_master_data(
    context: AssetExecutionContext,
    config: PipelineConfig,
    master_mapping_data: pd.DataFrame,
    universe_enriched_results: pd.DataFrame,
) -> Output[pd.DataFrame]:
    """Append newly-resolved Stage 1-3 mappings to the original master file.

    Schema matches the original ``Master Mapping.csv`` so the output is
    drop-in usable as the next run's master file.

    Conflict resolution
    -------------------
    On a key collision (same ``(DEBT_STATE, SECP_STATE, SECP_NAME)`` triple
    present in both the original master AND the new pipeline output) the
    **original master row wins**. The original master is the human-verified
    source of truth; pipeline output is provisional. This means re-running
    the pipeline never overwrites a known-good mapping with a guess.

    Writes ``updated_master_{ts}.csv`` to ``config.output_dir``.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Normalise the original master to the canonical column set.
    # ------------------------------------------------------------------
    master = master_mapping_data.copy()
    master = _standardize_state_columns(master)
    if "Firm ID" not in master.columns and "Firm_ID" in master.columns:
        # Original master file uses 'Firm ID' (space). Defensive rename in
        # case an upstream change normalised it to underscore form.
        master = master.rename(columns={"Firm_ID": "Firm ID"})
    keep_master_cols = [c for c in _UPDATED_MASTER_COLUMNS if c in master.columns]
    master = master[keep_master_cols].copy()
    master["_source"] = "original_master"
    n_master = len(master)

    # ------------------------------------------------------------------
    # 2. Filter pipeline output to eligible match methods + valid mapping.
    # ------------------------------------------------------------------
    enriched = universe_enriched_results.copy()
    if "match_method" not in enriched.columns:
        context.log.warning(
            "updated_master_data: universe_enriched_results has no "
            "match_method column; emitting original master only."
        )
        eligible = enriched.iloc[0:0].copy()
    else:
        method_mask = (
            enriched["confidence"].astype("string").isin(_UPDATED_MASTER_CONFIDENCE)
            if "confidence" in enriched.columns
            else enriched["match_method"].astype("string").str.startswith("Exact Match", na=False)
        )
        # Must also have a non-null mapped name to be useful as a master row.
        if "SECP_NAME_MAPPED" in enriched.columns:
            mapped_mask = (
                enriched["SECP_NAME_MAPPED"].notna()
                & (enriched["SECP_NAME_MAPPED"].astype("string").str.strip() != "")
            )
        else:
            mapped_mask = pd.Series(False, index=enriched.index)
        eligible = enriched.loc[method_mask & mapped_mask].copy()

    # ------------------------------------------------------------------
    # 3. Project pipeline output into the master schema.
    # ------------------------------------------------------------------
    if len(eligible):
        if "Firm_ID" in eligible.columns and "Firm ID" not in eligible.columns:
            eligible["Firm ID"] = eligible["Firm_ID"]
        elif "Firm ID" not in eligible.columns:
            eligible["Firm ID"] = pd.NA
        new_rows_cols = [c for c in _UPDATED_MASTER_COLUMNS if c in eligible.columns]
        new_rows = eligible[new_rows_cols].copy()
        new_rows = _standardize_state_columns(new_rows)
        new_rows["_source"] = (
            "pipeline_" + eligible["match_method"].astype("string").fillna("unknown")
        ).values
    else:
        new_rows = pd.DataFrame(
            columns=list(_UPDATED_MASTER_COLUMNS) + ["_source"]
        )

    n_new_candidate = len(new_rows)

    # ------------------------------------------------------------------
    # 4. Concat master first, new rows second; dedupe on the three-column
    #    key keeping the FIRST occurrence (= original master wins).
    # ------------------------------------------------------------------
    combined = pd.concat([master, new_rows], ignore_index=True, sort=False)
    # The master is read as nullable Int64 but pipeline rows carry a float
    # Firm_ID (NaN-propagated), so the concat demotes the column to float and
    # the CSV would print "1257.0". Restore the master's integer formatting.
    if "Firm ID" in combined.columns:
        fid = pd.to_numeric(combined["Firm ID"], errors="coerce")
        if (fid.dropna() % 1 == 0).all():
            combined["Firm ID"] = fid.astype("Int64")
    key_cols = ["DEBT_STATE", "SECP_STATE", "SECP_NAME"]
    # Filter to rows that have a non-null name (key would otherwise be useless).
    has_name = combined["SECP_NAME"].notna() & (
        combined["SECP_NAME"].astype("string").str.strip() != ""
    )
    combined = combined.loc[has_name].copy()
    before_dedup = len(combined)
    combined = combined.drop_duplicates(subset=key_cols, keep="first").reset_index(drop=True)
    after_dedup = len(combined)

    n_added = max(0, after_dedup - n_master)
    n_conflicts_resolved = max(0, n_new_candidate - n_added)

    context.log.info(
        f"updated_master_data: master={n_master:,} rows; "
        f"new candidates={n_new_candidate:,}; added={n_added:,}; "
        f"conflicts (kept master)={n_conflicts_resolved:,}; "
        f"final={after_dedup:,}"
    )
    if before_dedup != after_dedup:
        context.log.info(
            f"updated_master_data: deduped {before_dedup - after_dedup:,} "
            "duplicate keys (original master wins on conflict)"
        )

    # Persistence: drop the internal _source column from the canonical CSV
    # (mirrors original master schema), but keep a sidecar with provenance.
    out_path = output_dir / f"updated_master_{timestamp}.csv"
    sidecar_path = output_dir / f"updated_master_{timestamp}_with_source.csv"
    canonical_cols = [c for c in _UPDATED_MASTER_COLUMNS if c in combined.columns]
    try:
        combined[canonical_cols].to_csv(out_path, index=False)
        combined.to_csv(sidecar_path, index=False)
        context.log.info(f"updated_master_data: wrote {out_path}")
        context.log.info(f"updated_master_data: wrote {sidecar_path} (with _source)")
    except Exception as e:
        context.log.warning(f"updated_master_data: failed to write CSV: {e!r}")

    return Output(
        value=combined,
        metadata={
            "original_master_rows": MetadataValue.int(n_master),
            "new_candidate_rows": MetadataValue.int(n_new_candidate),
            "rows_added": MetadataValue.int(n_added),
            "conflicts_kept_master": MetadataValue.int(n_conflicts_resolved),
            "final_row_count": MetadataValue.int(after_dedup),
            "output_path": MetadataValue.path(str(out_path)),
            "sidecar_path": MetadataValue.path(str(sidecar_path)),
            "timestamp": MetadataValue.text(timestamp),
            "preview": MetadataValue.md(_preview_markdown(combined[canonical_cols])),
        },
    )


@asset(
    description=(
        "Merge the raw UCC file with the updated master mapping on "
        "(DEBT_STATE, SECP_STATE, SECP_NAME) to bring SECP_NAME_MAPPED / "
        "Firm_ID onto every raw filing row. Re-reads the raw file fresh so "
        "that any extra columns present in the source CSV (e.g. DUNS, "
        "F_STATEMENT_TYPE_CODE) flow through automatically."
    ),
    compute_kind="pandas",
    group_name="reporting",
    tags={"domain": "ucc-mapping", "stage": "reporting"},
)
def merged_raw_with_firm_id(
    context: AssetExecutionContext,
    config: PipelineConfig,
    updated_master_data: pd.DataFrame,
) -> Output[pd.DataFrame]:
    """Row-per-filing merge of raw UCC against the updated master mapping.

    Re-reads the raw CSV directly (not via ``ucc_raw_data``) so:
      1. The merged file has **all** raw columns, not just the three
         match-key columns. When the user later swaps in the full
         production CSV (with DUNS / F_STATEMENT_TYPE_CODE / etc.), those
         columns appear in the output automatically.
      2. The merged file has **one row per filing**, not deduped. Lead-bank
         and downstream count-based analyses need raw filing counts.

    Merge keys: ``DEBT_STATE``, ``SECP_STATE``, ``SECP_NAME``.
    Adds: ``SECP_NAME_MAPPED``, ``Firm_ID``, ``Firm_ID_source``
          ("original_master" or "pipeline_<match_method>" when matched,
          else NaN).

    TODO (next PR, blocked on full production CSV):
      - Switch ``lead_bank_designation`` to consume this asset.
      - Add DUNS-based company key when ``config.duns_column`` resolves.
      - Add strict ``F_STATEMENT_TYPE_CODE`` (2=ORIGINAL, 4=CONTINUATION)
        classifier with fallback to the existing substring path.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load the full raw CSV (all columns, no dedupe).
    # ------------------------------------------------------------------
    try:
        raw_full = pd.read_csv(
            config.ucc_file,
            sep=_detect_delimiter(config.ucc_file),
            encoding="latin-1",
        )
    except Exception as e:
        context.log.error(
            f"merged_raw_with_firm_id: could not read raw UCC file "
            f"({config.ucc_file}): {e!r}"
        )
        empty = pd.DataFrame(
            columns=list(RAW_MATCH_COLUMNS)
            + ["SECP_NAME_MAPPED", "Firm_ID", "Firm_ID_source"]
        )
        return Output(
            value=empty,
            metadata={
                "row_count": MetadataValue.int(0),
                "notes": MetadataValue.text("Raw CSV unreadable; merge skipped."),
            },
        )

    raw_full = _standardize_state_columns(raw_full)
    n_raw = len(raw_full)
    context.log.info(f"merged_raw_with_firm_id: loaded {n_raw:,} raw filings")

    # ------------------------------------------------------------------
    # 2. Verify required join keys are present in the raw file.
    # ------------------------------------------------------------------
    missing_in_raw = [c for c in RAW_MATCH_COLUMNS if c not in raw_full.columns]
    if missing_in_raw:
        context.log.error(
            f"merged_raw_with_firm_id: raw CSV missing join key columns "
            f"{missing_in_raw}; emitting raw file unchanged."
        )
        out = raw_full.copy()
        for c in ("SECP_NAME_MAPPED", "Firm_ID", "Firm_ID_source"):
            if c not in out.columns:
                out[c] = pd.NA
        out_path = output_dir / f"merged_raw_with_firm_id_{timestamp}.csv"
        try:
            out.to_csv(out_path, index=False)
        except Exception as e:
            context.log.warning(f"merged_raw_with_firm_id: write failed: {e!r}")
        return Output(
            value=out,
            metadata={
                "row_count": MetadataValue.int(len(out)),
                "missing_join_keys": MetadataValue.json(missing_in_raw),
                "output_path": MetadataValue.path(str(out_path)),
            },
        )

    # ------------------------------------------------------------------
    # 3. Prep the right-hand side of the merge: updated_master, with
    #    'Firm ID' renamed to 'Firm_ID' for downstream consistency. Dedupe
    #    on the 3-key so a one-to-many merge doesn't explode row counts;
    #    keep the first row (== original master / earliest pipeline source).
    # ------------------------------------------------------------------
    master = updated_master_data.copy()
    master = _standardize_state_columns(master)
    if "Firm_ID" not in master.columns and "Firm ID" in master.columns:
        master = master.rename(columns={"Firm ID": "Firm_ID"})
    rhs_cols = [
        c for c in ("DEBT_STATE", "SECP_STATE", "SECP_NAME",
                    "SECP_NAME_MAPPED", "Firm_ID", "_source")
        if c in master.columns
    ]
    rhs = master[rhs_cols].drop_duplicates(
        subset=["DEBT_STATE", "SECP_STATE", "SECP_NAME"], keep="first"
    )
    if "_source" in rhs.columns:
        rhs = rhs.rename(columns={"_source": "Firm_ID_source"})

    # ------------------------------------------------------------------
    # 4. Left-merge: raw on the left so every filing survives, even those
    #    with no master mapping (Firm_ID will be NaN for those).
    # ------------------------------------------------------------------
    merged = raw_full.merge(rhs, on=list(RAW_MATCH_COLUMNS), how="left")

    n_with_firm = (
        int(merged["Firm_ID"].notna().sum()) if "Firm_ID" in merged.columns else 0
    )
    n_with_mapped = (
        int(merged["SECP_NAME_MAPPED"].notna().sum())
        if "SECP_NAME_MAPPED" in merged.columns else 0
    )
    pct_firm = (n_with_firm / n_raw * 100) if n_raw else 0.0
    pct_mapped = (n_with_mapped / n_raw * 100) if n_raw else 0.0

    context.log.info(
        f"merged_raw_with_firm_id: {n_with_mapped:,}/{n_raw:,} rows mapped "
        f"({pct_mapped:.1f}%); {n_with_firm:,}/{n_raw:,} rows have Firm_ID "
        f"({pct_firm:.1f}%)"
    )

    # ------------------------------------------------------------------
    # 5. Write the merged file.
    # ------------------------------------------------------------------
    out_path = output_dir / f"merged_raw_with_firm_id_{timestamp}.csv"
    try:
        merged.to_csv(out_path, index=False)
        context.log.info(f"merged_raw_with_firm_id: wrote {out_path}")
    except Exception as e:
        context.log.warning(f"merged_raw_with_firm_id: write failed: {e!r}")

    return Output(
        value=merged,
        metadata={
            "row_count": MetadataValue.int(len(merged)),
            "rows_with_mapped_name": MetadataValue.int(n_with_mapped),
            "rows_with_firm_id": MetadataValue.int(n_with_firm),
            "pct_with_firm_id": MetadataValue.float(round(pct_firm, 2)),
            "output_path": MetadataValue.path(str(out_path)),
            "timestamp": MetadataValue.text(timestamp),
            "columns": MetadataValue.json(list(merged.columns)),
        },
    )


# =============================================================================
# Stage 8: Lead-bank designation
# =============================================================================

# Candidate column names searched, in priority order, when the user has not
# pinned an explicit override on PipelineConfig. Matched case-insensitively.
_FILING_TYPE_COLUMN_CANDIDATES: tuple = (
    "FILING_TYPE", "FILING_TYP", "FILE_TYPE", "FILETYPE",
    "RECORD_TYPE", "REC_TYPE", "TYPE", "FILING",
    "FILING_NATURE", "FILE_NATURE", "NATURE_OF_FILING",
)
_DEBTOR_COLUMN_CANDIDATES: tuple = (
    "DEBT_NAME", "DEBTOR_NAME", "DEBT_FIRM", "DEBTOR_FIRM",
    "DEBTOR", "DEBTOR_COMPANY", "COMPANY_NAME", "COMPANY",
    "DEBT_COMPANY", "DEBTORS_NAME",
)

# Substring signals (matched on uppercased, stripped value) used to classify a
# filing row as ORIGINAL or CONTINUATION. Order matters: continuation is
# checked first so an "INITIAL CONTINUATION" (rare) classifies as continuation.
_CONTINUATION_SIGNALS: tuple = ("CONTINU", "CONT.", " CONT", "CT ", "-CT", "_CT")
_ORIGINAL_SIGNALS: tuple = (
    "ORIGINAL", "INITIAL", "NEW", "FINANC",  # FINANCING / FINANCING STATEMENT
    "UCC1", "UCC-1", "UCC 1",
)


def _classify_filing_type(value) -> str:
    """Map a raw filing-type cell value to 'ORIGINAL', 'CONTINUATION', or 'OTHER'.

    Falsy / NaN values map to 'OTHER'. Matching is case-insensitive and
    substring-based to tolerate variants like 'Continuation Statement',
    'CT', 'UCC1 - Original', etc. Continuation is checked first so a value
    that contains both signals (rare) is classified as a continuation.
    """
    if value is None:
        return "OTHER"
    try:
        if pd.isna(value):
            return "OTHER"
    except (TypeError, ValueError):
        pass
    s = str(value).strip().upper()
    if not s:
        return "OTHER"
    for sig in _CONTINUATION_SIGNALS:
        if sig in s:
            return "CONTINUATION"
    for sig in _ORIGINAL_SIGNALS:
        if sig in s:
            return "ORIGINAL"
    return "OTHER"


def _resolve_column(
    available: List[str],
    override: str,
    candidates: tuple,
) -> str:
    """Resolve a column name from an explicit override or a candidate list.

    Matching is case-insensitive. Returns the actual column name as it appears
    in `available`, or an empty string if no candidate is present.
    """
    upper_to_actual = {c.upper(): c for c in available}
    if override:
        actual = upper_to_actual.get(override.strip().upper())
        if actual:
            return actual
    for cand in candidates:
        actual = upper_to_actual.get(cand.upper())
        if actual:
            return actual
    return ""


@asset(
    description=(
        "Stage 8: Count originals + continuations per (company, bank) and "
        "designate a lead bank when one bank has >= the configured share "
        "(default 66%) of filings. Single-bank companies are designated lead "
        "only when they have multiple filings."
    ),
    compute_kind="pandas",
    group_name="reporting",
    tags={"domain": "ucc-mapping", "stage": "reporting"},
)
def lead_bank_designation(
    context: AssetExecutionContext,
    config: PipelineConfig,
    universe_enriched_results: pd.DataFrame,
) -> Output[pd.DataFrame]:
    """Per-company / per-bank filing counts plus lead-bank designation.

    SCOPE: In-universe banks only.
    --------------------------------
    Only filings where the secured party was resolved to a Firm_ID (i.e. the
    bank is in ``Universe.csv``) are counted. Non-universe banks are dropped
    entirely — both from the share denominator and from the output rows —
    because the lead-bank designation is about how *tracked* banks rank
    against each other for a debtor, not about absolute market share.

    Logic
    -----
    For each debtor / company (grouped by debtor name + DEBT_STATE), among
    the IN-UNIVERSE banks:

      1. Count ORIGINAL filings and CONTINUATION filings per bank
         (bank = ``SECP_NAME_MAPPED``, falling back to ``SECP_NAME`` when
         unmapped).
      2. Compute total filings (originals + continuations + other) per bank.
      3. Compute share = bank_total / company_total_filings, where the
         company total is the sum of all IN-UNIVERSE filings for that debtor.
      4. Designate ``is_lead_bank = True`` when:
           * the company has exactly one in-universe bank AND that bank has
             >= 2 filings (sole bank with sustained relationship), OR
           * the company has multiple in-universe banks AND this bank has
             the strictly highest total AND ``share >= threshold``
             (default 0.66).
         A company with a single in-universe bank and a single filing is
         NOT designated (insufficient evidence, per design discussion).
         A company with NO in-universe banks does not appear in the output.

    Outputs two CSVs to ``config.output_dir``:
      * ``bank_filing_counts_{ts}.csv`` -- one row per (company, bank).
      * ``lead_bank_by_company_{ts}.csv`` -- one row per company with the
        designated lead bank, if any.

    The returned DataFrame is the per-(company, bank) table.
    """
    enriched = universe_enriched_results.copy()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Re-read the full raw CSV (all columns) so we can access filing
    #    type + debtor name fields that the matching pipeline drops.
    # ------------------------------------------------------------------
    try:
        raw_full = pd.read_csv(
            config.ucc_file,
            sep=_detect_delimiter(config.ucc_file),
            encoding="latin-1",
        )
    except Exception as e:
        context.log.error(f"lead_bank_designation: could not read raw UCC file: {e}")
        empty = pd.DataFrame(
            columns=[
                "Firm_ID", "company_name", "company_state",
                "bank_name", "bank_firm_id",
                "originals", "continuations", "other_filings", "total_filings",
                "company_total_filings", "company_bank_count",
                "share_of_company", "is_lead_bank", "lead_bank_reason",
            ]
        )
        return Output(
            value=empty,
            metadata={
                "row_count": MetadataValue.int(0),
                "notes": MetadataValue.text("Raw CSV unreadable; lead-bank step skipped."),
            },
        )

    raw_full = _standardize_state_columns(raw_full)

    # ------------------------------------------------------------------
    # 2. Resolve filing-type and debtor-name columns (with config overrides).
    # ------------------------------------------------------------------
    filing_type_col = _resolve_column(
        list(raw_full.columns),
        getattr(config, "filing_type_column", ""),
        _FILING_TYPE_COLUMN_CANDIDATES,
    )
    debtor_col = _resolve_column(
        list(raw_full.columns),
        getattr(config, "debtor_company_column", ""),
        _DEBTOR_COLUMN_CANDIDATES,
    )

    if not filing_type_col:
        context.log.warning(
            "lead_bank_designation: no filing-type column found in raw CSV. "
            f"Looked for: {list(_FILING_TYPE_COLUMN_CANDIDATES)}. "
            "All filings will be classified as 'OTHER' and counted toward total only. "
            "Set PipelineConfig.filing_type_column to override."
        )
    else:
        context.log.info(f"lead_bank_designation: filing-type column = '{filing_type_col}'")

    if debtor_col:
        context.log.info(f"lead_bank_designation: debtor-name column = '{debtor_col}'")
    else:
        context.log.info(
            "lead_bank_designation: no debtor-name column in raw CSV; "
            "will group companies by DEBT_STATE only (no company-level granularity)."
        )

    # ------------------------------------------------------------------
    # 3. Bring the resolved mapping (SECP_NAME_MAPPED, Firm_ID) from the
    #    enriched pipeline output onto the full raw filings via the same
    #    state/name key used elsewhere in the pipeline.
    # ------------------------------------------------------------------
    keep_from_enriched = [
        c for c in [
            "DEBT_STATE", "SECP_STATE", "SECP_NAME",
            "SECP_NAME_MAPPED", "Firm_ID", "match_method", "confidence",
        ] if c in enriched.columns
    ]
    join_keys_present = [c for c in RAW_MATCH_COLUMNS if c in enriched.columns and c in raw_full.columns]
    if len(join_keys_present) == len(RAW_MATCH_COLUMNS):
        map_df = enriched[keep_from_enriched].drop_duplicates(subset=RAW_MATCH_COLUMNS, keep="first")
        joined = raw_full.merge(map_df, on=RAW_MATCH_COLUMNS, how="left")
    else:
        context.log.warning(
            "lead_bank_designation: raw CSV / enriched results are missing one "
            f"of the join key columns {RAW_MATCH_COLUMNS}; bank mapping may be incomplete."
        )
        joined = raw_full.copy()
        for c in keep_from_enriched:
            if c not in joined.columns:
                joined[c] = pd.NA

    # ------------------------------------------------------------------
    # 4. Build the per-row classification + bank label + company key.
    # ------------------------------------------------------------------
    if filing_type_col:
        joined["_filing_class"] = joined[filing_type_col].map(_classify_filing_type)
    else:
        joined["_filing_class"] = "OTHER"

    # Bank label: prefer the resolved mapped name (cleaner / canonical),
    # fall back to the raw secured-party name when unmapped.
    if "SECP_NAME_MAPPED" in joined.columns:
        bank_series = joined["SECP_NAME_MAPPED"].astype("string")
    else:
        bank_series = pd.Series([pd.NA] * len(joined), index=joined.index, dtype="string")
    fallback_bank = (
        joined["SECP_NAME"].astype("string")
        if "SECP_NAME" in joined.columns
        else pd.Series([pd.NA] * len(joined), index=joined.index, dtype="string")
    )
    has_mapped = bank_series.notna() & (bank_series.str.strip() != "")
    joined["_bank_name"] = bank_series.where(has_mapped, fallback_bank)

    # bank_firm_id: the secured-party Firm_ID resolved by the pipeline.
    if "Firm_ID" not in joined.columns:
        joined["Firm_ID"] = pd.NA
    joined["_bank_firm_id"] = joined["Firm_ID"]

    # Company key: prefer debtor name from raw (stripped, upper-cased) so
    # that 'ACME, INC.' and 'acme inc' share a key. Append DEBT_STATE to
    # disambiguate same-named companies in different states.
    if debtor_col:
        debtor_name_clean = joined[debtor_col].astype("string").str.strip()
        debtor_key = debtor_name_clean.str.upper().fillna("")
    else:
        debtor_name_clean = pd.Series([pd.NA] * len(joined), index=joined.index, dtype="string")
        debtor_key = pd.Series([""] * len(joined), index=joined.index, dtype="string")

    state_key = (
        joined["DEBT_STATE"].astype("string").fillna("")
        if "DEBT_STATE" in joined.columns
        else pd.Series([""] * len(joined), index=joined.index, dtype="string")
    )
    joined["_company_key"] = debtor_key + "||" + state_key
    joined["_company_display"] = debtor_name_clean

    # Drop rows where we have no usable bank label at all.
    has_bank_name = (
        joined["_bank_name"].notna()
        & (joined["_bank_name"].astype("string").str.strip() != "")
    )

    # Restrict to IN-UNIVERSE banks only. Per the user's scoping decision
    # (2026-06-16): non-Universe banks (no Firm_ID resolved) are excluded
    # entirely — both from the share denominator and from the output rows.
    # The lead-bank designation is therefore computed against the in-universe
    # subset only, so e.g. a debtor with 1 in-universe bank + 50 non-universe
    # filings will have that in-universe bank designated lead if it has >=2
    # filings (sole-in-universe-bank rule), because we only care about how
    # tracked banks rank against each other.
    has_firm_id = joined["_bank_firm_id"].notna()
    n_total_with_bank = int(has_bank_name.sum())
    n_in_universe = int((has_bank_name & has_firm_id).sum())
    if n_total_with_bank:
        context.log.info(
            f"lead_bank_designation: restricting to in-universe banks: "
            f"{n_in_universe:,}/{n_total_with_bank:,} filings retained "
            f"({n_in_universe / n_total_with_bank:.1%}); "
            f"{n_total_with_bank - n_in_universe:,} filings dropped because "
            "the secured party has no Firm_ID (bank not in Universe.csv)."
        )
    usable = has_bank_name & has_firm_id
    if not usable.any():
        context.log.warning(
            "lead_bank_designation: no in-universe banks found in any filing; "
            "producing empty output. Check that Universe.csv contains the "
            "banks you expect, and that the matching pipeline resolved them."
        )
        empty = pd.DataFrame(
            columns=[
                "Firm_ID", "company_name", "company_state",
                "bank_name", "bank_firm_id",
                "originals", "continuations", "other_filings", "total_filings",
                "company_total_filings", "company_bank_count",
                "share_of_company", "is_lead_bank", "lead_bank_reason",
            ]
        )
        return Output(value=empty, metadata={"row_count": MetadataValue.int(0)})

    work = joined.loc[usable].copy()

    # Drop rows where we don't have a real company identifier (empty key '||XX')
    # because aggregating across all anonymous debtors would be misleading.
    if not debtor_col:
        context.log.warning(
            "lead_bank_designation: aggregating by DEBT_STATE only because no "
            "debtor-name column is available; the resulting 'companies' are "
            "really per-state buckets. Configure debtor_company_column for "
            "true company-level granularity."
        )

    # ------------------------------------------------------------------
    # 5. Aggregate per (company, bank).
    # ------------------------------------------------------------------
    bank_key_cols = ["_company_key", "_bank_name"]
    pivot = (
        work
        .groupby(bank_key_cols + ["_filing_class"], dropna=False)
        .size()
        .unstack("_filing_class", fill_value=0)
        .reset_index()
    )
    for col in ("ORIGINAL", "CONTINUATION", "OTHER"):
        if col not in pivot.columns:
            pivot[col] = 0
    pivot["total_filings"] = pivot[["ORIGINAL", "CONTINUATION", "OTHER"]].sum(axis=1)

    # Carry through display / id columns.
    def _first_non_null(s: pd.Series):
        nn = s.dropna()
        return nn.iloc[0] if len(nn) else pd.NA

    def _mode_or_first(s: pd.Series):
        nn = s.dropna()
        if not len(nn):
            return pd.NA
        m = nn.mode()
        return m.iloc[0] if len(m) else nn.iloc[0]

    agg_spec = {
        "company_name": ("_company_display", _first_non_null),
        "bank_firm_id": ("_bank_firm_id", _mode_or_first),
    }
    if "DEBT_STATE" in work.columns:
        agg_spec["company_state"] = ("DEBT_STATE", _first_non_null)

    meta_agg = work.groupby(bank_key_cols, dropna=False).agg(**agg_spec).reset_index()
    counts = pivot.merge(meta_agg, on=bank_key_cols, how="left")

    # Company-level totals + bank counts for the share computation.
    company_totals = (
        counts.groupby("_company_key", dropna=False)
        .agg(
            company_total_filings=("total_filings", "sum"),
            company_bank_count=("_bank_name", "nunique"),
        )
        .reset_index()
    )
    counts = counts.merge(company_totals, on="_company_key", how="left")
    counts["share_of_company"] = np.where(
        counts["company_total_filings"] > 0,
        counts["total_filings"] / counts["company_total_filings"],
        0.0,
    )

    # ------------------------------------------------------------------
    # 6. Apply lead-bank rule.
    # ------------------------------------------------------------------
    threshold = float(getattr(config, "lead_bank_threshold", 0.66))
    counts["is_lead_bank"] = False
    counts["lead_bank_reason"] = ""

    # Single-bank / single-filing case -> never lead.
    single_bank_single_filing = (
        (counts["company_bank_count"] == 1) & (counts["company_total_filings"] <= 1)
    )
    counts.loc[single_bank_single_filing, "lead_bank_reason"] = (
        "Insufficient evidence: 1 bank, 1 filing"
    )

    # Single-bank / multi-filing case -> lead (sustained sole relationship).
    sole_bank_multi_filing = (
        (counts["company_bank_count"] == 1) & (counts["company_total_filings"] >= 2)
    )
    counts.loc[sole_bank_multi_filing, "is_lead_bank"] = True
    counts.loc[sole_bank_multi_filing, "lead_bank_reason"] = (
        "Sole bank with >= 2 filings"
    )

    # Multi-bank case -> lead only if this bank has the strict top total in
    # its company AND share >= threshold. Ties (rare with integer counts)
    # are broken by total desc then originals desc.
    multi_bank = counts["company_bank_count"] >= 2
    lead_keys: set = set()
    if multi_bank.any():
        sub = counts.loc[multi_bank].copy()
        sub = sub.sort_values(
            by=["_company_key", "total_filings", "ORIGINAL"],
            ascending=[True, False, False],
        )
        top_per_company = sub.drop_duplicates("_company_key", keep="first")

        lead_candidates = top_per_company[
            top_per_company["share_of_company"] >= threshold
        ].copy()

        if len(lead_candidates):
            # Tie-check: ensure no other bank in the same company has the
            # same total_filings as the candidate top.
            candidate_keys = list(zip(
                lead_candidates["_company_key"].tolist(),
                lead_candidates["total_filings"].tolist(),
            ))
            tie_lookup = (
                counts.groupby(["_company_key", "total_filings"]).size()
                .rename("n_at_total").reset_index()
            )
            # Use a row-positional zip to avoid itertuples renaming `_company_key`
            # (it strips leading underscores from column names).
            tie_lookup_map = dict(zip(
                zip(tie_lookup["_company_key"].tolist(),
                    tie_lookup["total_filings"].tolist()),
                tie_lookup["n_at_total"].tolist(),
            ))
            unique_mask = [
                tie_lookup_map.get((ck, tf), 0) == 1
                for ck, tf in candidate_keys
            ]
            lead_candidates = lead_candidates[pd.Series(unique_mask, index=lead_candidates.index)]

            lead_keys = set(zip(
                lead_candidates["_company_key"].tolist(),
                lead_candidates["_bank_name"].tolist(),
            ))

            row_keys = list(zip(counts["_company_key"], counts["_bank_name"]))
            lead_rows_mask = pd.Series(
                [k in lead_keys for k in row_keys],
                index=counts.index,
            )
            counts.loc[lead_rows_mask, "is_lead_bank"] = True
            # Populate per-row reason with the actual share / total.
            for idx in counts.index[lead_rows_mask]:
                counts.at[idx, "lead_bank_reason"] = (
                    f"Dominant bank: {counts.at[idx, 'share_of_company']:.0%} "
                    f"share across {int(counts.at[idx, 'total_filings'])} filings"
                )

        # Multi-bank companies with no winner -> record reason.
        winner_companies = {k for (k, _) in lead_keys}
        no_winner_companies = set(top_per_company["_company_key"]) - winner_companies
        if no_winner_companies:
            no_winner_mask = (
                counts["_company_key"].isin(no_winner_companies)
                & (counts["lead_bank_reason"] == "")
            )
            counts.loc[no_winner_mask, "lead_bank_reason"] = (
                f"No bank >= {threshold:.0%} share"
            )

    # ------------------------------------------------------------------
    # 7. Final tidy: rename internal columns, order output.
    # ------------------------------------------------------------------
    out = counts.rename(
        columns={
            "_bank_name": "bank_name",
            "ORIGINAL": "originals",
            "CONTINUATION": "continuations",
            "OTHER": "other_filings",
        }
    )
    # 'Firm_ID' here would refer to the *debtor* Firm_ID, which the pipeline
    # does not resolve (the matching pipeline maps secured parties only).
    # We leave the column for forward-compatibility but blank for now.
    out["Firm_ID"] = pd.NA
    if "company_state" not in out.columns:
        out["company_state"] = pd.NA

    ordered = [
        "Firm_ID", "company_name", "company_state",
        "bank_name", "bank_firm_id",
        "originals", "continuations", "other_filings", "total_filings",
        "company_total_filings", "company_bank_count",
        "share_of_company", "is_lead_bank", "lead_bank_reason",
    ]
    ordered = [c for c in ordered if c in out.columns]
    out = (
        out[ordered]
        .sort_values(
            by=["company_name", "total_filings"],
            ascending=[True, False],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------------
    # 8. Write CSVs (per-bank counts + per-company lead summary).
    # ------------------------------------------------------------------
    bank_counts_path = output_dir / f"bank_filing_counts_{timestamp}.csv"
    out.to_csv(bank_counts_path, index=False)
    context.log.info(
        f"Lead-bank stage: wrote per-(company, bank) counts -> {bank_counts_path}"
    )

    leads_only = out[out["is_lead_bank"]].copy()
    if len(leads_only):
        leads_only = leads_only.sort_values(
            ["company_name", "total_filings"], ascending=[True, False]
        ).drop_duplicates(
            subset=[c for c in ["company_name", "company_state"] if c in leads_only.columns],
            keep="first",
        )
    leads_path = output_dir / f"lead_bank_by_company_{timestamp}.csv"
    leads_only.to_csv(leads_path, index=False)
    context.log.info(
        f"Lead-bank stage: wrote per-company lead designation "
        f"({len(leads_only):,} companies with a lead) -> {leads_path}"
    )

    # ------------------------------------------------------------------
    # 9. Summary log + Dagster metadata.
    # ------------------------------------------------------------------
    company_id_cols = [c for c in ["company_name", "company_state"] if c in out.columns]
    n_companies = (
        int(out[company_id_cols].drop_duplicates().shape[0]) if company_id_cols else 0
    )
    n_pairs = len(out)
    n_leads = int(out["is_lead_bank"].sum())
    n_originals = int(out["originals"].sum())
    n_continuations = int(out["continuations"].sum())
    context.log.info(
        f"Lead-bank summary: {n_companies:,} companies | {n_pairs:,} (company, bank) pairs | "
        f"{n_originals:,} originals + {n_continuations:,} continuations | "
        f"{n_leads:,} lead-bank rows designated (threshold = {threshold:.0%})"
    )

    return Output(
        value=out,
        metadata={
            "companies": MetadataValue.int(n_companies),
            "company_bank_pairs": MetadataValue.int(n_pairs),
            "originals": MetadataValue.int(n_originals),
            "continuations": MetadataValue.int(n_continuations),
            "lead_bank_rows": MetadataValue.int(n_leads),
            "threshold": MetadataValue.float(threshold),
            "filing_type_column": MetadataValue.text(filing_type_col or "<none found>"),
            "debtor_column": MetadataValue.text(debtor_col or "<none found>"),
            "bank_counts_csv": MetadataValue.path(str(bank_counts_path)),
            "lead_bank_csv": MetadataValue.path(str(leads_path)),
        },
    )


@asset(
    description="Generate output files for different confidence levels",
    compute_kind="io",
    group_name="reporting",
    tags={"domain": "ucc-mapping", "stage": "reporting"},
)
def pipeline_outputs(
    context: AssetExecutionContext,
    config: PipelineConfig,
    universe_enriched_results: pd.DataFrame,
) -> MaterializeResult:
    """Generate output files for different confidence levels."""
    df = universe_enriched_results.copy()
    
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    base_cols = ['DEBT_STATE', 'SECP_STATE', 'SECP_NAME', 'SECP_NAME_NORMALIZED']
    match_cols = [
        'SECP_NAME_MAPPED',
        'matched_master_raw',
        'match_method',
        'match_score',
        'confidence',
        'ambiguous_master_candidates'
    ]
    cluster_cols = ['cluster_id', 'cluster_label', 'cluster_size']
    suggestion_cols = ['suggested_mapping', 'suggestion_score', 'suggestion_confidence', 'alternative_suggestions']
    universe_cols = ['Firm_ID']
    
    all_cols = base_cols + match_cols + cluster_cols + suggestion_cols + universe_cols
    available_cols = [c for c in all_cols if c in df.columns]
    
    output_df = df[available_cols].copy()
    
    # Auto-mapped
    auto_mask = output_df['confidence'] == 'AUTO'
    auto_df = output_df[auto_mask].sort_values(['match_method', 'SECP_NAME'])
    auto_path = output_dir / f"auto_mapped_{timestamp}.csv"
    auto_df.to_csv(auto_path, index=False)
    context.log.info(f"Auto-mapped: {len(auto_df):,} records -> {auto_path}")
    
    # Review needed
    review_mask = output_df['confidence'].isin(['REVIEW_HIGH', 'REVIEW_MEDIUM', 'REVIEW_LOW'])
    review_df = output_df[review_mask].sort_values(['confidence', 'match_score'], ascending=[True, False])
    review_path = output_dir / f"review_needed_{timestamp}.csv"
    review_df.to_csv(review_path, index=False)
    context.log.info(f"Review needed: {len(review_df):,} records -> {review_path}")
    
    # New entities
    new_mask = output_df['confidence'] == 'NEW_ENTITY'
    new_df = output_df[new_mask].sort_values(
        ['suggestion_confidence', 'cluster_size', 'cluster_id', 'SECP_NAME'],
        ascending=[True, False, True, True],
        na_position='last'
    )
    new_path = output_dir / f"new_entities_{timestamp}.csv"
    new_df.to_csv(new_path, index=False)
    context.log.info(f"New entities: {len(new_df):,} records -> {new_path}")
    
    # Full results
    full_path = output_dir / f"full_results_{timestamp}.csv"
    output_df.to_csv(full_path, index=False)
    context.log.info(f"Full results: {len(output_df):,} records -> {full_path}")

    # ------------------------------------------------------------------
    # Exact, Fuzzy, and Advanced master matches — slim export (deduped on
    # DEBT_STATE + SECP_STATE + SECP_NAME) and left-join onto the full raw UCC file. Unmatched rows
    # are excluded.
    # ------------------------------------------------------------------
    def _master_match_mask(series: pd.Series) -> pd.Series:
        s = series.astype("string")
        return series.notna() & (
            s.str.startswith("Exact Match", na=False)
            | s.str.startswith("Fuzzy Match", na=False)
            | s.str.startswith("Advanced Match", na=False)
        )

    enrich_cols = [
        "SECP_NAME_MAPPED",
        "match_method",
        "match_score",
        "confidence",
        "Firm_ID",
    ]
    ef_mask = _master_match_mask(df["match_method"])
    ef_source = df.loc[ef_mask].copy()

    n_before_dedup = len(ef_source)
    join_key_cols = [col for col in RAW_MATCH_COLUMNS if col in ef_source.columns]
    ef_dedup = ef_source.drop_duplicates(subset=join_key_cols, keep="first")
    n_dup_dropped = n_before_dedup - len(ef_dedup)
    if n_dup_dropped:
        context.log.info(
            f"Slim / raw-join lookup: dropped {n_dup_dropped:,} duplicate state/name "
            f"rows (keep=first) before export and merge"
        )

    slim_cols = [
        "DEBT_STATE",
        "SECP_STATE",
        "SECP_NAME",
        "SECP_NAME_MAPPED",
        "match_method",
        "match_score",
        "confidence",
        "Firm_ID",
    ]
    slim_available = [c for c in slim_cols if c in ef_dedup.columns]
    ef_slim = ef_dedup[slim_available].copy()
    ef_slim_path = output_dir / f"exact_fuzzy_mapped_{timestamp}.csv"
    ef_slim.to_csv(ef_slim_path, index=False)
    context.log.info(
        f"Master-matched slim (deduped state/name): {len(ef_slim):,} records -> {ef_slim_path}"
    )

    raw_mapped_path = None
    raw_mapped_rows = 0
    raw_with_mapping = 0

    if not all(col in ef_dedup.columns for col in RAW_MATCH_COLUMNS):
        context.log.warning(
            "Enriched dataframe missing state/name join columns; skipping raw merge"
        )
    else:
        map_for_join = ef_dedup[RAW_MATCH_COLUMNS + enrich_cols].drop_duplicates(
            subset=RAW_MATCH_COLUMNS,
            keep="first",
        )
        try:
            raw_full = pd.read_csv(
                config.ucc_file,
                sep=_detect_delimiter(config.ucc_file),
                encoding="latin-1",
            )
        except Exception as e:
            context.log.error(f"Could not read full raw UCC for merge: {e}")
        else:
            if not all(col in raw_full.columns for col in RAW_MATCH_COLUMNS):
                context.log.warning(
                    "Raw file missing state/name join columns; skipping raw merge"
                )
            else:
                raw_full = _standardize_state_columns(raw_full)
                raw_enriched = raw_full.merge(
                    map_for_join,
                    on=RAW_MATCH_COLUMNS,
                    how="left",
                )
                raw_mapped_path = output_dir / f"raw_mapped_with_firm_ids_{timestamp}.csv"
                raw_enriched.to_csv(raw_mapped_path, index=False)
                raw_mapped_rows = len(raw_enriched)
                raw_with_mapping = int(
                    raw_enriched["SECP_NAME_MAPPED"].notna().sum()
                )
                context.log.info(
                    f"Raw mapped output (left join on DEBT_STATE + SECP_STATE + SECP_NAME): "
                    f"{raw_mapped_rows:,} rows, "
                    f"{raw_with_mapping:,} with SECP_NAME_MAPPED and Firm_ID "
                    f"-> {raw_mapped_path}"
                )
    
    # Summary
    summary_path = output_dir / f"summary_{timestamp}.txt"
    method_counts = output_df['match_method'].value_counts().to_dict()
    confidence_counts = output_df['confidence'].value_counts().to_dict()
    
    with open(summary_path, 'w') as f:
        f.write("UCC Mapping Pipeline Summary (Dagster)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Overall Results:\n")
        f.write(f"  Total UCC Records: {len(df):,}\n")
        f.write(f"  Auto-Mapped: {len(auto_df):,}\n")
        f.write(f"  Review Needed: {len(review_df):,}\n")
        f.write(f"  New Entities: {len(new_df):,}\n")
        
        f.write("\nBy Match Method:\n")
        for method, count in sorted(method_counts.items(), key=lambda x: -x[1]):
            f.write(f"  {method}: {count:,}\n")
        
        f.write("\nBy Confidence Level:\n")
        for conf, count in sorted(confidence_counts.items(), key=lambda x: -x[1]):
            f.write(f"  {conf}: {count:,}\n")
        
        if new_mask.any():
            n_clusters = output_df.loc[new_mask, 'cluster_id'].nunique()
            avg_size = output_df.loc[new_mask, 'cluster_size'].mean()
            f.write(f"\nClustering Stats:\n")
            f.write(f"  Number of clusters: {n_clusters:,}\n")
            f.write(f"  Average cluster size: {avg_size:.1f}\n")

        # Universe coverage
        if 'Firm_ID' in output_df.columns:
            with_firm = output_df['Firm_ID'].notna().sum()
            f.write(f"\nTracked Bank Coverage:\n")
            f.write(f"  Records with Firm_ID:         {with_firm:,}\n")

        f.write("\nExact / Fuzzy / Advanced (excludes Unmatched); slim deduped on state/name:\n")
        f.write(f"  Slim export rows: {len(ef_slim):,}\n")
        if raw_mapped_path is not None:
            f.write(
                f"  Raw mapped-with-firm-ids file rows: {raw_mapped_rows:,} "
                f"({raw_with_mapping:,} rows with SECP_NAME_MAPPED filled)\n"
            )

    firm_matched = int(output_df['Firm_ID'].notna().sum()) if 'Firm_ID' in output_df.columns else 0
    auto_pct = len(auto_df) / len(df) * 100 if len(df) > 0 else 0
    
    meta = {
        "total_records": MetadataValue.int(len(df)),
        "auto_mapped_count": MetadataValue.int(len(auto_df)),
        "auto_mapped_pct": MetadataValue.float(auto_pct),
        "review_needed_count": MetadataValue.int(len(review_df)),
        "new_entities_count": MetadataValue.int(len(new_df)),
        "firm_id_matched_count": MetadataValue.int(firm_matched),
        "mapping_slim_deduped_count": MetadataValue.int(len(ef_slim)),
        "output_dir": MetadataValue.path(str(output_dir)),
        "timestamp": MetadataValue.text(timestamp),
    }
    if raw_mapped_path is not None:
        meta["raw_mapped_with_firm_ids_rows"] = MetadataValue.int(raw_mapped_rows)
        meta["raw_mapped_with_firm_ids_with_mapping"] = MetadataValue.int(
            raw_with_mapping
        )
        meta["raw_mapped_with_firm_ids_path"] = MetadataValue.path(str(raw_mapped_path))
    return MaterializeResult(metadata=meta)

