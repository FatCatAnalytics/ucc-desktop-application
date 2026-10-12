# UCC Retrieval and Mapping Pipeline — Complete Technical Documentation

> **Version:** 1.0  
> **Framework:** Dagster (asset-based orchestration)  
> **Language:** Python 3.14  
> **ML Models:** SentenceTransformers (bge-small-en-v1.5), FAISS, rapidfuzz  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)  
2. [Architecture Overview](#2-architecture-overview)  
3. [Configuration — `resources.py`](#3-configuration--resourcespy)  
4. [Constants & Pattern Definitions — `constants.py`](#4-constants--pattern-definitions--constantspy)  
5. [Data Structures — `data_structures.py`](#5-data-structures--data_structurespy)  
6. [Utility Functions — `utils.py`](#6-utility-functions--utilspy)  
7. [Pipeline Stages — `assets.py`](#7-pipeline-stages--assetspy)  
8. [Dagster Wiring — `definitions.py`](#8-dagster-wiring--definitionspy)  
9. [False Positive Prevention System](#9-false-positive-prevention-system)  
10. [Test Suite — `test_normalize.py`](#10-test-suite--test_normalizepy)  
11. [Output Files & Reporting](#11-output-files--reporting)  
12. [Performance & Optimization](#12-performance--optimization)  
13. [Appendix: Complete Function Reference](#13-appendix-complete-function-reference)

---

## 1. Executive Summary

This pipeline maps **Uniform Commercial Code (UCC) filing records** (secured party names) to a canonical **master mapping table**. The challenge is that UCC filings contain raw, unstandardized entity names with inconsistent formatting, abbreviations, legal suffixes, and spelling variations. The pipeline progressively resolves these through a **seven-stage matching and clustering process**:

| Stage | Method | Typical Confidence |
|-------|--------|--------------------|
| **Stage 1** | Exact match on raw names | AUTO (100%) |
| **Stage 2** | Exact match on normalized names | AUTO / REVIEW_HIGH |
| **Stage 3** | Fuzzy string matching | AUTO / REVIEW_HIGH |
| **Stage 4** | Advanced multi-signal embedding match | AUTO / REVIEW_HIGH / REVIEW_MEDIUM |
| **Stage 5** | Clustering of unmatched records | NEW_ENTITY |
| **Stage 6** | Hierarchical cluster merging | NEW_ENTITY |
| **Stage 7** | Suggested mappings for clusters | SUGGEST_HIGH / SUGGEST_MEDIUM / SUGGEST_LOW |

Each subsequent stage only processes records that remain **unmatched** from prior stages (waterfall strategy), ensuring no double-processing and preserving higher-confidence results.

---

## 2. Architecture Overview

### 2.1 Dagster Asset Graph

```
ucc_raw_data ──────────────────────────┐
                                       ▼
master_mapping_data ──► normalized_master_data ──► master_embeddings_cache
                                       │                    │
                                       ▼                    │
                              exact_match_raw_results       │
                                       │                    │
                                       ▼                    │
                         exact_match_normalized_results     │
                                       │                    │
                                       ▼                    │
                              fuzzy_match_results           │
                                       │                    │
                                       ▼                    ▼
                         advanced_embedding_results ◄───────┘
                                       │                    │
                                       ▼                    │
                              clustered_results ◄───────────┘
                                       │                    │
                                       ▼                    │
                           merged_cluster_results ◄─────────┘
                                       │
                                       ▼
                         suggested_mapping_results
                                       │
                                       ▼
                              pipeline_outputs
```

### 2.2 File Structure

```
ucc_mapping_project/
├── ucc_mapping_project/
│   ├── __init__.py
│   ├── definitions.py        # Dagster wiring (assets + resources)
│   ├── assets.py              # 7 pipeline stages + ingestion + output
│   ├── resources.py           # PipelineConfig (all thresholds & paths)
│   ├── constants.py           # Regex patterns, abbreviations, entity types
│   └── utils.py               # Core matching/normalization logic
├── models/
│   ├── data_structures.py     # NGramIndex, TFIDFNGramIndex, UnionFind
│   └── bge-small-en-v1.5/    # Local SentenceTransformer model
├── Inputs/
│   ├── ucc.csv                # Raw UCC filing records
│   ├── Master Mapping.csv     # Known good mappings with state-aware keys
│   └── Universe.csv           # Tracked bank names and Firm IDs
├── Outputs/                   # Generated CSV files per confidence level
├── tests/
│   └── test_normalize.py      # 67 unit tests
└── setup.py / pyproject.toml
```

### 2.3 Input Data

| File | Columns | Rows | Description |
|------|---------|------|-------------|
| `ucc.csv` | `DEBT_STATE`, `SECP_STATE`, `SECP_NAME` | varies | Raw UCC filing records — each row is a secured party from a UCC filing |
| `Master Mapping.csv` | `DEBT_STATE`, `SECP_STATE`, `SECP_NAME`, `SECP_NAME_MAPPED`, `Firm ID` | varies | Known good mappings with state-aware match keys |
| `Universe.csv` | `Bank Name`, `Firm ID` | varies | Tracked bank names used to enrich results with Firm IDs |

---

## 3. Configuration — `resources.py`

The `PipelineConfig` class extends Dagster's `ConfigurableResource` and centralizes **every tunable parameter** for the pipeline. This ensures reproducibility and easy experimentation.

### 3.1 Path Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ucc_file` | `Inputs/ucc.csv` | Path to UCC filings (resolved from project root) |
| `master_file` | `Inputs/Master Mapping.csv` | Path to master mapping table |
| `universe_file` | `Inputs/Universe.csv` | Path to tracked bank Firm ID table |
| `output_dir` | `Outputs` | Directory for all output files |
| `embedding_model` | `models/bge-small-en-v1.5` | Local path to the SentenceTransformer model |

All paths are resolved as **absolute paths** using `Path(__file__).parent` to avoid platform-specific issues.

### 3.2 Fuzzy Matching Thresholds

| Parameter | Value | Description |
|-----------|-------|-------------|
| `fuzzy_auto_threshold` | 96 | Score ≥ 96 → AUTO confidence (no human review needed) |
| `fuzzy_review_threshold` | 85 | Score ≥ 85 → REVIEW_HIGH (needs human verification) |
| `fuzzy_min_score` | 80 | Absolute floor — scores below this are discarded entirely |
| `fuzzy_batch_size` | 500 | Number of queries processed per fuzzy batch |
| `fuzzy_workers` | 1 | Thread count for `rapidfuzz.process.cdist` |

### 3.3 Embedding & Multi-Signal Thresholds

| Parameter | Value | Description |
|-----------|-------|-------------|
| `embed_auto_threshold` | 0.95 | Composite score ≥ 0.95 → AUTO |
| `embed_high_threshold` | 0.90 | Composite score ≥ 0.90 → REVIEW_HIGH |
| `embed_medium_threshold` | 0.85 | Composite score ≥ 0.85 → REVIEW_MEDIUM |
| `embed_batch_size` | 64 | Batch size for SentenceTransformer encoding |
| `embed_top_k` | 20 | Number of nearest neighbors retrieved from FAISS |
| `max_candidates_per_query` | 50 | Cap on total candidates (FAISS + n-gram) |
| `early_stop_threshold` | 0.97 | Stop searching once score exceeds this |

### 3.4 Clustering Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `cluster_sim_threshold` | 0.88 | Minimum embedding cosine similarity to consider clustering |
| `cluster_fuzzy_guard` | 60 | Minimum `token_sort_ratio` for cluster linkage |
| `cluster_k_neighbors` | 15 | Neighbors searched per record in FAISS |
| `cluster_merge_threshold` | 0.88 | Embedding threshold for hierarchical merge |
| `cluster_merge_iterations` | 3 | Maximum rounds of hierarchical merging |
| `cluster_merge_k` | 10 | Neighbors checked during merge |
| `cluster_max_size` | 500 | Maximum records per cluster (prevents mega-clusters) |

### 3.5 HNSW Parameters (Large Dataset Optimization)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `cluster_hnsw_threshold` | 200,000 | Switch from brute-force to HNSW above this count |
| `cluster_hnsw_m` | 16 | HNSW graph connectivity (higher = more accurate but slower) |
| `cluster_hnsw_ef_search` | 64 | Search beam width |
| `cluster_hnsw_ef_construction` | 100 | Construction quality parameter |

### 3.6 CPU & Threading

| Parameter | Value | Description |
|-----------|-------|-------------|
| `cpu_threads` | 1 | BLAS/MKL threads per process |
| `embedding_workers` | 2 | ThreadPool workers for auxiliary computations |
| `search_batch_size` | 4096 | FAISS search batch size |

---

## 4. Constants & Pattern Definitions — `constants.py`

This module defines the **regex patterns**, **abbreviation dictionaries**, and **entity type classification rules** that drive normalization.

### 4.1 `LEGAL_SUFFIXES` — Legal Entity Suffixes

A list of 15 compiled regex patterns that match legal entity type indicators such as:

- `LLC`, `L.L.C.`, `L L C`
- `INC`, `INCORPORATED`
- `CORP`, `CORPORATION`
- `LTD`, `LIMITED`
- `LP`, `LLP`, `LLLP` (limited partnerships)
- `NA`, `N.A.` (national association)
- `CO`, `COMPANY`
- `PLC`, `PLLC`
- `PC`, `PA` (professional corporations/associations)

Each pattern uses `\b` word boundaries and optional punctuation to handle variations like `L.L.C.` vs `LLC` vs `L L C`.

> **Design decision:** `BANK` and `BANKING` were deliberately **removed** from this list to prevent entity names like "US BANK" from being stripped down to just "US", which would cause massive false positive matches.

### 4.2 `ROLE_TRIGGERS` — Role Phrase Detection

16 regex patterns that identify **role descriptions** appended to entity names:

```
AS TRUSTEE, AS AGENT, AS CUSTODIAN, AS FIDUCIARY,
AS INDENTURE TRUSTEE, AS COLLATERAL AGENT, AS ADMINISTRATIVE AGENT,
AS SUCCESSOR, AS ASSIGNEE, IN ITS CAPACITY AS, BY AND THROUGH,
AS SERVICER, AS LENDER, AS SECURED PARTY, DBA, D/B/A
```

The compiled `ROLE_RE` regex removes everything from the first matching trigger onward. For example:

```
"WELLS FARGO BANK AS TRUSTEE FOR ABC TRUST" → "WELLS FARGO BANK"
```

### 4.3 `COMMON_WORDS` — Generic Word Removal

A list of 25 commonly occurring but non-distinguishing words, **sorted by length descending** to ensure longer phrases are removed before shorter ones (preventing partial removals):

```
EQUIPMENT FINANCE, THE, OF, AND, FOR, GROUP, SERVICES,
FINANCIAL, HOLDINGS, INTERNATIONAL, CAPITAL, MANAGEMENT,
CREDIT, COMMERCIAL, SOLUTIONS, PARTNERS, ASSOCIATES, GLOBAL,
ENTERPRISES, INDUSTRIES, SYSTEMS, TECHNOLOGIES, RESOURCES,
PROPERTIES, VENTURES
```

### 4.4 `ABBREVIATIONS` — Standard Abbreviation Expansion

18 abbreviation-to-full-form mappings:

| Abbreviation | Expansion |
|-------------|-----------|
| `INTL` | `INTERNATIONAL` |
| `NATL` | `NATIONAL` |
| `ASSN` | `ASSOCIATION` |
| `GOVT` | `GOVERNMENT` |
| `FED` | `FEDERAL` |
| `MTG` | `MORTGAGE` |
| `INS` | `INSURANCE` |
| `SVCS` | `SERVICES` |
| `MGMT` | `MANAGEMENT` |
| `DEPT` | `DEPARTMENT` |
| `UNIV` | `UNIVERSITY` |
| `CTR` | `CENTER` |
| `DIST` | `DISTRICT` |
| `AUTH` | `AUTHORITY` |
| `COMM` | `COMMISSION` |
| `DEV` | `DEVELOPMENT` |
| `TECHS` | `TECHNOLOGIES` |
| `SYS` | `SYSTEMS` |

### 4.5 `ENTITY_TYPE_PATTERNS` — Entity Classification Rules

Five categories, each with a list of keyword patterns used by `detect_entity_type()`:

| Type | Keywords |
|------|----------|
| **BANK** | `BANK`, `BANKING`, `SAVINGS`, `CREDIT UNION`, `FEDERAL CREDIT`, `THRIFT` |
| **GOVERNMENT** | `STATE OF`, `CITY OF`, `COUNTY OF`, `DEPARTMENT`, `AGENCY`, `FEDERAL`, `AUTHORITY`, `COMMISSION`, `BUREAU`, `MUNICIPAL`, `GOVERNMENT` |
| **INVESTMENT** | `FUND`, `CAPITAL`, `INVESTMENT`, `VENTURE`, `EQUITY`, `ASSET MANAGEMENT`, `SECURITIES`, `HEDGE`, `PORTFOLIO` |
| **MANUFACTURER** | `MANUFACTURING`, `MFG`, `INDUSTRIES`, `INDUSTRIAL`, `FACTORY`, `FABRICAT` |
| **LEGAL** | `TRUST`, `TRUSTEE`, `ESTATE`, `CUSTODIAN`, `FIDUCIARY`, `GUARDIAN`, `CONSERVATOR` |

If no patterns match, the entity is classified as `OTHER`.

---

## 5. Data Structures — `data_structures.py`

### 5.1 `NGramIndex`

An inverted index for fast character n-gram lookup.

**Purpose:** Given a query name, quickly find master records that share significant character n-gram overlap — acts as a candidate retrieval mechanism complementary to FAISS embedding search.

#### Methods

| Method | Description |
|--------|-------------|
| `__init__(n=3)` | Initialize with n-gram size (default: trigrams) |
| `add(idx, name)` | Generate n-grams from `name` and store in inverted index |
| `add_precomputed(idx, name, ngrams)` | Add pre-computed n-grams (used for parallel construction) |
| `query(name, min_overlap=2)` | Return indices with ≥ `min_overlap` shared n-grams |

**Internal structure:**
- `_index`: `Dict[str, Set[int]]` — maps each n-gram string to the set of record indices containing it
- `_ngrams`: `Dict[int, Set[str]]` — maps each record index to its set of n-grams

### 5.2 `TFIDFNGramIndex`

An IDF-weighted n-gram index for computing **cosine similarity** between names using TF-IDF weighted character n-gram vectors.

**Purpose:** Standard Jaccard n-gram similarity treats all n-grams equally. This structure weights n-grams by their **inverse document frequency** — rare n-grams (unique to a few names) get higher weight, common n-grams (shared by many names) get lower weight. This dramatically improves matching quality for names with common prefixes.

#### Methods

| Method | Description |
|--------|-------------|
| `__init__(n=3)` | Initialize with n-gram size |
| `add(idx, name)` | Register a document's n-grams (pre-freeze) |
| `freeze()` | Compute IDF weights as `log(N / df)` for each n-gram. Must be called before `similarity()`. |
| `similarity(name1, name2)` | Compute TF-IDF weighted cosine similarity between two names |

**IDF Calculation:**
$$\text{IDF}(t) = \ln\left(\frac{N}{df(t)}\right)$$

Where $N$ is the total number of documents and $df(t)$ is the number of documents containing n-gram $t$.

**Cosine Similarity:**
$$\text{sim}(a, b) = \frac{\sum_{t \in a \cap b} w_t^2}{\sqrt{\sum_{t \in a} w_t^2} \cdot \sqrt{\sum_{t \in b} w_t^2}}$$

Where $w_t = \text{IDF}(t)$ is the weight for n-gram $t$.

### 5.3 `UnionFind`

A disjoint-set (union-find) data structure with **path compression** and **union-by-rank** optimizations.

**Purpose:** Used in Stage 5 (Clustering) and Stage 6 (Hierarchical Merging) to efficiently group records into clusters. When two records are determined to be "the same entity," their clusters are merged via `union()`.

#### Methods

| Method | Description |
|--------|-------------|
| `__init__(n)` | Initialize `n` elements, each in its own singleton set |
| `find(x)` | Find the root representative of `x`'s set, with path compression |
| `union(x, y)` | Merge the sets containing `x` and `y`, using union-by-rank |
| `get_clusters()` | Return `Dict[int, List[int]]` mapping root → list of members |

**Complexity:** Nearly O(1) amortized per `find`/`union` operation due to path compression + union-by-rank (inverse Ackermann).

---

## 6. Utility Functions — `utils.py`

This is the core algorithmic module. It contains all normalization, scoring, phonetic encoding, and false-positive prevention logic.

### 6.1 `normalize_name(name: str) → str`

**The single most important function in the pipeline.** Transforms a raw entity name into a canonical normalized form for comparison.

#### Normalization Steps (in order)

1. **Uppercase conversion**: `name.upper()`
2. **Role stripping**: Apply `ROLE_RE` regex to remove everything after role triggers (e.g., "AS TRUSTEE", "DBA")
3. **Legal suffix removal**: Strip all 15 `LEGAL_SUFFIXES` patterns (LLC, INC, CORP, etc.)
4. **Common word removal**: Remove all words in `COMMON_WORDS` list (processed longest-first to prevent partial removal)
5. **Abbreviation expansion**: Replace all 18 abbreviations with full forms (INTL → INTERNATIONAL, etc.)
6. **Punctuation stripping**: Remove all non-alphanumeric characters except spaces
7. **Whitespace normalization**: Collapse multiple spaces to single space and strip leading/trailing

#### Example Transformations

| Input | Output |
|-------|--------|
| `"WELLS FARGO BANK, N.A. AS TRUSTEE"` | `"WELLS FARGO BANK"` |
| `"ABC EQUIPMENT FINANCE, LLC"` | `"ABC"` |
| `"THE FIRST NATL BANK OF TEXAS"` | `"FIRST NATIONAL BANK TEXAS"` |
| `"JOHN DEERE FINANCIAL, INC. DBA JOHN DEERE"` | `"JOHN DEERE"` |

### 6.2 `soundex(name: str) → str`

Implements the **ANSI/AIM Soundex** phonetic encoding algorithm with the **H/W separator rule**.

**Purpose:** Two names that sound alike but are spelled differently will produce the same Soundex code. Used as a bonus signal in the multi-signal scorer.

#### Algorithm

1. Keep the first letter of the name
2. Replace consonants with digits according to the Soundex mapping:
   - B, F, P, V → 1
   - C, G, J, K, Q, S, X, Z → 2
   - D, T → 3
   - L → 4
   - M, N → 5
   - R → 6
3. **H/W separator rule**: H and W act as separators — two consonants separated only by H or W that would produce the same digit are coded as a single digit
4. Remove vowels and Y
5. Pad or truncate to exactly 4 characters

#### Examples

| Name | Soundex |
|------|---------|
| `"ROBERT"` | `R163` |
| `"RUPERT"` | `R163` |
| `"ASHCRAFT"` | `A261` |

### 6.3 `detect_entity_type(name: str) → str`

Classifies an entity into one of six categories based on keyword pattern matching against `ENTITY_TYPE_PATTERNS`.

**Purpose:** Prevents cross-type false matches. A bank should not match a manufacturer, even if they share similar words.

#### Return Values

`'BANK'` | `'GOVERNMENT'` | `'INVESTMENT'` | `'MANUFACTURER'` | `'LEGAL'` | `'OTHER'`

The function checks patterns in priority order and returns the first match.

### 6.4 `generate_ngrams(name: str, n: int = 3) → Set[str]`

Generates character-level n-grams from a name string.

**Example:** `generate_ngrams("WELLS", 3)` → `{"WEL", "ELL", "LLS"}`

### 6.5 `ngram_similarity(name1: str, name2: str, n: int = 3) → float`

Computes **Jaccard similarity** between the n-gram sets of two names:

$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

Returns a float in [0.0, 1.0].

### 6.6 `tfidf_ngram_similarity(name1, name2, idf_weights, n=3) → float`

Computes **TF-IDF weighted cosine similarity** using externally provided IDF weights.

**Advantage over plain Jaccard:** Rare n-grams that are distinctive to a particular entity (e.g., "QUA" in "QUAKER") get much higher weight than common n-grams (e.g., "THE", "ING").

If `idf_weights` is `None`, falls back to unweighted Jaccard `ngram_similarity()`.

### 6.7 `extract_acronym(name: str) → str`

Extracts an acronym from a multi-word name by taking the first letter of each word.

**Example:** `extract_acronym("FEDERAL DEPOSIT INSURANCE")` → `"FDI"`

### 6.8 `is_acronym_match(name1: str, name2: str) → bool`

Checks if one name is an acronym of the other.

**Examples:**
- `is_acronym_match("IBM", "INTERNATIONAL BUSINESS MACHINES")` → `True`
- `is_acronym_match("FDIC", "FEDERAL DEPOSIT INSURANCE CORPORATION")` → `True`

### 6.9 `compute_match_score(query, master, embed_sim, ...) → Tuple[float, Dict]`

**The multi-signal scoring function.** Combines five independent similarity signals into a single composite score.

#### Signal Weights (Default)

| Signal | Weight | Range | Description |
|--------|--------|-------|-------------|
| **Phonetic (Soundex)** | 0.10 | 0 or 1 | Binary: 1 if Soundex codes match, 0 otherwise |
| **N-gram** | 0.15 | [0, 1] | TF-IDF weighted cosine similarity of character trigrams |
| **Fuzzy** | 0.25 | [0, 1] | `combined_fuzzy_score()` / 100 |
| **Embedding** | 0.40 | [0, 1] | FAISS cosine similarity from bge-small-en-v1.5 |
| **Acronym** | 0.10 | 0 or 1 | Binary: 1 if acronym match detected, 0 otherwise |

#### Formula

$$\text{score} = w_p \cdot S_p + w_n \cdot S_n + w_f \cdot S_f + w_e \cdot S_e + w_a \cdot S_a$$

#### Additional Bonuses (Applied in Stage 4)

- **Soundex match bonus:** +0.05 if query and master Soundex codes are identical
- **Entity type match bonus:** +0.03 if both entities have the same non-OTHER type

### 6.10 `combined_fuzzy_score(name1: str, name2: str) → float`

A **weighted ensemble** of four rapidfuzz scoring functions with **five guardrails** against false positives.

#### Component Weights

| Scorer | Weight | Description |
|--------|--------|-------------|
| `fuzz.ratio` | 0.30 | Character-level Levenshtein similarity |
| `fuzz.partial_ratio` | 0.10 | Best partial substring match |
| `fuzz.token_sort_ratio` | 0.50 | Token-sorted Levenshtein (order-independent) |
| `fuzz.token_set_ratio` | 0.10 | Set-based overlap (most permissive) |

#### Five Guardrails

1. **Meaningful identifier check**: Calls `has_shared_meaningful_identifier()`. If names share no meaningful words → return 0.0
2. **Minimum `fuzz.ratio` of 50**: If raw character similarity < 50 → return 0.0
3. **Token overlap check via `min_token_overlap()`**: Names must share at least 1 token of length > 2 → else return 0.0
4. **Entity type compatibility**: If `entity_types_compatible()` returns False → return 0.0
5. **Score ceiling from `fuzz.ratio`**: Final score is capped at `fuzz.ratio + 10` to prevent inflated scores from `partial_ratio` or `token_set_ratio`

### 6.11 `get_meaningful_words(name: str) → Set[str]`

**Two-tier word classification** — the core of false-positive prevention.

#### Tier 1: Entity Type Words (ALWAYS Excluded)

Words that describe what kind of entity it is, never what distinguishes one entity from another:

```
BANK, BANKING, TRUST, FINANCIAL, CREDIT, UNION, SAVINGS,
LENDING, LOAN, MORTGAGE, INSURANCE, FUND, CAPITAL,
INVESTMENT, SECURITIES, CORP, CORPORATION, INC,
INCORPORATED, LLC, LTD, LIMITED, COMPANY, CO, GROUP,
HOLDINGS, PARTNERS, ASSOCIATES, SERVICES, SOLUTIONS,
NATIONAL, FEDERAL, STATE, EQUIPMENT, FINANCE, COMMERCIAL
```

#### Tier 2: Generic Prefixes (Conditionally Excluded)

Common geographic or ordinal prefixes. These are excluded **only if meaningful words remain after exclusion**:

```
AMERICAN, UNITED, US, USA, NEW, OLD, CENTRAL, WESTERN,
EASTERN, NORTHERN, SOUTHERN, NORTH, SOUTH, EAST, WEST,
PACIFIC, ATLANTIC, CONTINENTAL, GENERAL, ROYAL, PREMIER,
GLOBAL, INTERNATIONAL, UNIVERSAL, GREAT, GREATER,
COMMUNITY, REGIONAL, METRO, METROPOLITAN, CITIZENS,
PEOPLES, SECURITY, COMMERCE, FIRST, SECOND, THIRD
```

#### Logic

1. Tokenize name → lowercase → filter words length > 1
2. Remove all entity type words (Tier 1)
3. If remaining words contain any non-generic-prefix words → also remove generic prefixes (Tier 2)
4. If removing generic prefixes would leave nothing → keep them

#### Examples

| Name | Entity Type Words Removed | Generic Prefixes | Result |
|------|--------------------------|-------------------|--------|
| `"AMERICAN EXPRESS"` | none removed | AMERICAN removed (EXPRESS remains) | `{'express'}` |
| `"AMERICAN HERITAGE NATIONAL BANK"` | NATIONAL, BANK removed | AMERICAN removed (HERITAGE remains) | `{'heritage'}` |
| `"FIRST BANK"` | BANK removed | FIRST kept (nothing else remains) | `{'first'}` |
| `"US BANK"` | BANK removed | US kept (nothing else remains) | `{'us'}` |
| `"FIRST CITIZENS BANK"` | BANK removed | FIRST, CITIZENS both generic (nothing else) → kept | `{'first', 'citizens'}` |
| `"FIRST AMERICAN BANK"` | BANK removed | FIRST, AMERICAN both generic (nothing else) → kept | `{'first', 'american'}` |

### 6.12 `has_shared_meaningful_identifier(name1: str, name2: str) → bool`

Determines whether two entity names share enough meaningful identifying words to be considered potential matches.

#### Logic (Decision Tree)

```
get_meaningful_words(name1) → words1
get_meaningful_words(name2) → words2

IF words1 is empty OR words2 is empty → FALSE
shared = words1 ∩ words2
IF shared is empty → FALSE
IF words1 == words2 → TRUE
IF smaller ⊆ larger:
    IF smaller has unique words (not in generic_prefixes) → TRUE
    ELSE → FALSE  (all words are generic → subset isn't meaningful)
IF overlap1 > 50% AND overlap2 > 50% → TRUE  (bidirectional overlap)
ELSE → FALSE
```

#### Key Examples

| Name 1 | Name 2 | Result | Reason |
|--------|--------|--------|--------|
| `"AMERICAN HERITAGE NATIONAL BANK"` | `"AMERICAN EXPRESS"` | **FALSE** | words1={heritage}, words2={express}, no overlap |
| `"WELLS FARGO BANK"` | `"WELLS FARGO FINANCIAL"` | **TRUE** | words1={wells,fargo}, words2={wells,fargo}, identical |
| `"CITIZENS BANK"` | `"FIRST CITIZENS BANK"` | **TRUE** | Both have {first,citizens}, identical after processing |
| `"FIRST BANK"` | `"FIRST AMERICAN BANK"` | **FALSE** | words1={first}, words2={first,american}, subset but "first" is generic only |

### 6.13 `extract_entity_core(name: str) → str`

Extracts the **core identifying portion** of an entity name — the single most distinctive word.

#### Strategy

1. Define `skippable_prefixes` (geographic/ordinal words) and `generic_words` (legal/industry terms)
2. Walk through words left-to-right, skip any word in either set
3. Return the first word that's in neither set and has length > 1
4. **Fallback:** If no meaningful word found, return the first two non-article words as a compound (e.g., "FIRST AMERICAN" for "FIRST AMERICAN BANK")

#### Examples

| Input | Core | Explanation |
|-------|------|-------------|
| `"STEARNS BANK EQUIPMENT FINANCE"` | `STEARNS` | First meaningful word |
| `"AMERICAN HERITAGE NATIONAL BANK"` | `HERITAGE` | Skip AMERICAN (generic prefix), HERITAGE is meaningful |
| `"AMERICAN EXPRESS"` | `EXPRESS` | Skip AMERICAN, EXPRESS is meaningful |
| `"US BANK"` | `US BANK` | No meaningful word → fallback to first 2 |
| `"FIRST CHOICE BANK"` | `CHOICE` | Skip FIRST, CHOICE is meaningful |
| `"FIRST CITIZENS BANK"` | `FIRST` | CITIZENS is also generic → fallback, FIRST is first non-article |

### 6.14 `min_token_overlap(name1, name2, min_overlap=1) → bool`

Quick check whether two names share at least `min_overlap` tokens (words with length > 2).

**Used as a pre-filter** in Stage 4 to skip candidate pairs that have no token-level overlap, even if embedding similarity is moderate.

### 6.15 `entity_types_compatible(type1, type2) → bool`

Checks whether two entity types are compatible for matching.

#### Compatibility Rules

- **Same type** → always compatible
- **Either is OTHER** → compatible (OTHER is a catch-all)
- **BANK ↔ LEGAL** → compatible (banks often act as trustees)
- **INVESTMENT ↔ LEGAL** → compatible (investment firms as custodians)
- All other cross-type pairs → **incompatible** (e.g., BANK ↔ MANUFACTURER)

### 6.16 `should_merge_clusters(label_a, label_b, embedding_sim) → bool`

**Conservative merge decision function** for Stage 6. Requires BOTH high embedding similarity AND strong fuzzy text similarity.

#### Decision Tiers

| Embedding Similarity | Required Fuzzy Scores | Rationale |
|---------------------|----------------------|-----------|
| ≥ 0.92 | `token_set_ratio` ≥ 80 AND `ratio` ≥ 60 | Very high embedding → moderate fuzzy guard |
| ≥ 0.88 | `token_set_ratio` ≥ 85 AND `partial_ratio` ≥ 85 | High embedding → strong fuzzy required |
| ≥ 0.85 | `token_set_ratio` ≥ 90 AND `partial_ratio` ≥ 90 AND `ratio` ≥ 75 | Moderate → very strict fuzzy |
| ≥ 0.80 | Substring containment + ≥ 70% word overlap + min length 15 | Low → only near-identical strings |
| < 0.80 | **Never merge** | Insufficient similarity |

**Special rule for ≥ 0.88:** Substring containment is allowed if both strings are ≥ 10 characters and `ratio` ≥ 70.

---

## 7. Pipeline Stages — `assets.py`

### 7.0 Ingestion Assets (Pre-Processing)

#### `ucc_raw_data`

**Type:** Dagster asset (ingestion)  
**Input:** `ucc.csv`  
**Output:** `pd.DataFrame` with columns `[DEBT_STATE, SECP_STATE, SECP_NAME]`

- Reads the CSV file from the configured path
- Drops duplicate rows
- Logs the record count

#### `master_mapping_data`

**Type:** Dagster asset (ingestion)  
**Input:** `Master Mapping.csv`  
**Output:** `pd.DataFrame` with columns `[DEBT_STATE, SECP_STATE, SECP_NAME, SECP_NAME_MAPPED, Firm ID]`

- Reads the master mapping CSV
- Logs the record count

#### `normalized_master_data`

**Type:** Dagster asset (preparation)  
**Depends on:** `master_mapping_data`  
**Output:** Dictionary containing:

| Key | Type | Description |
|-----|------|-------------|
| `raw_lookup` | `Dict[str, str]` | Raw SECP_NAME → SECP_NAME_MAPPED |
| `norm_to_mapped` | `Dict[str, str]` | Normalized name → SECP_NAME_MAPPED |
| `norm_to_raw` | `Dict[str, str]` | Normalized name → original SECP_NAME |
| `normalized_list` | `List[str]` | All unique normalized master names |
| `mapped_list` | `List[str]` | Corresponding SECP_NAME_MAPPED values |
| `raw_list` | `List[str]` | Corresponding original SECP_NAME values |
| `entity_types` | `List[str]` | Entity type for each master name |
| `ambiguous_norms` | `Set[str]` | Normalized names that map to multiple SECP_NAME_MAPPED |
| `norm_to_mapped_candidates` | `Dict[str, List[str]]` | Ambiguous norm → all possible mapped values |

**Ambiguity detection:** If a normalized name maps to multiple different `SECP_NAME_MAPPED` values, it's flagged as ambiguous. Matches to ambiguous names get `REVIEW_HIGH` confidence instead of `AUTO`.

#### `master_embeddings_cache`

**Type:** Dagster asset (ML preparation)  
**Depends on:** `normalized_master_data`  
**Output:** Dictionary containing:

| Key | Type | Description |
|-----|------|-------------|
| `embeddings` | `np.ndarray` | Dense embeddings for all master names (float32) |
| `faiss_index` | `faiss.Index` | FAISS index (IndexFlatIP or IndexHNSWFlat) |
| `ngram_index` | `NGramIndex` | Character trigram inverted index |
| `tfidf_idf_weights` | `Dict[str, float]` | IDF weights for TF-IDF n-gram similarity |
| `soundex` | `List[str]` | Soundex codes for all master names |
| `model` | `SentenceTransformer` | The loaded embedding model |

**Embedding model:** `BAAI/bge-small-en-v1.5` (384-dimensional dense embeddings, optimized for semantic similarity). The model is loaded from a local path to avoid network dependencies.

**FAISS index selection:**
- ≤ 100,000 names → `IndexFlatIP` (exact inner product search, fastest for moderate sizes)
- \> 100,000 names → `IndexHNSWFlat` (approximate search, sub-linear scaling)

**Parallel construction:** N-grams and Soundex codes are computed in parallel using `ThreadPoolExecutor`.

---

### 7.1 Stage 1: Exact Match (Raw) — `exact_match_raw_results`

**Depends on:** `ucc_raw_data`, `normalized_master_data`  
**Strategy:** Direct dictionary lookup of raw `SECP_NAME` against master `raw_lookup`

#### Process

1. For each UCC record, look up `SECP_NAME` in the `raw_lookup` dictionary
2. If found → assign `SECP_NAME_MAPPED`, set `match_score = 1.0`, `confidence = 'AUTO'`
3. If not found → leave unmapped (passed to Stage 2)

#### Output Columns Added

- `SECP_NAME_MAPPED`: The canonical mapped name (or null)
- `match_method`: `'Exact Match (Raw)'` or null
- `match_score`: `1.0` or NaN
- `confidence`: `'AUTO'` or null

**Speed:** O(n) dictionary lookups — near-instantaneous for ~100K records.

---

### 7.2 Stage 2: Exact Match (Normalized) — `exact_match_normalized_results`

**Depends on:** `exact_match_raw_results`, `normalized_master_data`  
**Strategy:** Normalize both UCC names and master names, then match on the normalized form

#### Process

1. Apply `normalize_name()` to every UCC `SECP_NAME` → store as `SECP_NAME_NORMALIZED`
2. For unmatched records, look up `SECP_NAME_NORMALIZED` in `norm_to_mapped`
3. If found AND not ambiguous → `confidence = 'AUTO'`
4. If found AND ambiguous → `confidence = 'REVIEW_HIGH'`, `match_method = 'Exact Match (Normalized - Ambiguous)'`
5. For ambiguous matches, store all possible candidates in `ambiguous_master_candidates`

#### Why This Catches More Matches

Names like:
- `"ABC CORP."` and `"ABC CORPORATION"` → both normalize to `"ABC"`
- `"WELLS FARGO BANK, N.A."` and `"WELLS FARGO BANK NA"` → both normalize to `"WELLS FARGO BANK"`

---

### 7.3 Stage 3: Fuzzy Matching — `fuzzy_match_results`

**Depends on:** `exact_match_normalized_results`, `normalized_master_data`  
**Strategy:** Use `rapidfuzz.process.cdist` with `token_sort_ratio` as the primary scorer, then apply multi-layered false positive prevention

#### Process

1. Extract all unmatched records with non-empty normalized names
2. Pre-compute entity types for all query names
3. Process in batches (`fuzzy_batch_size = 500`):
   a. Compute `cdist` matrix of `token_sort_ratio` scores between batch queries and all master names
   b. Find the best match for each query (highest score)
   c. Apply **four-layer gating** (see below)
   d. Compute `combined_fuzzy_score()` for surviving candidates
   e. Assign confidence based on final score

#### Four-Layer False Positive Gating

Each candidate match must pass ALL four gates:

```
Gate 1: entity_types_compatible(query_type, match_type)
   └─ Reject if BANK matched to MANUFACTURER, etc.

Gate 2: has_shared_meaningful_identifier(query_name, match_name)
   └─ Reject if names share no meaningful words
   └─ e.g., "AMERICAN HERITAGE BANK" vs "AMERICAN EXPRESS" → REJECTED

Gate 3: combined_fuzzy_score(query_name, match_name) ≥ 85.0
   └─ Reject if weighted ensemble score is below minimum
   └─ This applies its own 5 internal guardrails

Gate 4: extract_entity_core() similarity ≥ 70
   └─ Reject if core identifiers are dissimilar
   └─ e.g., core("AMERICAN HERITAGE BANK") = "HERITAGE"
   └─       core("AMERICAN EXPRESS") = "EXPRESS"
   └─       fuzz.ratio("HERITAGE", "EXPRESS") = ~20 → REJECTED
   └─ Exception: Acronym matches bypass this gate
```

#### Confidence Assignment

| Condition | Confidence |
|-----------|------------|
| Ambiguous normalized name | `REVIEW_HIGH` |
| `combined_score ≥ 96` | `AUTO` |
| `combined_score ≥ 85` but `< 96` | `REVIEW_HIGH` |

---

### 7.4 Stage 4: Advanced Multi-Signal Embedding Match — `advanced_embedding_results`

**Depends on:** `fuzzy_match_results`, `normalized_master_data`, `master_embeddings_cache`  
**Strategy:** Combine FAISS semantic embedding search with n-gram, phonetic, fuzzy, and acronym signals for a holistic match score

#### Process

1. Extract remaining unmatched records
2. Pre-compute query metadata in parallel (soundex, entity type, n-gram candidates)
3. For each batch of queries:
   a. Encode queries using SentenceTransformer → 384-dim vectors
   b. FAISS top-k search → retrieve `embed_top_k` nearest neighbors
   c. Merge FAISS candidates with n-gram index candidates
   d. Sort merged candidates by embedding similarity (descending)
   e. For each candidate:
      - Apply entity type compatibility filter
      - Apply token overlap filter (for embed_sim < 0.85)
      - Compute `compute_match_score()` (5 signals)
      - Add soundex and entity type bonuses
      - Track best score, with early termination at 0.97
4. Assign confidence based on composite score

#### Candidate Retrieval Strategy

Candidates come from **two sources**:

1. **FAISS embedding search** — top-k by cosine similarity (semantic)
2. **N-gram index** — records sharing ≥ 2 character trigrams (lexical)

These are merged, deduplicated, and capped at `max_candidates_per_query = 50`. FAISS candidates get priority since they're pre-sorted by relevance.

#### Multi-Signal Score Weights (Stage 4)

| Signal | Weight | Note |
|--------|--------|------|
| Phonetic | 0.08 | Slightly reduced from default |
| N-gram | 0.12 | TF-IDF weighted |
| Fuzzy | 0.20 | `combined_fuzzy_score` ensemble |
| Embedding | 0.50 | Increased — primary driver |
| Acronym | 0.10 | Binary bonus |

#### Confidence Assignment

| Composite Score | Confidence |
|----------------|------------|
| ≥ 0.95 | `AUTO` |
| ≥ 0.90 | `REVIEW_HIGH` |
| ≥ 0.85 | `REVIEW_MEDIUM` |
| < 0.85 | Not matched (passed to Stage 5) |

#### CPU Optimization

- Environment variables `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` set to `cpu_threads`
- Batch encoding with `batch_size=128` for CPU
- Early termination at score ≥ 0.97
- Periodic `gc.collect()` every 500 records

---

### 7.5 Stage 5: Clustering — `clustered_results`

**Depends on:** `advanced_embedding_results`, `master_embeddings_cache`  
**Strategy:** Group remaining unmatched records into clusters of similar names using embedding similarity + fuzzy guard

#### Process

1. Extract all records still unmatched after Stages 1-4
2. Encode all unmatched names with SentenceTransformer
3. Build FAISS index over unmatched embeddings
4. For each record, find `cluster_k_neighbors = 15` nearest neighbors
5. For each neighbor pair with `embedding_similarity ≥ cluster_sim_threshold (0.88)`:
   - Compute `fuzz.token_sort_ratio`
   - If `≥ cluster_fuzzy_guard (60)` → merge into same cluster via UnionFind
6. Assign cluster IDs and compute cluster labels

#### Cluster Label Selection

For each cluster:
- **Singleton clusters:** Use the single name
- **Large clusters (> 50 members):** Use the longest name (most descriptive)
- **Medium clusters (2-50 members):** Compute the **medoid** — the name whose embedding is closest to the cluster centroid (most representative)

#### Output Columns Added

- `cluster_id`: Unique cluster identifier (e.g., `C000042`)
- `cluster_label`: Representative name for the cluster
- `cluster_size`: Number of records in the cluster
- `match_method`: `'Unmatched (Clustered)'`
- `confidence`: `'NEW_ENTITY'`

---

### 7.6 Stage 6: Hierarchical Cluster Merging — `merged_cluster_results`

**Depends on:** `clustered_results`, `master_embeddings_cache`  
**Strategy:** Iteratively merge clusters whose labels are semantically and lexically similar

#### Process (Up to 3 Iterations)

1. Collect unique cluster labels
2. Encode all cluster labels with SentenceTransformer
3. Build FAISS index over label embeddings
4. For each label, find `cluster_merge_k = 10` nearest neighbors
5. For each pair, call `should_merge_clusters(label_a, label_b, embedding_sim)` (see §6.16)
6. If merge approved AND combined size ≤ `cluster_max_size (500)` → UnionFind merge
7. Reassign cluster IDs and select best labels for merged clusters
8. Stop early if no merges were made or cluster count stabilized

#### Merge Label Selection Strategy

When clusters merge, the best label is chosen by scoring:
1. **Best (priority 0):** 2-5 words, 20-80 characters
2. **Good (priority 1):** Multi-word but outside ideal range
3. **Less preferred (priority 2):** Single word

Very long names (> 100 chars) are filtered out as they're typically truncated or less representative.

#### Safety Guards

- **Maximum cluster size (500):** Prevents formation of mega-clusters that would be unhelpful
- **No threshold relaxation:** Unlike some implementations, the merge threshold stays constant across iterations
- **Size tracking:** Merged sizes are tracked to enforce the cap even across multi-hop merges

---

### 7.7 Stage 7: Suggested Mappings — `suggested_mapping_results`

**Depends on:** `merged_cluster_results`, `normalized_master_data`, `master_embeddings_cache`  
**Strategy:** For each unmatched cluster, suggest the closest master mapping as a candidate

#### Process

1. Extract clusters with `confidence = 'NEW_ENTITY'`
2. Normalize cluster labels
3. Encode normalized labels with SentenceTransformer
4. FAISS search against master embeddings → top-3 matches per cluster
5. Compute combined score: `0.6 × embedding_sim + 0.4 × fuzzy_token_set_ratio`
6. Apply substring containment bonus (+0.10)
7. Assign suggestion confidence tiers

#### Suggestion Confidence Tiers

| Combined Score | Tier | Description |
|---------------|------|-------------|
| ≥ 0.80 | `SUGGEST_HIGH` | Very likely the same entity |
| ≥ 0.70 | `SUGGEST_MEDIUM` | Probable match, needs verification |
| ≥ 0.60 | `SUGGEST_LOW` | Possible match, lower confidence |
| < 0.60 | No suggestion | Too dissimilar to suggest |

#### Output Columns Added

- `suggested_mapping`: Best matching master entity name
- `suggestion_score`: Combined score
- `suggestion_confidence`: `SUGGEST_HIGH` / `SUGGEST_MEDIUM` / `SUGGEST_LOW`
- `alternative_suggestions`: Second and third best matches (semicolon-separated)

---

### 7.8 Output Generation — `pipeline_outputs`

**Depends on:** `suggested_mapping_results`  
**Output:** `MaterializeResult` with metadata + 5 CSV/TXT files

#### Generated Files

| File | Contents | Sorting |
|------|----------|---------|
| `auto_mapped_{timestamp}.csv` | Confidence = AUTO | By match_method, SECP_NAME |
| `review_needed_{timestamp}.csv` | Confidence = REVIEW_HIGH/MEDIUM/LOW | By confidence, match_score desc |
| `new_entities_{timestamp}.csv` | Confidence = NEW_ENTITY | By suggestion_confidence, cluster_size desc |
| `full_results_{timestamp}.csv` | All records | Original order |
| `summary_{timestamp}.txt` | Statistics text report | N/A |

#### Summary Report Contents

- Total UCC records
- Auto-mapped count
- Review needed count
- New entities count
- Breakdown by match method
- Breakdown by confidence level
- Clustering statistics (cluster count, average size)

#### Dagster Metadata

The `MaterializeResult` exposes metrics in the Dagster UI:
- `total_records`, `auto_mapped_count`, `auto_mapped_pct`
- `review_needed_count`, `new_entities_count`
- `output_dir`, `timestamp`

---

## 8. Dagster Wiring — `definitions.py`

```python
from dagster import Definitions, load_assets_from_modules
from . import assets
from .resources import PipelineConfig

all_assets = load_assets_from_modules([assets])
defs = Definitions(
    assets=all_assets,
    resources={"config": PipelineConfig()},
)
```

- `load_assets_from_modules` discovers all `@asset`-decorated functions in `assets.py`
- `PipelineConfig()` is injected as a shared resource available to all assets via the `config` parameter
- Dagster automatically resolves the dependency graph from function signatures

---

## 9. False Positive Prevention System

This section consolidates the multi-layered defense against false positive matches — the most critical aspect of pipeline accuracy.

### 9.1 The Problem

Semantic embeddings and fuzzy string matchers are prone to matching entities that share common words but are completely different entities. For example:

| Query | False Match | Why It Happened |
|-------|-------------|-----------------|
| AMERICAN HERITAGE NATIONAL BANK | AMERICAN EXPRESS | Both start with "AMERICAN" |
| FIRST NATIONAL BANK | FIRST CHOICE BANK | Both start with "FIRST" |
| US BANK | US STEEL | Both start with "US" |
| CITIZENS BANK | CITIZENS INSURANCE | Both contain "CITIZENS" |

### 9.2 Defense Layers

```
Layer 1: Entity Type Gating
   └─ BANK cannot match MANUFACTURER
   └─ GOVERNMENT cannot match INVESTMENT
   └─ Only compatible pairs (BANK↔LEGAL, INVESTMENT↔LEGAL) allowed

Layer 2: Two-Tier Word Classification
   └─ Entity type words (BANK, TRUST, etc.) ALWAYS stripped before comparison
   └─ Generic prefixes (AMERICAN, FIRST, etc.) stripped only if meaningful words remain
   └─ Prevents both over-stripping (US BANK → "") and under-stripping (AMERICAN → match)

Layer 3: Meaningful Identifier Overlap
   └─ Names must share meaningful words to be considered
   └─ Bidirectional overlap > 50% for non-subset pairs
   └─ Subset matching only when unique identifiers present

Layer 4: Entity Core Comparison
   └─ Extract the single most distinctive word from each name
   └─ Cores must be ≥ 70% similar (or acronym match)

Layer 5: Combined Fuzzy Score Floor
   └─ Weighted ensemble must score ≥ 85/100
   └─ Internal guardrails: ratio ≥ 50, token overlap, entity type, score ceiling

Layer 6 (Clustering): Conservative Merge Function
   └─ Requires BOTH high embedding AND high fuzzy similarity
   └─ Tiered thresholds prevent runaway merges
```

### 9.3 Edge Cases Handled

| Scenario | Expected Behavior | Mechanism |
|----------|-------------------|-----------|
| "FIRST BANK" should NOT match "FIRST AMERICAN BANK" | ✅ Rejected | words1={first}, words2={first,american}; subset but 'first' is generic-only |
| "CITIZENS BANK" SHOULD match "FIRST CITIZENS BANK" | ✅ Matched | Both produce {first,citizens} after processing |
| "AMERICAN HERITAGE BANK" should NOT match "AMERICAN EXPRESS" | ✅ Rejected | words1={heritage}, words2={express}; no overlap |
| "US BANK" should NOT match "US STEEL" | ✅ Rejected | Entity type gating (BANK vs MANUFACTURER) |
| "WELLS FARGO BANK" SHOULD match "WELLS FARGO FINANCIAL" | ✅ Matched | words1={wells,fargo}, words2={wells,fargo}; identical |

---

## 10. Test Suite — `test_normalize.py`

67 unit tests organized into 10 test classes:

| Test Class | Count | Tests |
|------------|-------|-------|
| `TestNormalizeName` | 25 | Legal suffix removal, role stripping, abbreviation expansion, whitespace handling, common word removal, edge cases |
| `TestSoundex` | 5 | Standard encoding, H/W separator rule, empty input, single char |
| `TestDetectEntityType` | 5 | BANK, GOVERNMENT, INVESTMENT, MANUFACTURER, OTHER classification |
| `TestNGrams` | 4 | Trigram generation, empty input, short strings |
| `TestEntityTypesCompatible` | 4 | Same type, OTHER wildcard, compatible pairs, incompatible pairs |
| `TestEntityHelpers` | 5 | `extract_entity_core` for various name patterns |
| `TestAcronymMatch` | 3 | IBM→INTERNATIONAL BUSINESS MACHINES, non-matches |
| `TestMinTokenOverlap` | 2 | Overlapping and non-overlapping token sets |
| `TestCombinedFuzzyScore` | 3 | Identical names, completely different names, similar names |
| `TestFalsePositivePrevention` | 11 | All edge cases from §9.3 including first_american_bank scenarios |

---

## 11. Output Files & Reporting

### 11.1 Confidence Level Taxonomy

| Level | Meaning | Action Required |
|-------|---------|----------------|
| `AUTO` | High-confidence match | No review needed — auto-accept |
| `REVIEW_HIGH` | Likely correct but uncertain | Human review recommended |
| `REVIEW_MEDIUM` | Possible match | Human review required |
| `REVIEW_LOW` | Weak match | Human review required |
| `NEW_ENTITY` | No match found, clustered | Analyst determines if truly new |
| `SUGGEST_HIGH` | Strong suggestion for new entity | Consider accepting suggestion |
| `SUGGEST_MEDIUM` | Moderate suggestion | Evaluate suggestion |
| `SUGGEST_LOW` | Weak suggestion | Use as starting point only |

### 11.2 Match Method Labels

| Label | Stage | Description |
|-------|-------|-------------|
| `Exact Match (Raw)` | 1 | Verbatim string match |
| `Exact Match (Normalized)` | 2 | Match after normalization |
| `Exact Match (Normalized - Ambiguous)` | 2 | Normalized match with multiple possible mappings |
| `Fuzzy Match` | 3 | Fuzzy string matching |
| `Fuzzy Match (Ambiguous)` | 3 | Fuzzy match with multiple candidates |
| `Advanced Match` | 4 | Multi-signal embedding-based match |
| `Unmatched (Clustered)` | 5-6 | Grouped with similar unmatched records |

---

## 12. Performance & Optimization

### 12.1 CPU-Specific Optimizations

- **Thread control:** `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` set at module level to prevent oversubscription
- **FAISS on CPU:** Uses `IndexFlatIP` for moderate datasets (faster than HNSW on CPU for < 100K records)
- **Batch encoding:** SentenceTransformer batch_size=128 optimized for CPU memory bandwidth
- **Contiguous arrays:** `np.ascontiguousarray` ensures optimal memory layout for FAISS

### 12.2 Memory Management

- `gc.collect()` called after every 500 records in Stage 4
- FAISS indices deleted (`del index`) after use in Stages 5-7
- Model cached in `master_embeddings_cache` — loaded once, reused across stages
- Embeddings stored as `float32` (not `float64`) for 50% memory savings

### 12.3 Algorithmic Optimizations

- **Early termination:** In Stage 4, stop searching once score ≥ 0.97
- **Candidate capping:** Maximum 50 candidates per query (FAISS + n-gram combined)
- **Sorted candidate processing:** Candidates sorted by embedding similarity descending — worst candidates never reach expensive scoring
- **Parallel metadata computation:** Soundex, entity types, n-gram candidates computed in ThreadPool
- **Pre-sorted COMMON_WORDS:** Sorted by length descending to prevent partial removal issues

### 12.4 Scalability Thresholds

| Dataset Size | FAISS Index | Approximate Runtime |
|-------------|-------------|---------------------|
| < 100K | IndexFlatIP | Minutes |
| 100K - 200K | IndexFlatIP | Tens of minutes |
| > 200K | IndexHNSWFlat | Sub-linear scaling |

---

## 13. Appendix: Complete Function Reference

### `constants.py`

| Symbol | Type | Description |
|--------|------|-------------|
| `LEGAL_SUFFIXES` | `List[re.Pattern]` | 15 compiled regexes for legal entity suffixes |
| `ROLE_TRIGGERS` | `List[str]` | 16 role phrase patterns |
| `ROLE_RE` | `re.Pattern` | Compiled regex combining all role triggers |
| `COMMON_WORDS` | `List[str]` | 25 generic words sorted by length descending |
| `ABBREVIATIONS` | `Dict[str, str]` | 18 abbreviation → full form mappings |
| `ENTITY_TYPE_PATTERNS` | `Dict[str, List[str]]` | 5 entity categories with keyword lists |

### `utils.py`

| Function | Signature | Returns |
|----------|-----------|---------|
| `normalize_name` | `(name: str) → str` | Canonicalized name |
| `soundex` | `(name: str) → str` | 4-character Soundex code |
| `detect_entity_type` | `(name: str) → str` | Entity type string |
| `generate_ngrams` | `(name: str, n: int) → Set[str]` | Set of character n-grams |
| `ngram_similarity` | `(name1: str, name2: str, n: int) → float` | Jaccard similarity [0, 1] |
| `tfidf_ngram_similarity` | `(name1, name2, idf_weights, n) → float` | TF-IDF cosine similarity [0, 1] |
| `extract_acronym` | `(name: str) → str` | Acronym string |
| `is_acronym_match` | `(name1: str, name2: str) → bool` | Whether one is acronym of other |
| `compute_match_score` | `(query, master, embed_sim, ...) → Tuple[float, Dict]` | (composite_score, signal_breakdown) |
| `get_meaningful_words` | `(name: str) → Set[str]` | Set of meaningful lowercase words |
| `combined_fuzzy_score` | `(name1: str, name2: str) → float` | Weighted fuzzy score [0, 100] |
| `has_shared_meaningful_identifier` | `(name1: str, name2: str) → bool` | Whether names share identifiers |
| `extract_entity_core` | `(name: str) → str` | Core identifying word(s) |
| `min_token_overlap` | `(name1, name2, min_overlap) → bool` | Whether token overlap meets minimum |
| `entity_types_compatible` | `(type1: str, type2: str) → bool` | Whether types can match |
| `should_merge_clusters` | `(label_a, label_b, embedding_sim) → bool` | Whether clusters should merge |

### `data_structures.py`

| Class | Key Methods |
|-------|-------------|
| `NGramIndex` | `add()`, `add_precomputed()`, `query()` |
| `TFIDFNGramIndex` | `add()`, `freeze()`, `similarity()` |
| `UnionFind` | `find()`, `union()`, `get_clusters()` |

### `assets.py`

| Asset | Stage | Group | Kind |
|-------|-------|-------|------|
| `ucc_raw_data` | Ingestion | ingestion | io |
| `master_mapping_data` | Ingestion | ingestion | io |
| `normalized_master_data` | Preparation | ingestion | python |
| `master_embeddings_cache` | Preparation | ingestion | ml |
| `exact_match_raw_results` | 1 | matching | python |
| `exact_match_normalized_results` | 2 | matching | python |
| `fuzzy_match_results` | 3 | matching | python |
| `advanced_embedding_results` | 4 | matching | ml |
| `clustered_results` | 5 | clustering | ml |
| `merged_cluster_results` | 6 | clustering | ml |
| `suggested_mapping_results` | 7 | clustering | ml |
| `pipeline_outputs` | Output | reporting | io |

---

*Generated for the UCC Retrieval and Mapping Pipeline — Dagster-based entity resolution system.*
