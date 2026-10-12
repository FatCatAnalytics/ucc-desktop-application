# =============================================================================
# UTILITIES
# =============================================================================
import re
import math
from functools import lru_cache
from typing import List, Set, Dict, Tuple, Optional
from difflib import SequenceMatcher
from collections import Counter
import pandas as pd
from rapidfuzz import fuzz
from .constants import (
    LEGAL_SUFFIXES,
    ROLE_RE,
    COMMON_WORDS,
    ABBREVIATIONS,
    ENTITY_TYPE_PATTERNS,
    PARENTHETICAL_RE,
    DIVISION_OF_RE,
    RECEIVER_OF_RE,
    PREDECESSOR_TAIL_RE,
    CAPACITY_TAIL_RE,
    CHARTER_MID_RE,
    BRANCH_TAIL_RE,
)

# =============================================================================
# MODULE-LEVEL CONSTANTS
# =============================================================================
# Defining these once (frozensets) avoids re-building Python sets and re-running
# set unions on every call from hot loops in Stages 3-7.

# Tier 1: Entity type words - ALWAYS excluded from meaningful word sets.
ENTITY_TYPE_WORDS = frozenset({
    'bank', 'banking', 'trust', 'savings', 'credit', 'union',
    'company', 'corporation', 'corp', 'inc', 'llc', 'ltd', 'limited',
    'the', 'of', 'and', 'a', 'an',
    'financial', 'services', 'group', 'holdings', 'capital', 'partners',
    'investment', 'investments', 'management', 'fund', 'funds',
    'advisory', 'advisors', 'asset', 'assets', 'lending', 'loan',
    'equipment', 'finance', 'mortgage', 'solutions', 'associates',
    'na', 'n.a.', 'fsb', 'ssb',
})

# Tier 2: Generic prefixes - excluded only when truly-unique words exist.
GENERIC_PREFIXES = frozenset({
    'american', 'united', 'us', 'usa', 'new', 'old', 'central',
    'western', 'eastern', 'northern', 'southern', 'north', 'south',
    'east', 'west', 'pacific', 'atlantic', 'continental', 'general',
    'royal', 'premier', 'global', 'international', 'universal',
    'great', 'greater', 'community', 'regional', 'state', 'metro',
    'metropolitan', 'citizens', 'peoples', 'security', 'commerce',
    'first', 'second', 'third', 'national', 'federal',
})

ALL_GENERIC = ENTITY_TYPE_WORDS | GENERIC_PREFIXES

# Upper-case variants used by combined_fuzzy_score / extract_entity_core.
SKIPPABLE_PREFIXES_UPPER = frozenset({
    'THE', 'A', 'AN',
    'AMERICAN', 'UNITED', 'US', 'USA', 'NEW', 'OLD', 'CENTRAL',
    'WESTERN', 'EASTERN', 'NORTHERN', 'SOUTHERN', 'NORTH', 'SOUTH',
    'EAST', 'WEST', 'PACIFIC', 'ATLANTIC', 'CONTINENTAL', 'GENERAL',
    'ROYAL', 'PREMIER', 'GLOBAL', 'INTERNATIONAL', 'UNIVERSAL',
    'GREAT', 'GREATER', 'COMMUNITY', 'REGIONAL', 'STATE', 'METRO',
    'METROPOLITAN', 'CITIZENS', 'PEOPLES', 'SECURITY', 'COMMERCE',
    'FIRST', 'SECOND', 'THIRD',
})
GENERIC_WORDS_UPPER = frozenset({
    'BANK', 'BANKING', 'CORP', 'CORPORATION', 'INC', 'LLC', 'LTD', 'LIMITED',
    'COMPANY', 'CO', 'TRUST', 'FINANCIAL', 'SERVICES', 'GROUP',
    'EQUIPMENT', 'FINANCE', 'CAPITAL', 'INVESTMENT', 'FUND',
    'NATIONAL', 'FEDERAL', 'STATE', 'COMMUNITY', 'COMMERCIAL',
    'SAVINGS', 'CREDIT', 'UNION', 'LENDING', 'LOAN', 'MORTGAGE',
    'HOLDINGS', 'PARTNERS', 'ASSOCIATES', 'SOLUTIONS',
})
ALL_GENERIC_UPPER = SKIPPABLE_PREFIXES_UPPER | GENERIC_WORDS_UPPER

ACRONYM_SKIP_WORDS = frozenset({'THE', 'OF', 'AND', 'A', 'AN', 'IN', 'FOR', 'TO', 'BY'})

# Cache size tuned for typical UCC dataset (~200K rows, heavy dedup expected)
_CACHE_SIZE = 200_000


@lru_cache(maxsize=_CACHE_SIZE)
def pre_clean_name(name: str) -> str:
    """Reduce a filed secured-party string to the one lender it names.

    Order matters: pick the lender out of two-name phrases first, then cut the
    capacity / predecessor / branch tails, then drop charter words. Each step
    is a no-op when its pattern is absent. Examples:

        STARION BANK (LENDER AND LETTER OF CREDIT ISSUER)        -> STARION BANK
        KEY EQUIPMENT FINANCE, A DIVISION OF KEYBANK NA          -> KEYBANK
        FDIC AS RECEIVER OF FIRST REPUBLIC BANK, SAN FRANCISCO   -> FIRST REPUBLIC BANK
        TD BANK, NA, AS SUCCESSOR BY MERGER TO MERCANTILE BANK   -> TD BANK
        WELLS FARGO BANK, N.A., AS ADMINISTRIVE AGENT            -> WELLS FARGO BANK
        PLAINSCAPITAL BANK, A TEXAS STATE BANKING ASSOCIATION    -> PLAINSCAPITAL BANK
        PARK BANK- BROOKFIELD                                    -> PARK BANK
    """
    s = str(name).upper().strip()
    if not s:
        return ""
    s = PARENTHETICAL_RE.sub(" ", s)
    m = DIVISION_OF_RE.match(s)
    if m:
        s = s[m.end():]
    m = RECEIVER_OF_RE.match(s)
    if m:
        s = s[m.end():].split(",", 1)[0]
    s = PREDECESSOR_TAIL_RE.sub("", s)
    s = CAPACITY_TAIL_RE.sub("", s)
    s = BRANCH_TAIL_RE.sub(r"\1", s)
    s = CHARTER_MID_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip(" ,;:-")
    # Never clean a name down to nothing - fall back to the raw string.
    return s if re.search(r"[A-Z0-9]", s) else str(name).upper().strip()


def normalize_name(name: str) -> str:
    """Normalize SECP_NAME for better matching."""
    if pd.isna(name) or not str(name).strip():
        return ""
    
    name = pre_clean_name(str(name))
    if not name:
        return ""

    # Strip role qualifiers first (e.g. "as trustee", "as agent")
    m = ROLE_RE.search(name)
    if m:
        name = name[:m.start()].rstrip(" ,;:-")
    
    # Remove legal suffixes (INC, LLC, etc.) — but NOT entity-type words like BANK
    for suffix in LEGAL_SUFFIXES:
        name = re.sub(suffix, '', name)
    
    # Remove common noise words — apply AFTER suffix stripping
    # Process longer phrases first so they match before their sub-words
    for word in COMMON_WORDS:
        name = re.sub(word, ' ', name)
    
    # Expand known abbreviations
    for abbr, full in ABBREVIATIONS.items():
        name = re.sub(abbr, full, name)
    
    # Strip non-alphanumeric (keep spaces), collapse whitespace
    name = re.sub(r'[^\w\s]', ' ', name)
    name = re.sub(r'\s+', ' ', name)
    name = name.strip()
    
    return name

@lru_cache(maxsize=_CACHE_SIZE)
def soundex(name: str) -> str:
    """Generate Soundex code for phonetic matching (ANSI/AIM standard)."""
    name = name.upper()
    name = re.sub(r'[^A-Z]', '', name)
    if not name:
        return ''

    first_letter = name[0]
    code_map = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6',
    }
    # Vowels and H/W/Y are ignored (mapped to '')
    # Adjacent same-code letters are collapsed

    prev_code = code_map.get(first_letter, '')
    coded = first_letter
    for char in name[1:]:
        cur_code = code_map.get(char, '')
        if cur_code and cur_code != prev_code:
            coded += cur_code
        # Per spec, H and W don't reset the previous code group
        if char not in ('H', 'W'):
            prev_code = cur_code

    coded = coded[:4].ljust(4, '0')
    return coded

@lru_cache(maxsize=_CACHE_SIZE)
def detect_entity_type(name: str) -> str:
    """Detect the type of entity based on name patterns."""
    name_upper = name.upper()
    
    for entity_type, patterns in ENTITY_TYPE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, name_upper):
                return entity_type
    
    return 'OTHER'

def generate_ngrams(text: str, n: int = 3) -> set:
    """Generate character n-grams for blocking.

    Returns a fresh ``set`` so callers can safely mutate the result. For
    cacheable reuse use :func:`generate_ngrams_frozen`.
    """
    return set(generate_ngrams_frozen(text, n))


@lru_cache(maxsize=_CACHE_SIZE)
def generate_ngrams_frozen(text: str, n: int = 3) -> frozenset:
    """Cached character-n-gram generator returning an immutable frozenset."""
    text = text.lower().replace(' ', '')
    if len(text) < n:
        return frozenset({text})
    return frozenset(text[i:i + n] for i in range(len(text) - n + 1))

def ngram_similarity(text1: str, text2: str, n: int = 3) -> float:
    """Jaccard similarity of n-grams (unweighted baseline)."""
    ngrams1 = generate_ngrams(text1, n)
    ngrams2 = generate_ngrams(text2, n)
    
    if not ngrams1 or not ngrams2:
        return 0.0
    
    intersection = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)
    
    return intersection / union if union > 0 else 0.0


def build_tfidf_vec(
    text: str,
    idf_weights: Dict[str, float],
    n: int = 3,
) -> Tuple[Dict[str, float], float]:
    """Build a sparse TF-IDF vector and its L2 norm for a single string.

    Used as a one-shot pre-compute so per-candidate scoring avoids re-tokenising
    the query. Returns ``({ngram: weight}, l2_norm)``.
    """
    ng = generate_ngrams_frozen(text, n)
    if not ng:
        return {}, 0.0
    tf = 1.0 / len(ng)
    vec = {g: tf * idf_weights.get(g, 1.0) for g in ng}
    norm = math.sqrt(sum(v * v for v in vec.values()))
    return vec, norm


def tfidf_cosine_precomputed(
    query_vec: Dict[str, float],
    query_norm: float,
    cand_vec: Dict[str, float],
    cand_norm: float,
) -> float:
    """Cosine similarity from two pre-built TF-IDF sparse vectors."""
    if query_norm == 0.0 or cand_norm == 0.0:
        return 0.0
    if len(query_vec) > len(cand_vec):
        query_vec, cand_vec = cand_vec, query_vec
    dot = 0.0
    cand_get = cand_vec.get
    for k, v in query_vec.items():
        cv = cand_get(k)
        if cv is not None:
            dot += v * cv
    return dot / (query_norm * cand_norm)


def tfidf_ngram_similarity(
    text1: str,
    text2: str,
    idf_weights: Optional[Dict[str, float]] = None,
    n: int = 3,
) -> float:
    """
    TF-IDF weighted cosine similarity of character n-gram vectors.

    If *idf_weights* is ``None`` falls back to plain Jaccard similarity so
    it can be used as a drop-in replacement before an IDF table is built.

    Parameters
    ----------
    text1, text2 : str
        The two strings to compare.
    idf_weights : dict, optional
        Mapping ``{ngram: idf_value}``.  Obtain from
        ``TFIDFNGramIndex._idf`` after calling ``freeze()``.
    n : int
        N-gram size (default 3).
    """
    if idf_weights is None:
        return ngram_similarity(text1, text2, n)

    ng1 = generate_ngrams(text1, n)
    ng2 = generate_ngrams(text2, n)
    if not ng1 or not ng2:
        return 0.0

    def _vec(ngrams: set) -> Dict[str, float]:
        tf = 1.0 / len(ngrams)
        return {ng: tf * idf_weights.get(ng, 1.0) for ng in ngrams}

    v1 = _vec(ng1)
    v2 = _vec(ng2)
    shared = set(v1) & set(v2)
    if not shared:
        return 0.0

    dot = sum(v1[k] * v2[k] for k in shared)
    n1 = math.sqrt(sum(v * v for v in v1.values()))
    n2 = math.sqrt(sum(v * v for v in v2.values()))
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)

@lru_cache(maxsize=_CACHE_SIZE)
def extract_acronym(name: str) -> str:
    """Extract potential acronym from name."""
    words = name.upper().split()
    return ''.join(w[0] for w in words if w and w not in ACRONYM_SKIP_WORDS)

def is_acronym_match(short_name: str, long_name: str) -> bool:
    """Check if short_name could be an acronym of long_name."""
    short = short_name.upper().replace(' ', '').replace('.', '')
    acronym = extract_acronym(long_name)
    
    if len(short) <= 5 and len(short) >= 2:
        return short == acronym or short in acronym or acronym.startswith(short)
    return False

def compute_match_score_fast(
        name1: str,
        name2: str,
        embedding_sim: float,
        *,
        soundex1: str,
        soundex2: str,
        query_tfidf_vec: Dict[str, float],
        query_tfidf_norm: float,
        cand_tfidf_vec: Dict[str, float],
        cand_tfidf_norm: float,
        phonetic_weight: float = 0.1,
        ngram_weight: float = 0.15,
        fuzzy_weight: float = 0.25,
        embed_weight: float = 0.4,
        acronym_weight: float = 0.1,
) -> float:
    """Fast multi-signal score using fully pre-computed signals.

    Avoids re-running soundex / generating n-grams in the hot loop. Returns
    only the total score; callers needing the breakdown should use
    :func:`compute_match_score`.
    """
    phonetic_score = 1.0 if soundex1 == soundex2 else 0.0
    ngram_score = tfidf_cosine_precomputed(
        query_tfidf_vec, query_tfidf_norm, cand_tfidf_vec, cand_tfidf_norm
    )

    # rapidfuzz scorers are C-implemented and individually cheap; keep all four
    # to preserve scoring behaviour from compute_match_score.
    fuzzy_score = max(
        fuzz.ratio(name1, name2),
        fuzz.partial_ratio(name1, name2),
        fuzz.token_sort_ratio(name1, name2),
        fuzz.token_set_ratio(name1, name2),
    ) / 100.0

    acronym_score = 1.0 if (
        is_acronym_match(name1, name2) or is_acronym_match(name2, name1)
    ) else 0.0

    return (
        phonetic_weight * phonetic_score
        + ngram_weight * ngram_score
        + fuzzy_weight * fuzzy_score
        + embed_weight * embedding_sim
        + acronym_weight * acronym_score
    )


def compute_match_score(
        name1: str,
        name2: str,
        embedding_sim: float,
        phonetic_weight: float = 0.1,
        ngram_weight: float = 0.15,
        fuzzy_weight: float = 0.25,
        embed_weight: float = 0.4,
        acronym_weight: float = 0.1,
        idf_weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, dict]:
    """Compute weighted match score using multiple signals.

    Parameters
    ----------
    idf_weights : dict, optional
        If provided, n-gram similarity uses TF-IDF cosine instead of
        plain Jaccard.  Pass ``TFIDFNGramIndex._idf`` here.
    """
    
    soundex1, soundex2 = soundex(name1), soundex(name2)
    phonetic_score = 1.0 if soundex1 == soundex2 else 0.0
    
    ngram_score = tfidf_ngram_similarity(name1, name2, idf_weights=idf_weights, n=3)
    
    fuzzy_scores = [
        fuzz.ratio(name1, name2) / 100.0,
        fuzz.partial_ratio(name1, name2) / 100.0,
        fuzz.token_sort_ratio(name1, name2) / 100.0,
        fuzz.token_set_ratio(name1, name2) / 100.0,
    ]
    fuzzy_score = max(fuzzy_scores)
    
    acronym_score = 1.0 if (is_acronym_match(name1, name2) or is_acronym_match(name2, name1)) else 0.0
    
    total_score = (
            phonetic_weight * phonetic_score +
            ngram_weight * ngram_score +
            fuzzy_weight * fuzzy_score +
            embed_weight * embedding_sim +
            acronym_weight * acronym_score
    )
    
    breakdown = {
        'phonetic': phonetic_score,
        'ngram': ngram_score,
        'fuzzy': fuzzy_score,
        'embedding': embedding_sim,
        'acronym': acronym_score,
        'total': total_score
    }
    
    return total_score, breakdown

@lru_cache(maxsize=_CACHE_SIZE)
def _get_meaningful_words_frozen(name: str) -> frozenset:
    """Cached implementation returning an immutable frozenset.

    Hot-path callers can use this directly; ``get_meaningful_words`` returns
    a fresh mutable ``set`` for backward-compat.
    """
    word_list = name.lower().split()

    # Remove entity type words (always)
    words_without_entity_types = [
        w for w in word_list if w not in ENTITY_TYPE_WORDS and len(w) > 1
    ]

    # Check if there are meaningful words after removing BOTH tiers
    truly_meaningful = [
        w for w in word_list if w not in ALL_GENERIC and len(w) > 1
    ]

    if truly_meaningful:
        return frozenset(truly_meaningful)

    if words_without_entity_types:
        return frozenset(words_without_entity_types)

    return frozenset()


def get_meaningful_words(name: str) -> set:
    """
    Extract meaningful distinguishing words from a name.

    Two-tier exclusion system:
    1. Entity type suffixes (BANK, CORP, INC) - ALWAYS excluded
    2. Generic prefixes (FIRST, AMERICAN) - excluded only if meaningful words exist

    Examples:
    - "STEARNS BANK" -> {'stearns'}
    - "AMERICAN HERITAGE BANK" -> {'heritage'}
    - "US BANK" -> {'us'} (bank excluded, us kept as fallback)
    - "FIRST BANK" -> {'first'} (both excluded but kept as fallback)
    - "FIRST CHOICE BANK" -> {'choice'}
    - "FIRST AMERICAN BANK" -> {'first', 'american'} (fallback)
    """
    return set(_get_meaningful_words_frozen(name))

@lru_cache(maxsize=_CACHE_SIZE)
def get_primary_identifier(name_upper: str) -> Optional[str]:
    """Return the first meaningful identifier word in an upper-cased name.

    Cached on the upper-cased input. The caller is responsible for passing
    ``name.upper().strip()``.
    """
    words = name_upper.split()
    if not words:
        return None

    # Find first meaningful word
    for word in words:
        if word not in ALL_GENERIC_UPPER and len(word) > 2:
            return word

    # No meaningful word - return first non-article word
    for word in words:
        if word not in {'THE', 'A', 'AN'} and len(word) > 2:
            return word
    return None


@lru_cache(maxsize=_CACHE_SIZE)
def combined_fuzzy_score(name1: str, name2: str) -> float:
    """
    Compute weighted ensemble of fuzzy scores with safeguards against
    common-word-only matches (e.g. "Heritage Bank" matching "Old National Bank").

    Cached on (name1, name2) - heavy dedup is expected on UCC data.
    """

    ratio = fuzz.ratio(name1, name2)
    partial = fuzz.partial_ratio(name1, name2)
    token_sort = fuzz.token_sort_ratio(name1, name2)
    token_set = fuzz.token_set_ratio(name1, name2)

    # Weighted combination heavily favoring full string similarity (token_sort & ratio)
    # token_set is dangerous because it gives 100% for subset matches
    score = 0.3 * ratio + 0.1 * partial + 0.5 * token_sort + 0.1 * token_set

    # GUARDRAIL 1: Meaningful Word Overlap
    words1 = _get_meaningful_words_frozen(name1)
    words2 = _get_meaningful_words_frozen(name2)

    if words1 and words2 and not (words1 & words2):
        score *= 0.1  # Severe penalty - almost certain false positive
    elif words1 and words2:
        overlap_ratio = len(words1 & words2) / len(words1 | words2)
        if overlap_ratio < 0.3:
            score *= 0.5

    # GUARDRAIL 2: Entity Name Mismatch Detection
    name1_clean = name1.upper().strip()
    name2_clean = name2.upper().strip()
    primary1 = get_primary_identifier(name1_clean)
    primary2 = get_primary_identifier(name2_clean)

    # If both have clear primary identifiers that are completely different
    if primary1 and primary2 and primary1 != primary2:
        # Check if they're similar enough (e.g., abbreviations)
        identifier_sim = fuzz.ratio(primary1, primary2)
        if identifier_sim < 60:  # Primary identifiers are very different
            # Special cases: acronyms, abbreviations
            if not (is_acronym_match(primary1, primary2) or is_acronym_match(primary2, primary1)):
                score *= 0.2  # Heavy penalty for different primary identifiers

        # GUARDRAIL 3: Length mismatch penalty (existing)
    len1, len2 = len(name1), len(name2)
    if len1 > 0 and len2 > 0:
        length_ratio = min(len1, len2) / max(len1, len2)
        if length_ratio < 0.5:
            score *= 0.7

    # GUARDRAIL 4: Very short strings (existing)
    if min(len1, len2) <= 4 and ratio < 80:
        score *= 0.5

    # GUARDRAIL 5: Token set ratio is too high compared to other scores
    # This catches cases where token_set gives false confidence
    if token_set > 90 and ratio < 70:
        # Token set is giving false confidence due to shared common words
        score *= 0.6

    # GUARDRAIL 6: Same words, different order. Bank names are order-sensitive
    # ("NATIONAL CITY BANK" is PNC, "CITY NATIONAL BANK" is RBC) but token_sort
    # and token_set score such pairs 100, so the ensemble reaches ~90 on
    # order-insensitive measures alone. Push them below the review floor; the
    # embedding stage still sees them.
    tokens1, tokens2 = name1.split(), name2.split()
    if tokens1 != tokens2 and sorted(tokens1) == sorted(tokens2):
        score *= 0.93

    return score


@lru_cache(maxsize=_CACHE_SIZE)
def has_shared_meaningful_identifier(name1: str, name2: str) -> bool:
    """
    Check if two entity names share sufficient meaningful identifying words.

    Logic:
    1. If words are identical, match
    2. If one set is a subset AND the subset contains a unique identifier
       (not just generic), match
    3. Otherwise require > 50% overlap in both directions
    """
    words1 = _get_meaningful_words_frozen(name1)
    words2 = _get_meaningful_words_frozen(name2)

    if not words1 or not words2:
        return False

    shared = words1 & words2
    if not shared:
        return False

    if words1 == words2:
        return True

    smaller, larger = (words1, words2) if len(words1) <= len(words2) else (words2, words1)
    if smaller <= larger and (smaller - GENERIC_PREFIXES):
        return True

    overlap1 = len(shared) / len(words1)
    overlap2 = len(shared) / len(words2)
    return overlap1 > 0.5 and overlap2 > 0.5

@lru_cache(maxsize=_CACHE_SIZE)
def extract_entity_core(name: str) -> str:
    """
    Extract the core identifying part of an entity name.

    Strategy: skip common prefixes only if there's a meaningful word after them.
    Uses module-level ALL_GENERIC_UPPER for the exclusion set.

    Examples:
    - "STEARNS BANK EQUIPMENT FINANCE" -> "STEARNS"
    - "AMERICAN HERITAGE NATIONAL BANK" -> "HERITAGE"
    - "FIRST CHOICE BANK" -> "CHOICE"
    """
    words = name.upper().split()
    if not words:
        return name

    for word in words:
        if word not in ALL_GENERIC_UPPER and len(word) > 1:
            return word

    fallback_words = []
    for word in words:
        if word not in {'THE', 'A', 'AN'} and len(word) > 1:
            fallback_words.append(word)
            if len(fallback_words) >= 2:
                break

    if fallback_words:
        return ' '.join(fallback_words)
    return words[0]

@lru_cache(maxsize=_CACHE_SIZE)
def _long_tokens(name: str) -> frozenset:
    """Lower-cased tokens of length > 2, cached."""
    return frozenset(t for t in name.lower().split() if len(t) > 2)


def min_token_overlap(name1: str, name2: str, min_overlap: int = 1) -> bool:
    """
    Check if two names share at least ``min_overlap`` tokens (len > 2).
    Helps filter out false positives from embedding similarity.
    """
    return len(_long_tokens(name1) & _long_tokens(name2)) >= min_overlap

def entity_types_compatible(type1: str, type2: str) -> bool:
    """
    Check if two entity types are compatible for matching.
    Returns False for clear mismatches (e.g., BANK vs MANUFACTURER).
    """
    # If either is OTHER, allow the match
    if type1 == 'OTHER' or type2 == 'OTHER':
        return True

    # Same type is always compatible
    if type1 == type2:
        return True

    # Define compatible type pairs
    compatible_pairs = {
        ('BANK', 'LEGAL'),      # Banks often act as trustees
        ('INVESTMENT', 'LEGAL'), # Investment firms as custodians
    }

    pair = (type1, type2) if type1 < type2 else (type2, type1)
    return pair in compatible_pairs

def should_merge_clusters(label_a: str, label_b: str, embedding_sim: float) -> bool:
    """
    Determine if two cluster labels should be merged.
    VERY CONSERVATIVE: Requires BOTH high embedding similarity AND high fuzzy similarity.
    This prevents merging semantically different entities.
    """
    
    a_lower = label_a.lower().strip()
    b_lower = label_b.lower().strip()

    # Always require fuzzy similarity check to prevent bad merges
    fuzzy_ratio = fuzz.ratio(a_lower, b_lower)
    fuzzy_token_set = fuzz.token_set_ratio(a_lower, b_lower)
    fuzzy_partial = fuzz.partial_ratio(a_lower, b_lower)

    # VERY HIGH embedding similarity (>=0.92) - still require strong fuzzy match
    if embedding_sim >= 0.92:
        # Require at least 80% fuzzy similarity
        if fuzzy_token_set >= 80 and fuzzy_ratio >= 60:
            return True
        return False

    # HIGH embedding similarity (>=0.88) - require very strong fuzzy match
    if embedding_sim >= 0.88:
        # Require both high token set AND partial ratio
        if fuzzy_token_set >= 85 and fuzzy_partial >= 85:
            return True
        # Substring containment only for longer, meaningful strings
        if len(a_lower) >= 10 and len(b_lower) >= 10:
            if a_lower in b_lower or b_lower in a_lower:
                if fuzzy_ratio >= 70:  # Additional guard
                    return True
        return False

    # Below 0.88 embedding similarity - VERY strict requirements
    if embedding_sim >= 0.85:
        # Only merge if fuzzy scores are very high
        if fuzzy_token_set >= 90 and fuzzy_partial >= 90 and fuzzy_ratio >= 75:
            return True
        return False

    # Below 0.85 - don't merge based on embeddings alone
    # Only consider exact substring matches for well-formed names
    if embedding_sim >= 0.80:
        min_len = min(len(a_lower), len(b_lower))
        # Only if shorter is substantial and contained in longer
        if min_len >= 15 and (a_lower in b_lower or b_lower in a_lower):
            # And they share most words
            words_a = set(a_lower.split())
            words_b = set(b_lower.split())
            overlap = len(words_a & words_b)
            total = len(words_a | words_b)
            if total > 0 and overlap / total >= 0.7:
                return True

    return False

