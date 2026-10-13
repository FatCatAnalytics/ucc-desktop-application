"""Matching-quality regression test (no models needed).

Hold-out on the real Master Mapping: take known (raw name -> bank) rows out,
feed the raw names back through the Stage 3 logic (normalise, fuzzy score,
the four guardrails) against the rest of the master, and measure precision /
recall at the auto threshold. Numbers below are the June 2026 baseline after
the pre-clean normaliser landed; they are floors, so a change that quietly
hurts matching fails here instead of in a client's updated master.

Skipped when the Master Mapping file is not present (it is reference data in
``input/`` at the repository root).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
MASTER = REPO / "input" / "Master Mapping.csv"

# Floors, not targets. Measured 2026-10-12 on 1,500 held-out names (seed 42):
#   current normaliser:  auto@96 precision 0.990 recall 0.551
#   + pre-clean:         auto@96 precision 0.993 recall 0.671 ; auto@90 precision 0.990 recall 0.719
MIN_PRECISION_AT_AUTO = 0.98
MIN_RECALL_AT_AUTO = 0.68
MIN_RECALL_ANY_SCORE = 0.84


def _holdout(n: int = 1500, seed: int = 42):
    m = pd.read_csv(MASTER, encoding="latin-1")
    m["raw"] = m["SECP_NAME"].astype(str).str.strip().str.upper()
    m["bank"] = m["SECP_NAME_MAPPED"].astype(str).str.strip()
    u = m.drop_duplicates("raw")[["raw", "bank"]]
    counts = u["bank"].value_counts()
    eligible = u[u["bank"].map(counts) >= 3]  # the bank must still be recoverable from other rows
    held = eligible.sample(min(n, len(eligible)), random_state=seed)
    return held, u.drop(held.index)


def _best_match(query_norm: str, pool_norm: list[str], pool_bank: list[str]):
    from rapidfuzz import fuzz, process
    from ucc_engine.core.utils import (
        combined_fuzzy_score, detect_entity_type, entity_types_compatible, has_shared_meaningful_identifier,
        min_token_overlap,
    )

    best = None
    for cand, _score, idx in process.extract(query_norm, pool_norm, scorer=fuzz.token_set_ratio, limit=15, score_cutoff=60):
        if not (has_shared_meaningful_identifier(query_norm, cand)
                and entity_types_compatible(detect_entity_type(query_norm), detect_entity_type(cand))
                and min_token_overlap(query_norm, cand)):
            continue
        s = combined_fuzzy_score(query_norm, cand)
        if best is None or s > best[0]:
            best = (s, pool_bank[idx])
    return best


@pytest.mark.skipif(not MASTER.exists(), reason="Master Mapping.csv not present")
def test_fuzzy_stage_precision_and_recall_on_master_holdout():
    from ucc_engine.config import PipelineConfig
    from ucc_engine.core.utils import normalize_name

    auto = PipelineConfig().fuzzy_auto_threshold
    held, rest = _holdout()
    pool = [(normalize_name(r), b) for r, b in zip(rest["raw"], rest["bank"])]
    pool = [(n, b) for n, b in pool if n]
    pool_norm, pool_bank = [p[0] for p in pool], [p[1] for p in pool]

    rows = []
    for raw, bank in zip(held["raw"], held["bank"]):
        q = normalize_name(raw)
        best = _best_match(q, pool_norm, pool_bank) if q else None
        rows.append({"raw": raw, "bank": bank, "score": best[0] if best else float("nan"),
                     "cand_bank": best[1] if best else None})
    d = pd.DataFrame(rows)
    d["correct"] = d["cand_bank"] == d["bank"]
    accepted = d[d["score"] >= auto]

    precision = accepted["correct"].mean()
    recall = accepted["correct"].sum() / len(d)
    recall_any = d["correct"].mean()
    print(f"\nfuzzy hold-out: auto@{auto}: accepted {len(accepted)}, precision {precision:.3f}, recall {recall:.3f}; "
          f"recall at any score {recall_any:.3f}")
    assert precision >= MIN_PRECISION_AT_AUTO, f"precision at auto threshold fell to {precision:.3f}"
    assert recall >= MIN_RECALL_AT_AUTO, f"recall at auto threshold fell to {recall:.3f}"
    assert recall_any >= MIN_RECALL_ANY_SCORE


def test_pre_clean_picks_the_lender_out_of_the_filed_string():
    from ucc_engine.core.utils import normalize_name, pre_clean_name

    cases = {
        # parentheticals / capacity tails
        "STARION BANK (LENDER AND LETTER OF CREDIT ISSUER)": "STARION BANK",
        "WELLS FARGO BANK, NATIONAL ASSOCIATION, AS ADMINISTRIVE AGENT": "WELLS FARGO BANK",
        "WILMINGTON TRUST, NATIONAL ASSOCIATION, AS LOAN TRUSTEE": "WILMINGTON TRUST",
        "WILMINGTON SAVINGS FUND SOCIETY, FSB, NOT IN ITS INDIVIDUAL CAPACITY BUT SOLELY AS INDENTURE TRUSTEE": "WILMINGTON SAVINGS FUND SOCIETY",
        "BANK ONE, NA, WITH ITS MAIN OFFICE IN COLUMBUS, OH": "BANK ONE",
        "CITIZENS BANK NA AS ADMIN AGENT": "CITIZENS BANK",
        # charter boilerplate that used to drive 0.99 false positives
        "PLAINSCAPITAL BANK, A TEXAS STATE BANKING ASSOCIATION": "PLAINSCAPITAL BANK",
        "PROGRESS BANK AND TRUST, AN ALABAMA BANKING CORPORATION": "PROGRESS BANK AND TRUST",
        "BANK OF THE WEST, A CALIFORNIA BANKING CORPORATION": "BANK OF THE WEST",
        # the current entity wins over predecessor / alias
        "TD BANK, NA, AS SUCCESSOR BY MERGER TO MERCANTILE BANK": "TD BANK",
        "WESTERN ALLIANCE BANK, AS SUCCESSOR-IN-INTEREST TO BRIDGE BANK, N.A.": "WESTERN ALLIANCE BANK",
        "CENTENNIAL BANK F/K/A LIBERTY BANK OF ARKANSAS": "CENTENNIAL BANK",
        "THE CENTRAL TRUST BANK DBA CENTRAL BANK OF ST. LOUIS": "THE CENTRAL TRUST BANK",
        "FIRST HORIZON BANK (SUCCESSOR BY MERGER TO IBERIABANK), AS ADMINISTRATIVE AGENT": "FIRST HORIZON BANK",
        # "division of" / "receiver of": the parent is the lender
        "KEY EQUIPMENT FINANCE, A DIVISION OF KEYBANK NA": "KEYBANK",
        "German American Bank d/b/a Heartland Bank, a Division of German American Bank": "GERMAN AMERICAN BANK",
        "FEDERAL DEPOSIT INSURANCE CORPORATION AS RECEIVER OF FIRST REPUBLIC BANK, SAN FRANCISCO, CALIFORNIA": "FIRST REPUBLIC BANK",
        # branch qualifiers
        "PARK BANK- BROOKFIELD": "PARK BANK",
        "FIRST SOUTHWEST BANK-BISMARCK MAIN": "FIRST SOUTHWEST BANK",
        "CITIZENS BANK - HUMBLE BANKING CENTER": "CITIZENS BANK",
        # untouched
        "FIRST-CITIZENS BANK & TRUST COMPANY": "FIRST-CITIZENS BANK & TRUST COMPANY",
        "STATE BANK OF TEXAS": "STATE BANK OF TEXAS",
        "FIRST NATIONAL BANK OF PANDORA": "FIRST NATIONAL BANK OF PANDORA",
        "MUFG BANK, LTD.": "MUFG BANK, LTD.",
        "U.S. SMALL BUSINESS ADMINISTRATION": "U.S. SMALL BUSINESS ADMINISTRATION",
    }
    for raw, want in cases.items():
        assert pre_clean_name(raw) == want, f"{raw!r} -> {pre_clean_name(raw)!r}, expected {want!r}"
    # never empties a name
    assert pre_clean_name("NA") == "NA" and normalize_name("") == ""
    # and the two boilerplate false positives from June no longer collide
    assert normalize_name("STATE BANK OF TEXAS") != normalize_name("PLAINSCAPITAL BANK, A TEXAS STATE BANKING ASSOCIATION")
    assert normalize_name("WESTERN ALLIANCE BANK, AS SUCCESSOR-IN-INTEREST TO BRIDGE BANK, N.A.") == "WESTERN ALLIANCE BANK"


def test_same_words_different_order_do_not_auto_match():
    from ucc_engine.core.utils import combined_fuzzy_score, normalize_name

    a, b = normalize_name("NATIONAL CITY BANK"), normalize_name("CITY NATIONAL BANK")
    assert combined_fuzzy_score(a, b) < 85, "order-swapped bank names must not reach the review floor on fuzzy alone"
    assert combined_fuzzy_score(a, a) == 100


TRACKED = REPO / "input" / "tracked_banks.csv"


@pytest.mark.skipif(not (MASTER.exists() and TRACKED.exists()), reason="input files not present")
def test_every_master_bank_is_a_tracked_bank_with_the_same_firm_id():
    """Every SECP_NAME_MAPPED in the master must resolve to a tracked bank (case-insensitive,
    the lookup the enrichment stage uses) and carry the same Firm ID, so no mapped filing
    is left without a Firm ID and the hand-mapping picker offers every bank the master uses."""
    from ucc_engine.datasources import detect_delimiter

    m = pd.read_csv(MASTER, encoding="latin-1", low_memory=False)
    t = pd.read_csv(TRACKED, sep=detect_delimiter(str(TRACKED)), encoding="latin-1", dtype=str, keep_default_na=False)
    id_col = "Firm_ID" if "Firm_ID" in t.columns else "Firm ID"
    tracked = {n.strip().upper(): i.strip() for n, i in zip(t["Bank Name"], t[id_col])}
    assert len(tracked) == len(t), "tracked bank names must be unique (case-insensitive)"

    m["bank"] = m["SECP_NAME_MAPPED"].astype(str).str.strip().str.upper()
    m["fid"] = pd.to_numeric(m["Firm ID"], errors="coerce").astype("Int64").astype(str)
    missing = sorted(set(m["bank"]) - set(tracked))
    assert not missing, f"master banks not in tracked_banks.csv: {missing}"
    per_bank = m[m["fid"] != "<NA>"].groupby("bank")["fid"].agg(lambda s: sorted(set(s)))
    clashes = {b: (ids, tracked[b]) for b, ids in per_bank.items() if ids != [tracked[b]]}
    assert not clashes, f"Firm ID differs between master and tracked list: {clashes}"
