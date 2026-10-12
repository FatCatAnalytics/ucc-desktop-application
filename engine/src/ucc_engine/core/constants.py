# =============================================================================
# CONSTANTS
# =============================================================================
import re

LEGAL_SUFFIXES = [
    r'\bINC\.?$', r'\bINCORPORATED$', r'\bCORPORATION$', r'\bCORP\.?$',
    r'\bLLC\.?$', r'\bL\.?L\.?C\.?$', r'\bLIMITED LIABILITY COMPANY$',
    r'\bL\.?P\.?$', r'\bLIMITED PARTNERSHIP$', r'\bLTD\.?$', r'\bLIMITED$',
    r'\bCO\.?$', r'\bCOMPANY$', r'\bN\.?A\.?$', r'\bNATIONAL ASSOCIATION$',
    r'\bF\.?S\.?B\.?$', r'\bFEDERAL SAVINGS BANK$',
    r'\bS\.?B\.?$', r'\bSAVINGS BANK$', r'\bTRUST COMPANY$',
    r'\bPCA$', r'\bSSB$', r'\bFSB$',
    # NOTE: BANK, BANKING, CREDIT UNION intentionally NOT stripped —
    # they carry entity-type signal needed for matching & clustering.
]

ROLE_TRIGGERS = [
    r"\bsolely\s+in\s+its\s+capacity\s+as\b",
    r"\bin\s+its\s+capacity\s+as\b",
    r"\bas\s+trustee\b",
    r"\bas\s+indenture\s+trustee\b",
    r"\bas\s+administrative\s+agent\b",
    r"\bas\s+collateral\s+agent\b",
    r"\bas\s+security\s+trustee\b",
    r"\bas\s+paying\s+agent\b",
    r"\bas\s+registrar\b",
    r"\bas\s+transfer\s+agent\b",
    r"\bas\s+fiscal\s+agent\b",
    r"\bas\s+custodian\b",
    r"\bas\s+purchaser\s+representative\b",
    r"\bas\s+agent\b",
    r"\bas\s+representative\b",
    r"\bfor\s+itself\s+and\b",
    r"\bon\s+behalf\s+of\b",
]

ROLE_RE = re.compile(r"(?i)\s*(?:,|\s)+(" + "|".join(ROLE_TRIGGERS) + r")\b.*$")

# ---------------------------------------------------------------------------
# PRE-CLEAN: structural junk UCC filers wrap around the lender's name.
# Applied by normalize_name() *before* suffix / common-word stripping. Each
# rule either cuts a tail off or picks the one name out of a phrase that
# names two. Motivation (June 2026 hold-out on the Master Mapping): a third of
# known names scored < 80 against their own master row purely because of
# this wrapping; stripping it lifted fuzzy auto-recall from 55% to 67% at
# 99.3% precision.
# ---------------------------------------------------------------------------

# "(LENDER AND LETTER OF CREDIT ISSUER)", "(AS SUCCESSOR IN INTEREST ...)"
PARENTHETICAL_RE = re.compile(r"\s*\([^)]*\)")

# "X, A DIVISION OF Y" / "X DIVISION OF Y"  -> Y is the legal lender.
DIVISION_OF_RE = re.compile(r"(?i)^.*?\b(?:A\s+)?DIVISION\s+OF\s+")

# "FDIC AS RECEIVER OF/FOR Y, CITY, STATE" -> Y (cut at the next comma).
RECEIVER_OF_RE = re.compile(r"(?i)^.*?\bAS\s+RECEIVER\s+(?:OF|FOR)\s+(?:THE\s+)?")

# The current entity comes first; what follows names the predecessor or an alias.
#   "TD BANK, NA, AS SUCCESSOR BY MERGER TO COMMERCE BANK"       -> TD BANK, NA
#   "CENTENNIAL BANK F/K/A LIBERTY BANK OF ARKANSAS"             -> CENTENNIAL BANK
#   "THE CENTRAL TRUST BANK DBA CENTRAL BANK OF ST. LOUIS"       -> THE CENTRAL TRUST BANK
PREDECESSOR_TAIL_RE = re.compile(
    r"(?i)[\s,]*\b(?:"
    r"(?:AS\s+)?SUC+ES+OR(?:S)?(?:[\s-]+(?:BY|IN)[\s-]+(?:MERGER|INTEREST|CONVERSION)(?:[\s-]+(?:BY[\s-]+MERGER|IN[\s-]+INTEREST))?)?[\s-]+(?:TO|WITH|OF)"
    r"|F/?K/?A|N/?K/?A|FORMERLY\s+KNOWN\s+AS|NOW\s+KNOWN\s+AS"
    r"|D/?B/?A|DOING\s+BUSINESS\s+AS"
    r")\b.*$"
)

# Capacity / role tails that ROLE_TRIGGERS miss (typos, extra words, no "AS"):
#   ", AS ADMINISTRIVE AGENT", "AS LOAN TRUSTEE", ", NOT IN ITS INDIVIDUAL CAPACITY ...",
#   ", A TEXAS STATE BANKING ASSOCIATION", ", AN ALABAMA BANKING CORPORATION",
#   ", WITH ITS MAIN OFFICE IN ...", ", ITS SUCCESSORS AND ASSIGNS"
ROLE_WORDS = (
    r"TRUSTEE|AGENT|LENDER|ISSUER|CUSTODIAN|REPRESENTATIVE|HOLDER|PURCHASER|DEPOSITOR|"
    r"SERVICER|MORTGAGEE|ASSIGNEE|SECURED\s+PARTY|PROVIDER|PARTICIPANT|ARRANGER|BUYER|SELLER|"
    r"NOMINEE|BAILEE|ESCROW\s+AGENT|DEPOSITARY|CREDITOR|LESSOR"
)
CAPACITY_TAIL_RE = re.compile(
    r"(?i)[\s,;:-]*\b(?:"
    r"(?:AS|AN?)\s+(?:[A-Z&/.'-]+\s+){0,5}?(?:" + ROLE_WORDS + r")"
    r"|NOT\s+IN\s+ITS\s+INDIVIDUAL\s+CAPACITY|SOLELY\s+(?:AS|IN)|IN\s+ITS\s+CAPACITY"
    r"|ITS\s+SUCCESSORS?\s+AND\s+ASSIGNS|WITH\s+ITS\s+MAIN\s+OFFICE"
    r"|AN?\s+(?:[A-Z]+\s+){0,3}?(?:STATE\s+)?(?:BANKING\s+|NATIONAL\s+)?(?:CORPORATION|ASSOCIATION|COMPANY|BANK|INSTITUTION)\b"
    r")\b.*$"
)

# Charter words that appear mid-name, not just at the end: "WELLS FARGO BANK, N.A., AS ..."
CHARTER_MID_RE = re.compile(
    r"(?i)[\s,]*\b(?:NATIONAL\s+ASSOCIATION|NATIONAL\s+BANKING\s+ASSOCIATION|N\.\s?A\.?|NA|F\.?S\.?B\.?|SSB)(?=[\s,;:)]|$)"
)

# "PARK BANK- BROOKFIELD", "PINNACLE BANK - NEWCASTLE", "FIRST SOUTHWEST BANK-BISMARCK MAIN",
# "CITIZENS BANK - HUMBLE BANKING CENTER": a branch/office qualifier after the bank name.
BRANCH_TAIL_RE = re.compile(r"(?i)(\bBANK)\s*-\s*[A-Z][A-Z .]*?(?:\s+(?:MAIN|OFFICE|BRANCH|BANKING\s+CENTER|CENTER))?\s*$")

COMMON_WORDS = sorted([
    # Multi-word phrases
    r'\bFORMERLY\s+KNOWN\s+AS\b', r'\bSUCCESSOR\s+BY\s+MERGER\s+OF\b',
    r'\bA\s+DIVISION\s+OF\b', r'\bDIVISION\s+OF\b', r'\bDIVISION\b',
    r'\bD/?B/?A\b', r'\bF/?K/?A\b', r'\bAKA\b',
    # Single words — keep only determiners/prepositions that never carry entity meaning
    r'\bTHE\b', r'\bAND\b', r'\b&\b',
    r'\bAS\b', r'\bBY\b', r'\bFOR\b', r'\bIN\b', r'\bTO\b', r'\bWITH\b',
    # NOTE: \bA\b and \bAN\b removed — too aggressive; they match single-letter
    # tokens that appear after punctuation is stripped (e.g. "N.A." → "N A").
], key=len, reverse=True)  # Longest patterns first → greedy matching

ABBREVIATIONS = {
    r'\bNTL\b': 'NATIONAL',
    r'\bFED\b': 'FEDERAL',
    r'\bCTR\b': 'CENTER',
    r'\bFIN\b': 'FINANCIAL',
    r'\bSVC\b': 'SERVICE',
    r'\bSVCS\b': 'SERVICES',
    r'\bINV\b': 'INVESTMENT',
    r'\bINVS\b': 'INVESTMENTS',
    r'\bMGT\b': 'MANAGEMENT',
    r'\bASSOC\b': 'ASSOCIATION',
    r'\bGRP\b': 'GROUP',
    r'\bCOMM\b': 'COMMERCIAL',
    r'\bBK\b': 'BANK',
    r'\bNATL\b': 'NATIONAL',
    r'\bINTL\b': 'INTERNATIONAL',
    r'\bSVGS\b': 'SAVINGS',
    r'\bEQUIP\b': 'EQUIPMENT',
    r'\bCONST\b': 'CONSTRUCTION',
}

ENTITY_TYPE_PATTERNS = {
    'BANK': [
        r'\bBANK\b', r'\bBANKING\b', r'\bSAVINGS\b', r'\bTRUST\b',
        r'\bN\.?A\.?\b', r'\bF\.?S\.?B\.?\b', r'\bCREDIT UNION\b'
    ],
    'GOVERNMENT': [
        r'\bADMINISTRATION\b', r'\bFEDERAL\b', r'\bU\.?S\.?\b',
        r'\bGOVERNMENT\b', r'\bAGENCY\b', r'\bDEPARTMENT\b'
    ],
    'INVESTMENT': [
        r'\bCAPITAL\b', r'\bPARTNERS\b', r'\bINVESTMENT\b',
        r'\bFUND\b', r'\bADVISOR\b', r'\bASSET\b', r'\bEQUITY\b'
    ],
    'MANUFACTURER': [
        r'\bMANUFACTURING\b', r'\bINDUSTRIES\b', r'\bPRODUCTS\b',
        r'\bEQUIPMENT\b', r'\bMACHINERY\b'
    ],
    'LEGAL': [
        r'\bTRUSTEE\b', r'\bAGENT\b', r'\bCUSTODIAN\b', r'\bLAW\b'
    ]
}