#!/usr/bin/env python3
"""
Column-mapping + capacity + pool-filter layer for the REAL TEL data.
=====================================================================
The two production spreadsheets are raw Google-Form exports whose headers are
full questions with smart quotes and embedded newlines. This module is the ONLY
place that knows about those headers: everything downstream works with clean
internal field names.

Everything tunable lives in the EDITABLE DICTS at the top:
  - MENTOR_FIELD_KEYS / STUDENT_FIELD_KEYS   real header  -> internal field
  - CAPACITY_RULES                           free-text time-commitment -> slots
  - ONE_TO_ONE_SIGNAL / CONSENT_*            pool-filter vocabulary

Matching is done on a NORMALIZED header (smart quotes flattened, punctuation and
newlines collapsed to single spaces, lowercased) so the keys stay short and
robust to the form's messy punctuation.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

# ─────────────────────────── EDITABLE: header → field ───────────────────────────
# Each value is matched as a substring against the normalized header; first hit
# wins, so order from most- to least-specific where two could collide.

MENTOR_FIELD_KEYS: dict[str, str] = {
    "email address": "email",
    "current role": "role_org",
    "location": "location",
    "linkedin": "linkedin",
    "industry": "industry",
    "topics youre most interested": "topics",          # multi
    "preferred format": "formats",                      # multi  (1:1 signal lives here)
    "general availability": "availability",             # text
    "time commitment youre open to": "time_commitment", # text  -> capacity
    "please share any key challenges": "notes",         # text
    "brief bio": "bio",                                 # text
    "may we share your bio and contact": "consent_to_share",
}

STUDENT_FIELD_KEYS: dict[str, str] = {
    "email": "email",
    "majors": "majors",
    "academic level": "academic_level",
    "expected graduation": "grad_date",
    "linkedin": "links",
    "what strengths or interests": "strengths",         # multi
    "what best describes where you are today": "where_today",
    "which experiences are you interested": "experiences",  # multi
    "what are your top one or two choices": "top_choices",  # text
}

# First/last name columns are combined into `name` (handled in code, not the dict).
_NAME_FIRST = "first name"
_NAME_LAST = "last name"

# ───────────────────── EDITABLE: free-text capacity → slot count ─────────────────────
# Ordered rules: the first rule whose any-substring is found in the normalized
# time-commitment text wins. `slots` is mentees/week-equivalent BEFORE the
# hours-per-mentee division (these are already in "students at once" units).
# Anything unmatched is UNKNOWN -> defaults to UNKNOWN_CAPACITY_DEFAULT and is flagged.
CAPACITY_RULES: list[tuple[list[str], int]] = [
    (["ongoing"], 4),                                   # "Ongoing engagement throughout the year"
    (["few hours per semester"], 2),                    # "A few hours per semester"
    (["one time", "one panel", "once or twice a year"], 1),
    (["anything depending", "open to what is needed", "whatever",
      "depends", "whenever", "as needed"], 3),
]
UNKNOWN_CAPACITY_DEFAULT = 1   # used when blank or unrecognized (and flagged + logged)

# ───────────────────── EDITABLE: pool-filter vocabulary ─────────────────────
# A mentor is in the 1:1 pool only if their `formats` mentions this phrase.
ONE_TO_ONE_SIGNAL = "mentoring conversations"           # i.e. "1:1 mentoring conversations"
CONSENT_YES = "yes"
CONSENT_NO = "no"


# ─────────────────────────────── normalization ───────────────────────────────

_SMART = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "…": " ",
}


def normalize(text) -> str:
    """Flatten smart quotes, drop apostrophes, turn every other non-alphanumeric
    run (punctuation, newlines, NBSP) into a single space, lowercase, strip."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = unicodedata.normalize("NFKC", str(text))
    for k, v in _SMART.items():
        s = s.replace(k, v)
    s = s.replace("'", "")                       # you're -> youre
    s = re.sub(r"[^a-z0-9]+", " ", s.lower())     # everything else -> space
    return re.sub(r"\s+", " ", s).strip()


def _find_col(columns: list[str], key: str) -> str | None:
    key_n = normalize(key)
    for col in columns:
        if key_n in normalize(col):
            return col
    return None


def _map_columns(df: pd.DataFrame, field_keys: dict[str, str]) -> dict[str, str]:
    """internal_field -> real column name (only the ones present)."""
    cols = list(df.columns)
    out: dict[str, str] = {}
    for key, field in field_keys.items():
        if field in out:           # earlier (more specific) key already claimed it
            continue
        col = _find_col(cols, key)
        if col is not None:
            out[field] = col
    return out


def _combine_name(row: pd.Series, first_col: str | None, last_col: str | None) -> str:
    parts = []
    for c in (first_col, last_col):
        if c is not None:
            v = row.get(c)
            if isinstance(v, str) and v.strip():
                parts.append(v.strip())
    return " ".join(parts).strip()


# ─────────────────────────────── capacity ───────────────────────────────

def capacity_from_text(text) -> tuple[int, bool, str]:
    """Return (slots, is_unknown, reason). Unknown -> default + flag, never raises."""
    n = normalize(text)
    if not n:
        return UNKNOWN_CAPACITY_DEFAULT, True, "blank"
    for substrings, slots in CAPACITY_RULES:
        if any(s in n for s in substrings):
            return slots, False, ""
    return UNKNOWN_CAPACITY_DEFAULT, True, f"unrecognized: {str(text).strip()[:80]}"


def _consent(value) -> str:
    n = normalize(value)
    if n == CONSENT_YES:
        return "yes"
    if n == CONSENT_NO:
        return "no"
    return "unknown"


# ─────────────────────────────── loaders ───────────────────────────────

def _read(path: str) -> pd.DataFrame:
    p = str(path)
    if p.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(p, dtype=str).fillna("")
    return pd.read_csv(p, dtype=str).fillna("")


def load_mentors(path: str) -> pd.DataFrame:
    raw = _read(path)
    m = _map_columns(raw, MENTOR_FIELD_KEYS)
    first = _find_col(list(raw.columns), _NAME_FIRST)
    last = _find_col(list(raw.columns), _NAME_LAST)

    rows = []
    for _, r in raw.iterrows():
        rec = {field: str(r.get(col, "")).strip() for field, col in m.items()}
        rec["name"] = _combine_name(r, first, last)
        cap, cap_unknown, cap_reason = capacity_from_text(rec.get("time_commitment", ""))
        rec["capacity_slots"] = cap
        rec["capacity_unknown"] = cap_unknown
        rec["capacity_reason"] = cap_reason
        rec["consent"] = _consent(rec.get("consent_to_share", ""))
        rec["offers_1to1"] = ONE_TO_ONE_SIGNAL in normalize(rec.get("formats", ""))
        rows.append(rec)

    df = pd.DataFrame(rows)
    # Drop junk rows: no name AND no email.
    df["_junk"] = (df.get("name", "") == "") & (df.get("email", "") == "")
    df = df[~df["_junk"]].drop(columns=["_junk"]).reset_index(drop=True)
    return df


def load_students(path: str) -> pd.DataFrame:
    raw = _read(path)
    m = _map_columns(raw, STUDENT_FIELD_KEYS)
    first = _find_col(list(raw.columns), _NAME_FIRST)
    last = _find_col(list(raw.columns), _NAME_LAST)

    rows = []
    for _, r in raw.iterrows():
        rec = {field: str(r.get(col, "")).strip() for field, col in m.items()}
        rec["name"] = _combine_name(r, first, last)
        rows.append(rec)

    df = pd.DataFrame(rows)
    df["_junk"] = (df.get("name", "") == "") & (df.get("email", "") == "")
    df = df[~df["_junk"]].drop(columns=["_junk"]).reset_index(drop=True)
    return df


def pool_filter_mentors(mentors: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Split mentors into the 1:1-eligible pool and an excluded list with reasons.
    Excluded if: formats lack the 1:1 signal, OR consent == 'no'."""
    excluded: list[dict] = []
    keep_idx = []
    for i, r in mentors.iterrows():
        reasons = []
        if not r["offers_1to1"]:
            reasons.append("no 1:1 mentoring format (panel/group/blank only)")
        if r["consent"] == "no":
            reasons.append("consent_to_share = No")
        if reasons:
            excluded.append({"name": r["name"] or "(unnamed)", "reasons": reasons})
        else:
            keep_idx.append(i)
    pool = mentors.loc[keep_idx].reset_index(drop=True)
    return pool, excluded


def reconcile(mentors_all: pd.DataFrame, pool: pd.DataFrame,
              excluded: list[dict], students: pd.DataFrame) -> dict:
    total_slots = int(pool["capacity_slots"].sum()) if len(pool) else 0
    unknown_caps = [
        {"name": r["name"], "reason": r["capacity_reason"]}
        for _, r in pool.iterrows() if r["capacity_unknown"]
    ]
    return {
        "mentors_total": int(len(mentors_all)),
        "mentors_in_1to1_pool": int(len(pool)),
        "mentors_excluded": len(excluded),
        "excluded_detail": excluded,
        "total_mentor_slots": total_slots,
        "students_total": int(len(students)),
        "expected_unmatched": max(0, int(len(students)) - total_slots),
        "unknown_capacity_count": len(unknown_caps),
        "unknown_capacity_detail": unknown_caps,
        # The student form has no explicit "request a 1:1 mentor" question, so all
        # students are treated as mentor-seeking by default (surfaced, not hidden).
        "mentor_request_field_present": False,
        "students_matched_by_default": int(len(students)),
    }
