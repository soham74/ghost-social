#!/usr/bin/env python3
"""
Rebuild display_data/results.json with REAL identities from the already-committed
LLM match results — NO recompute. Joins each match back to the original
spreadsheets by the deterministic student index (the loader order), producing one
internal view record per match:

  matched:   mentor_name, mentor_email, student_name, student_email, confidence, rationale
  unmatched: student_name, student_email

Rationale text is kept as-is (the original, unscrubbed LLM text from output/).

  python rebuild_display.py --mentors "<xlsx>" --students "<csv>"
"""

import argparse
import json
from pathlib import Path

import data_mapping as dm

BASE = Path(__file__).parent
PUBLIC = BASE / "display_data" / "results.json"     # current file (carries Student NNN labels + order)
INTERNAL = BASE / "output" / "results.json"          # unscrubbed real-name rationales, same order


def _idx(label: str) -> int:
    return int("".join(c for c in str(label) if c.isdigit())) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mentors", required=True)
    ap.add_argument("--students", required=True)
    args = ap.parse_args()

    students = dm.load_students(args.students).reset_index(drop=True)
    mentors = dm.load_mentors(args.mentors)
    mentor_email = {r["name"]: r.get("email", "") for _, r in mentors.iterrows()}

    cur = json.loads(PUBLIC.read_text())
    internal = json.loads(INTERNAL.read_text()) if INTERNAL.exists() else {"matches": []}
    imatches = internal.get("matches", [])

    matches, mismatches = [], 0
    for i, m in enumerate(cur.get("matches", [])):
        label = m.get("student", m.get("mentee_name"))
        j = _idx(label)
        stu = students.iloc[j]
        im = imatches[i] if i < len(imatches) else {}
        if im and str(im.get("mentee_name", "")).strip() != str(stu["name"]).strip():
            mismatches += 1
        mentor_name = im.get("mentor_name") or m.get("mentor")
        matches.append({
            "mentor_name": mentor_name,
            "mentor_email": mentor_email.get(mentor_name, ""),
            "student_name": stu["name"],
            "student_email": stu.get("email", ""),
            "confidence": m.get("confidence", m.get("confidence_score")),
            "rationale": im.get("rationale") or m.get("rationale"),   # as-is, unscrubbed
        })

    unmatched = []
    for label in cur.get("unmatched", []):
        j = _idx(label)
        stu = students.iloc[j]
        unmatched.append({"student_name": stu["name"], "student_email": stu.get("email", "")})

    out = {
        "generated_at": cur.get("generated_at"),
        "generation_mode": cur.get("generation_mode"),
        "scope": "Internal coordinator view: real mentor + student identities with contact info.",
        "matches": matches,
        "unmatched": unmatched,
    }
    PUBLIC.write_text(json.dumps(out, indent=2, default=str))
    print(f"Rebuilt {PUBLIC}: {len(matches)} matched + {len(unmatched)} unmatched (real identities).")
    if mismatches:
        print(f"  WARNING: {mismatches} index/name sanity mismatches — check ordering.")


if __name__ == "__main__":
    main()
