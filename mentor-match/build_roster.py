#!/usr/bin/env python3
"""
Build the ADMIN ROSTER (PII) from the already-committed public results — no re-run,
no API. Maps each anonymized "Student NNN" label back to the real student
name + email and adds the mentor's email, so the coordinator's admin view can show
full contact info per match.

Output: mentor-match/private/roster.json  (GIT-IGNORED — never committed).
On Render, supply this file's contents via the ADMIN_ROSTER env var instead.

  python build_roster.py --mentors "<xlsx>" --students "<csv>"

The "Student NNN" label encodes the student's row index (NNN = index+1), so the
mapping is exact even when two students share a name. We also sanity-check against
the internal output/results.json when present.
"""

import argparse
import json
from pathlib import Path

import data_mapping as dm

BASE = Path(__file__).parent
PUBLIC = BASE / "display_data" / "results.json"
INTERNAL = BASE / "output" / "results.json"
OUT = BASE / "private" / "roster.json"


def _label_to_index(label: str) -> int:
    # "Student 022" -> 21
    return int("".join(ch for ch in label if ch.isdigit())) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mentors", required=True)
    ap.add_argument("--students", required=True)
    args = ap.parse_args()

    students = dm.load_students(args.students).reset_index(drop=True)
    mentors = dm.load_mentors(args.mentors)
    # name -> email for mentors (pool + excluded; matched mentors are all consented)
    mentor_email = {}
    dupes = set()
    for _, r in mentors.iterrows():
        n = r["name"]
        if n in mentor_email and mentor_email[n] != r.get("email", ""):
            dupes.add(n)
        mentor_email[n] = r.get("email", "")

    pub = json.loads(PUBLIC.read_text())
    internal = json.loads(INTERNAL.read_text()) if INTERNAL.exists() else None

    def m_field(m, *keys):
        for k in keys:
            if k in m:
                return m[k]
        return ""

    matched, mismatches = {}, 0
    for i, m in enumerate(pub.get("matches", [])):
        label = m_field(m, "student", "mentee_name")
        j = _label_to_index(label)
        if not (0 <= j < len(students)):
            continue
        stu = students.iloc[j]
        # sanity check against the internal real-name export (same sort order)
        if internal and i < len(internal.get("matches", [])):
            if str(internal["matches"][i].get("mentee_name", "")).strip() != str(stu["name"]).strip():
                mismatches += 1
        mentor_name = m_field(m, "mentor", "mentor_name")
        matched[label] = {
            "student_name": stu["name"],
            "student_email": stu.get("email", ""),
            "mentor_email": mentor_email.get(mentor_name, ""),
        }

    unmatched = {}
    for label in pub.get("unmatched", []):
        j = _label_to_index(label)
        if 0 <= j < len(students):
            stu = students.iloc[j]
            unmatched[label] = {"student_name": stu["name"], "student_email": stu.get("email", "")}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"matched": matched, "unmatched": unmatched}, indent=2, default=str))
    print(f"Wrote {OUT} — {len(matched)} matched + {len(unmatched)} unmatched.")
    if mismatches:
        print(f"  WARNING: {mismatches} label/name sanity mismatches (check ordering).")
    if dupes:
        print(f"  NOTE: duplicate mentor names (email may be ambiguous): {sorted(dupes)}")


if __name__ == "__main__":
    main()
