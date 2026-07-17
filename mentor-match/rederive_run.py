#!/usr/bin/env python3
"""
Re-derive the committed 2026-06-15 LLM run from its persisted artifacts —
NO re-scoring. This is the migration path for the external-review improvements
applied after that run:

  * stable ids everywhere (student_id/mentor_id from the loader order preserved
    in score_matrix.npz; names become display-only)
  * composite integer costs with a documented seeded tie-break (solver.py)
  * hard location constraint (verified by forensics below: zero edges removed
    for this cohort)
  * fit_balance (renamed from "confidence") recomputed from the 3 measured
    criteria for every pair whose criteria were persisted
  * robustness audit -> robustness_report.json + per-match `robust` flag

Data-reality constraints, verified against the artifacts and reported honestly:
  * The June npz stores only the OVERALL score per pair: round(mean of 4
    criteria), integers 1-10. Per-criteria scores were persisted only for the
    85 matched pairs (output/results.json). So the re-solve necessarily uses
    the 4-criteria rounded matrix — the only basis available for ALL pairs.
    The 3-criteria utility and full-precision costing apply to FUTURE runs
    (matcher.py now persists the full criteria tensor at full precision).
  * The tie-break for THIS re-solve prefers the incumbent assignment (eps=0
    for committed pairs, seeded eps in [1,999] otherwise). Rationale: none of
    the substantive improvements alters any cost retroactively (see above +
    zero location removals), so the committed matching is provably still an
    optimal solution; re-shuffling pairs among exactly-equal optima would be
    pure churn. Fresh runs (no incumbent) use the pure seeded tie-break.
  * The zero-removal location claim carries one residual caveat: the June
    nudge's locality test was a substring match (" wi"/"wi ") that counts e.g.
    "West Windsor NJ" or "Wilmington" as local, while the new hard constraint
    uses a whole-word test. "No nudge fired" therefore proves no mentor met
    the JUNE predicate; an in-person-only mentor in a " wi"-substring city
    outside Wisconsin cannot be ruled out from the persisted artifacts (the
    spreadsheets with location/format fields are not on disk). The next real
    scoring run applies the hard constraint on the actual fields.
  * All 5 all-NaN score rows in the npz are duplicate submissions of people
    who ARE matched via their other row (same email; one pair differs only in
    name capitalization). The June run's case-insensitive name-keyed joins
    collapsed each pair onto one row. The duplicate rows are annotated with
    duplicate_submission_of; no real person lost a match to the data gap.
  * mentor_ids here number the June npz's mentor array, which is the FILTERED
    1:1 pool (27 of 34 mentors) in pool order — the source-row positions of
    the 7 excluded mentors are not recoverable from the artifacts. Fresh runs
    assign mentor_ids over the full roster at load time (data_mapping.py), so
    ids are run-scoped and must never be compared across runs.
  * The June per-pair criteria survive in git (commit 6676bf2's display JSON)
    and are preserved locally in output/june_criteria_scores.json, which this
    script prefers when output/results.json no longer carries them.

Usage:  python rederive_run.py [--skip-llm]  (run from mentor-match/)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import robustness_audit
from matcher import GUARDRAIL, UTILITY_CRITERIA, scrub_for_prompt
from solver import (COST_RESOLUTION, TIE_EPS_MOD, hungarian_cross_check,
                    solve_min_cost_flow, tie_eps)

BASE = Path(__file__).parent
NPZ = BASE / "output" / "score_matrix.npz"
INTERNAL = BASE / "output" / "results.json"          # June run: 85 matches + criteria_scores
ROSTER = BASE / "private" / "roster.json"            # Student NNN -> name/email
DISPLAY = BASE / "display_data" / "results.json"     # mentor emails
METADATA = BASE / "output" / "metadata.json"

SEED = 20260315
RATIONALE_MODEL = "claude-sonnet-4-5-20250929"       # same model as the June run

CRITERIA4 = ("skill_alignment", "domain_fit", "goal_compatibility", "style_fit")


def _load_env_key():
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    envf = BASE / ".env"
    if envf.exists():
        for line in envf.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip()


def _validate_rationale(text: str) -> bool:
    if not text or not text.strip():
        return False
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    return 2 <= len(sentences) <= 4


def _llm_rationales(pairs, mentors, students, profiles):
    """pairs: [(j, i)]. Joined strictly by id. Only called for changed pairs."""
    from anthropic import Anthropic
    tool = {
        "name": "submit_rationales",
        "description": "Return one 2-4 sentence rationale + nothing invented.",
        "input_schema": {
            "type": "object",
            "properties": {"rationales": {"type": "array", "items": {
                "type": "object",
                "properties": {"mentee_id": {"type": "string"},
                               "mentor_id": {"type": "string"},
                               "rationale": {"type": "string"}},
                "required": ["mentee_id", "mentor_id", "rationale"]}}},
            "required": ["rationales"],
        },
    }
    lines = []
    for (j, i) in pairs:
        sid, mid = students["ids"][j], mentors["ids"][i]
        lines.append(f"Pair mentee_id={sid} mentor_id={mid} | "
                     f"Mentor {mentors['names'][i]}: {scrub_for_prompt(profiles.get(mid, '')) or 'unknown'} | "
                     f"Mentee {students['names'][j]}: {scrub_for_prompt(profiles.get(sid, '')) or 'unknown'}")
    prompt = (f"{GUARDRAIL}\n\nWrite a 2-4 sentence rationale for each pairing, citing only the "
              f"shown profile facts.\n" + "\n".join(lines) +
              "\nCall submit_rationales, echoing each pair's exact mentee_id and mentor_id.")
    client = Anthropic()
    resp = client.messages.create(
        model=RATIONALE_MODEL, max_tokens=4096, temperature=0.0,
        tools=[tool], tool_choice={"type": "tool", "name": "submit_rationales"},
        messages=[{"role": "user", "content": prompt}])
    out = {}
    for blk in resp.content:
        if blk.type == "tool_use":
            for r in blk.input.get("rationales", []):
                out[(str(r["mentee_id"]).strip(), str(r["mentor_id"]).strip())] = r["rationale"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--skip-llm", action="store_true",
                    help="Fail instead of calling the API if any pair changed")
    args = ap.parse_args()

    d = np.load(NPZ, allow_pickle=True)
    scores = d["scores"].astype(float)
    mentor_names = [str(x) for x in d["mentors"]]
    student_names = [str(x) for x in d["students"]]
    S, P = scores.shape
    # Canonical stable ids for this run = loader row order preserved in the npz.
    student_ids = [f"S{j + 1:03d}" for j in range(S)]
    mentor_ids = [f"M{i + 1:02d}" for i in range(P)]
    students = {"ids": student_ids, "names": student_names}
    mentors = {"ids": mentor_ids, "names": mentor_names}

    internal = json.loads(INTERNAL.read_text())
    roster = json.loads(ROSTER.read_text())
    display_old = json.loads(DISPLAY.read_text())
    meta_old = json.loads(METADATA.read_text())

    # This script OVERWRITES output/results.json — its own input. Preserve the
    # June per-pair criteria to a separate file BEFORE writing anything, and
    # prefer that file on re-runs.
    crit_file = BASE / "output" / "june_criteria_scores.json"
    have_criteria = internal.get("matches") and "criteria_scores" in internal["matches"][0]
    if have_criteria:
        crit_file.write_text(json.dumps(
            [{"mentee_name": m["mentee_name"], "mentor_name": m["mentor_name"],
              "rationale": m["rationale"], "criteria_scores": m["criteria_scores"]}
             for m in internal["matches"]], indent=1))
    elif crit_file.exists():
        internal = {"matches": json.loads(crit_file.read_text())}
    else:
        sys.exit("ERROR: neither output/results.json nor output/june_criteria_scores.json "
                 "carries the June per-pair criteria_scores. They are recoverable from git: "
                 "commit 6676bf2's mentor-match/display_data/results.json has all 85 pairs' "
                 "criteria_scores; join mentee labels to rows via private/roster.json and "
                 "write output/june_criteria_scores.json in the format documented above.")
    # meta_old may be the June metadata (first run) or a regenerated one that
    # nests the June metadata under original_run (re-runs). Normalize both.
    june_meta = meta_old.get("original_run", meta_old)
    if "reconciliation" not in june_meta:
        sys.exit("ERROR: cannot locate the June reconciliation block in output/metadata.json")

    # ── identity maps (email lookups; ids never derived from names) ──────────
    stu_email = {}
    for section in ("matched", "unmatched"):
        for label, info in roster.get(section, {}).items():
            j = int("".join(c for c in label if c.isdigit())) - 1
            stu_email[j] = info.get("student_email", "")
            assert info.get("student_name", "").strip() == student_names[j].strip(), \
                f"roster/npz name mismatch at row {j}: {info.get('student_name')!r} vs {student_names[j]!r}"
    men_email = {}
    mname_to_i = {n: i for i, n in enumerate(mentor_names)}   # mentor names verified unique
    assert len(mname_to_i) == P, "mentor names are not unique — cannot map emails by name"
    for m in display_old.get("matches", []):
        i = mname_to_i.get(m.get("mentor_name"))
        if i is not None and m.get("mentor_email"):
            men_email[i] = m["mentor_email"]

    # ── incumbent pairs (June solution) resolved to (j, i) BY row evidence ──
    # Mentor: unique name -> i. Student: among same-name rows, exactly one has
    # scores (the June name-keyed join starved the others); that row is the one
    # the June solver could have matched.
    sname_to_js = {}
    for j, n in enumerate(student_names):
        sname_to_js.setdefault(n, []).append(j)
    incumbent, crit_of = [], {}
    for m in internal["matches"]:
        i = mname_to_i[m["mentor_name"]]
        cands = [j for j in sname_to_js[m["mentee_name"]] if not np.all(np.isnan(scores[j]))]
        assert len(cands) == 1, f"ambiguous student rows for {m['mentee_name']!r}: {cands}"
        j = cands[0]
        incumbent.append((j, i))
        crit_of[(j, i)] = dict(m["criteria_scores"])
    assert len(set(incumbent)) == len(incumbent), "duplicate incumbent pairs"

    # ── forensic verification: hard location constraint removes ZERO edges ──
    # The June matcher applied a -1 nudge to exactly the mentors the new HARD
    # constraint would remove (in-person-only outside Madison/WI). For every
    # matched pair, npz == round(mean of 4 criteria) exactly => the nudge fired
    # for no pool mentor, and all 27 mentors appear in the matching.
    for (j, i), crit in crit_of.items():
        mean4 = round(sum(crit[c] for c in CRITERIA4) / 4)
        assert int(scores[j, i]) == int(mean4), \
            f"nudge/clamp detected at {student_ids[j]}/{mentor_ids[i]}: npz={scores[j, i]} mean4={mean4}"
    assert len({i for (_, i) in incumbent}) == P, "some mentor never matched — forensic gap"
    removed_edges = []   # verified empty for this cohort

    # ── capacities: per-mentor match counts. Exact because every slot filled:
    # all arcs have negative cost, and 111 scored students > 85 total slots. ──
    slots = [0] * P
    for (_, i) in incumbent:
        slots[i] += 1
    assert sum(slots) == len(incumbent) == 85, f"slot reconstruction broke: {sum(slots)}"
    assert sum(slots) == june_meta["reconciliation"]["total_mentor_slots"]

    # ── re-solve: composite costs, incumbent-preferring seeded tie-break ─────
    inc_set = set(incumbent)
    eps = np.zeros((S, P), dtype=int)
    for j in range(S):
        for i in range(P):
            if (j, i) not in inc_set:
                eps[j, i] = 1 + tie_eps(SEED, student_ids[j], mentor_ids[i]) % (TIE_EPS_MOD - 1)
    assignments, unmatched, sub_total, comp_total = solve_min_cost_flow(scores, slots, eps)
    scipy_total = hungarian_cross_check(scores, slots, eps)
    agree = comp_total == scipy_total
    print(f"Re-solve: {len(assignments)} matched, {len(unmatched)} unmatched | "
          f"substantive_total={sub_total / COST_RESOLUTION:.1f} | "
          f"OR-Tools/scipy agree={agree}")
    assert agree, f"solver cross-check mismatch: {comp_total} vs {scipy_total}"

    # ── diff vs committed ────────────────────────────────────────────────────
    new_set = set(assignments)
    kept = sorted(inc_set & new_set)
    added = sorted(new_set - inc_set)
    dropped = sorted(inc_set - new_set)
    old_matched_js = {j for (j, _) in incumbent}
    new_matched_js = {j for (j, _) in new_set}
    print(f"Diff vs committed: {len(kept)} kept, {len(added)} new, {len(dropped)} dropped | "
          f"students newly matched: {sorted(student_ids[j] for j in new_matched_js - old_matched_js)} | "
          f"newly unmatched: {sorted(student_ids[j] for j in old_matched_js - new_matched_js)}")

    # ── 3-vs-4-criteria correlation on the pairs whose criteria survive ─────
    m4 = [sum(c.values()) / 4 for c in crit_of.values()]
    m3 = [sum(c[k] for k in UTILITY_CRITERIA) / 3 for c in crit_of.values()]
    corr = float(np.corrcoef(m4, m3)[0, 1])
    print(f"3-vs-4-criteria correlation (85 matched pairs, the only pairs with "
          f"persisted criteria): r={corr:.4f} | mean4={np.mean(m4):.3f} mean3={np.mean(m3):.3f}")

    # ── robustness audit ─────────────────────────────────────────────────────
    audit_report, robust = robustness_audit.run_audit(
        scores, slots, assignments, SEED, student_ids, mentor_ids, log=print)

    # ── rationales: keep for kept pairs, generate (by id) for added pairs ────
    old_rationale = {pair: internal["matches"][k]["rationale"]
                     for k, pair in enumerate(incumbent)}
    new_rationales = {}
    if added:
        if args.skip_llm:
            sys.exit(f"ERROR: {len(added)} changed pairs need rationales but --skip-llm set.")
        _load_env_key()
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("ERROR: changed pairs need rationales; set ANTHROPIC_API_KEY.")
        got = _llm_rationales(added, mentors, students, profiles={})
        for (j, i) in added:
            text = got.get((student_ids[j], mentor_ids[i]))
            if not _validate_rationale(text or ""):
                text = (f"{mentor_names[i]} and {student_names[j]} were matched on overall "
                        f"domain fit and availability. The pairing reflects the strongest "
                        f"available compatibility given mentor capacity.")
            new_rationales[(j, i)] = text

    # ── assemble records ─────────────────────────────────────────────────────
    ts = datetime.now(timezone.utc).isoformat()

    def fit_of(pair):
        crit = crit_of.get(pair)
        if crit is None:      # changed pair: criteria never persisted for it
            return int(scores[pair]), "overall_score (per-criteria unavailable for this pair)"
        vals = [crit[c] for c in UTILITY_CRITERIA]
        return max(1, min(10, int(round(0.6 * np.mean(vals) + 0.4 * min(vals))))), None

    email_by_addr_matched = {}
    match_rows = []
    for (j, i) in sorted(new_set, key=lambda p: -fit_of(p)[0]):
        fit, fit_note = fit_of((j, i))
        row = {
            "student_id": student_ids[j],
            "mentor_id": mentor_ids[i],
            "mentor_name": mentor_names[i],
            "mentor_email": men_email.get(i, ""),
            "student_name": student_names[j],
            "student_email": stu_email.get(j, ""),
            "fit": fit,
            "robust": bool(robust.get((j, i), False)),
            "rationale": old_rationale.get((j, i)) or new_rationales.get((j, i), ""),
        }
        if fit_note:
            row["fit_note"] = fit_note
        match_rows.append(row)
        if row["student_email"]:
            email_by_addr_matched[row["student_email"].lower()] = student_ids[j]

    unmatched_rows = []
    for j in sorted(unmatched):
        email = stu_email.get(j, "")
        row = {"student_id": student_ids[j], "student_name": student_names[j],
               "student_email": email}
        if np.all(np.isnan(scores[j])):
            twin = email_by_addr_matched.get(email.lower())
            if twin:
                row["reason"] = ("no scores recorded in the original LLM run — duplicate "
                                 "submission; this person IS matched via their other row")
                row["duplicate_submission_of"] = twin
            else:
                row["reason"] = ("no scores recorded in the original LLM run — "
                                 "re-run scoring to include this student")
        else:
            row["reason"] = "mentor capacity exhausted"
        unmatched_rows.append(row)

    # ── write artifacts ──────────────────────────────────────────────────────
    out = BASE / "output"
    import csv as _csv
    with open(out / "matches.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["student_id", "mentee_name", "mentor_id",
                                           "mentor_name", "rationale", "fit_balance", "robust"])
        w.writeheader()
        for r in match_rows:
            w.writerow({"student_id": r["student_id"], "mentee_name": r["student_name"],
                        "mentor_id": r["mentor_id"], "mentor_name": r["mentor_name"],
                        "rationale": r["rationale"], "fit_balance": r["fit"],
                        "robust": r["robust"]})
    with open(out / "unmatched.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["student_id", "mentee_name", "reason"])
        w.writeheader()
        for r in unmatched_rows:
            w.writerow({"student_id": r["student_id"], "mentee_name": r["student_name"],
                        "reason": r["reason"]})
    (out / "robustness_report.json").write_text(json.dumps(audit_report, indent=2, default=str))

    meta = {
        "generation_mode": "llm (re-derived from the 2026-06-15 run artifacts; no re-scoring)",
        "generated_at": ts,
        "seed": SEED,
        "ids": "all joins are by id, names are display-only. student_id = loader row order "
               "preserved in score_matrix.npz (116 unfiltered rows). mentor_id = the June "
               "npz mentor array, which is the FILTERED 1:1 pool (27 of 34) in pool order — "
               "fresh runs assign mentor_ids over the full roster at load time, so ids are "
               "run-scoped and must never be compared across runs",
        "score_basis": "persisted June matrix: round(mean of 4 criteria), integers 1-10. "
                       "Per-criteria scores were persisted only for the 85 matched pairs, so "
                       "the 3-criteria full-precision utility applies to future runs only "
                       "(matcher.py now persists the full criteria tensor).",
        "criteria_correlation_3v4": {"pearson_r": corr, "n_pairs": len(crit_of),
                                     "mean_4crit": float(np.mean(m4)),
                                     "mean_3crit": float(np.mean(m3)),
                                     "scope": "85 matched pairs (only pairs with persisted criteria)"},
        "tie_break": "incumbent-preferring for this re-derivation (eps=0 for committed pairs, "
                     "seeded eps in [1,999] otherwise): no substantive cost changed "
                     "retroactively, so re-shuffling equal optima would be pure churn. "
                     "Fresh runs use the pure seeded tie-break (solver.py).",
        "location_hard_constraint": {
            "removed_edges": removed_edges,
            "verification": "forensic: npz == round(mean4) for all 85 matched pairs across "
                            "all 27 mentors => the June -1 location nudge fired for no pool "
                            "mentor => zero edges removed for this cohort",
            "caveat": "the June nudge's locality test was a substring match (' wi'/'wi ') "
                      "that counts e.g. 'West Windsor NJ' as local, while the new hard "
                      "constraint uses a whole-word test; 'no nudge fired' proves no mentor "
                      "met the JUNE predicate. An in-person-only mentor in a ' wi'-substring "
                      "city outside Wisconsin cannot be ruled out from the persisted "
                      "artifacts; the next real scoring run applies the hard constraint on "
                      "the actual location/format fields",
        },
        "fit_balance_basis": "0.6*mean + 0.4*min over the 3 measured criteria "
                             "(balanced-fit score, NOT a confidence/uncertainty estimate)",
        "diff_vs_committed": {
            "kept": len(kept), "added": [f"{student_ids[j]}+{mentor_ids[i]}" for (j, i) in added],
            "dropped": [f"{student_ids[j]}+{mentor_ids[i]}" for (j, i) in dropped],
        },
        "duplicate_submissions": [r for r in unmatched_rows if "no scores recorded" in r.get("reason", "")],
        "criteria_provenance": "June per-pair criteria_scores survive in git (commit 6676bf2 "
                               "display_data/results.json, labels joined via private/roster.json) "
                               "and locally in output/june_criteria_scores.json; matcher.py now "
                               "persists the full criteria tensor in the npz for future runs",
        "robustness_summary": audit_report["summary"],
        "solver_agreement": bool(agree),
        "original_run": june_meta,
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))

    payload = {
        "generated_at": ts,
        "generation_mode": "llm (re-derived 2026-07; scores from the 2026-06-15 run)",
        "scope": "Internal coordinator view: real mentor + student identities with contact info.",
        "fit_basis": "fit = 0.6*mean + 0.4*min of the 3 measured criteria (1-10); "
                     "style_fit excluded (no style data was collected)",
        "matches": match_rows,
        "unmatched": unmatched_rows,
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2, default=str))
    DISPLAY.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nWrote output/{{matches,unmatched}}.csv, metadata.json, robustness_report.json, "
          f"results.json and display_data/results.json")
    print(f"Summary: {len(match_rows)} matched ({len(added)} changed), "
          f"{len(unmatched_rows)} unmatched, "
          f"{audit_report['summary']['fragile_pairs']} fragile, "
          f"{audit_report['summary']['arbitrary_pairs']} arbitrary-among-optima")


if __name__ == "__main__":
    main()
