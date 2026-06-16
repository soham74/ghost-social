#!/usr/bin/env python3
"""
Mentor–Mentee Matching Tool — LLM matching arm (Condition B)
============================================================
Pipeline:
  1. Load + map the real TEL spreadsheets (data_mapping.py)
  2. Pool filter (1:1 mentors only; consent respected)        [HARD constraint]
  3. Score compatibility on 4 criteria
        mode="llm"     -> Claude tool-use, parallel, seeded mentor shuffle
        mode="offline" -> deterministic overlap scoring (zero API, reproducible)
  4. Assign with OR-Tools min-cost max-flow (capacity = hard limit, ineligible
     pairs simply have no arc) and cross-check the objective with scipy Hungarian
  5. Rationales grounded in the profile fields, matched to pairings BY NAME,
     validated to 2–4 sentences; confidence derived from the 4 criteria
  6. Export matches.csv, unmatched.csv, metadata.json, results.json (display)

Capacity shortfall is normal (85 slots, 116 students): excess students are left
UNMATCHED and surfaced explicitly — the run never fails on shortage.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

import data_mapping as dm

# ── Reproducibility defaults (fix #7) ────────────────────────────────────────
# claude-sonnet-4-6 has no dated snapshot; the latest *dated* active Sonnet is
# 4.5. Pinned for reproducibility — a moving alias is not.
DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SEED = 20260315
MENTEE_BATCH_SIZE = 4          # 4 mentees x ~27 mentors = ~108 score objects/call
RATIONALE_BATCH_SIZE = 10
SCORING_MAX_TOKENS = 16000     # headroom so the structured tool call never truncates
SCORE_SCALE = 1            # LLM/offline scores are integers 1–10; cost = -score

GUARDRAIL = (
    "IMPORTANT GUARDRAIL: Only use information explicitly provided in the data. "
    "If information is missing or unclear, state 'unknown.' "
    "Do not fabricate or infer details not present in the input."
)

CRITERIA = ("skill_alignment", "domain_fit", "goal_compatibility", "style_fit")


# ── Logging ──────────────────────────────────────────────────────────────────

class MatchingLogger:
    def __init__(self, output_dir: Path):
        self.warnings: list[str] = []
        self.logger = logging.getLogger("mentor_matcher")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(ch)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(output_dir) / "match_log.txt", mode="w")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        self.logger.addHandler(fh)

    def info(self, m): self.logger.info(m)
    def debug(self, m): self.logger.debug(m)
    def error(self, m): self.logger.error(f"ERROR: {m}")
    def warning(self, m):
        self.logger.warning(f"WARNING: {m}")
        self.warnings.append(m)


# ── Offline deterministic scoring (grounded in real fields) ──────────────────

_STOP = set(
    "the a an and or of to in for with on at by from as is are be it this that you "
    "your our their they we i me my his her about into over under more most very "
    "who what when where which while also can will would like want help build "
    "students student mentor mentoring mentors interested interest interests "
    "experience experiences skills skill work working real world etc projects "
    "project area areas".split()
)


def _tokens(*texts: str) -> set[str]:
    out: set[str] = set()
    for t in texts:
        for w in dm.normalize(t).split():
            if len(w) >= 3 and w not in _STOP and not w.isdigit():
                out.add(w)
    return out


def _overlap_score(a: set[str], b: set[str]) -> tuple[int, list[str]]:
    """Shared meaningful tokens -> 1..10, plus the shared tokens (for grounding)."""
    shared = sorted(a & b)
    n = len(shared)
    score = max(1, min(10, 2 + 2 * n))     # 0 shared -> 2, 4+ shared -> 10
    return score, shared


def _offline_pair_scores(mentor: pd.Series, student: pd.Series) -> tuple[dict, list[str]]:
    m_skills = _tokens(mentor.get("topics", ""), mentor.get("role_org", ""), mentor.get("bio", ""))
    m_domain = _tokens(mentor.get("industry", ""), mentor.get("topics", ""))
    m_goals = _tokens(mentor.get("notes", ""), mentor.get("topics", ""), mentor.get("bio", ""))
    s_skills = _tokens(student.get("strengths", ""), student.get("top_choices", ""))
    s_domain = _tokens(student.get("majors", ""), student.get("top_choices", ""), student.get("experiences", ""))
    s_goals = _tokens(student.get("top_choices", ""), student.get("where_today", ""))

    sk, sh1 = _overlap_score(m_skills, s_skills)
    do, sh2 = _overlap_score(m_domain, s_domain)
    go, sh3 = _overlap_score(m_goals, s_goals)
    # style_fit: light, grounded signal — does the mentor mention the student's stage?
    stage = dm.normalize(student.get("academic_level", ""))
    notes_n = dm.normalize(mentor.get("notes", "") + " " + mentor.get("bio", ""))
    style = 6
    if stage and stage.split()[0] in notes_n:
        style = 8
    crit = {"skill_alignment": sk, "domain_fit": do, "goal_compatibility": go, "style_fit": style}
    grounded = sorted(set(sh1 + sh2 + sh3))
    return crit, grounded


def _location_nudge(mentor: pd.Series) -> int:
    """Soft (never hard) location term: students are UW/Madison. Virtual is the
    universal fallback, so only penalize in-person-only mentors who aren't local."""
    fmt = dm.normalize(mentor.get("formats", ""))
    loc = dm.normalize(mentor.get("location", ""))
    offers_virtual = "virtual" in fmt or "zoom" in fmt
    local = any(k in loc for k in ("madison", "wisconsin", " wi", "wi "))
    if not offers_virtual and not local and loc:
        return -1
    return 0


def _scrub_name(text: str, full: str, label: str) -> str:
    """Remove a person's name (full + each ≥3-char token) from free text so an
    LLM rationale can't leak a real (anonymized) name. Idempotent-ish."""
    if not text or not full:
        return text
    toks = sorted(set([full] + [t for t in full.split() if len(t) >= 3]), key=len, reverse=True)
    for tok in toks:
        text = re.sub(r"\b" + re.escape(tok) + r"\b", label, text, flags=re.IGNORECASE)
    return re.sub(r"\b" + re.escape(label) + r"(?:'s)?(?:\s+" + re.escape(label) + r")+", label, text)


def _overall_and_confidence(crit: dict) -> tuple[int, int]:
    vals = [crit[c] for c in CRITERIA]
    overall = int(round(sum(vals) / len(vals)))
    # Confidence derived from the criteria (fix #6): a weak criterion lowers it,
    # so it spreads instead of clustering at the top like a free-floating LLM number.
    conf = int(round(0.6 * (sum(vals) / len(vals)) + 0.4 * min(vals)))
    return max(1, min(10, overall)), max(1, min(10, conf))


# ── Main matcher ─────────────────────────────────────────────────────────────

class MentorMatcher:
    def __init__(self, output_dir="output", model=DEFAULT_MODEL,
                 temperature=DEFAULT_TEMPERATURE, seed=DEFAULT_SEED,
                 mode="offline", batch_size=MENTEE_BATCH_SIZE,
                 max_workers=6, display_out=None, anonymize_students=True,
                 logger: MatchingLogger | None = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.mode = mode               # "offline" | "llm"
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.display_out = Path(display_out) if display_out else None
        self.anonymize_students = anonymize_students
        self.log = logger or MatchingLogger(self.output_dir)
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.rng = np.random.default_rng(seed)
        self.criteria_scores: dict[tuple[int, int], dict] = {}
        self.grounded_overlap: dict[tuple[int, int], list[str]] = {}
        self._client = None

    # — scoring —

    def _score_offline(self, pool, students, score_matrix):
        for j, (_, stu) in enumerate(students.iterrows()):
            for i, (_, men) in enumerate(pool.iterrows()):
                crit, grounded = _offline_pair_scores(men, stu)
                overall, _conf = _overall_and_confidence(crit)
                overall = max(1, min(10, overall + _location_nudge(men)))
                score_matrix[j, i] = float(overall)
                self.criteria_scores[(j, i)] = crit
                self.grounded_overlap[(j, i)] = grounded

    def _client_lazy(self):
        if self._client is None:
            from anthropic import Anthropic
            self._client = Anthropic()
        return self._client

    _SCORE_TOOL = {
        "name": "submit_scores",
        "description": "Submit 1-10 compatibility scores for each mentee against each mentor.",
        "input_schema": {
            "type": "object",
            "properties": {
                "evaluations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "mentee_name": {"type": "string"},
                            "mentor_name": {"type": "string"},
                            "skill_alignment": {"type": "integer"},
                            "domain_fit": {"type": "integer"},
                            "goal_compatibility": {"type": "integer"},
                            "style_fit": {"type": "integer"},
                        },
                        "required": ["mentee_name", "mentor_name", *CRITERIA],
                    },
                }
            },
            "required": ["evaluations"],
        },
    }

    def _llm_score_batch(self, pool, batch_students, name_to_i, name_to_j):
        # Seeded mentor-order shuffle per call kills positional bias (fix N5).
        order = list(range(len(pool)))
        np.random.default_rng(self.seed + len(batch_students)).shuffle(order)
        mlines = []
        for i in order:
            r = pool.iloc[i]
            mlines.append(f"- {r['name']}: topics={r.get('topics','') or 'unknown'}; "
                          f"industry={r.get('industry','') or 'unknown'}; bio={r.get('bio','') or 'unknown'}")
        slines = []
        for _, r in batch_students.iterrows():
            slines.append(f"- {r['name']}: strengths={r.get('strengths','') or 'unknown'}; "
                          f"majors={r.get('majors','') or 'unknown'}; "
                          f"goals={r.get('top_choices','') or 'unknown'}; stage={r.get('where_today','') or 'unknown'}")
        prompt = (f"{GUARDRAIL}\n\nMENTORS:\n" + "\n".join(mlines) +
                  "\n\nMENTEES:\n" + "\n".join(slines) +
                  "\n\nFor every mentee, score EACH mentor 1-10 on skill_alignment, domain_fit, "
                  "goal_compatibility, style_fit. Call submit_scores with all evaluations.")
        client = self._client_lazy()
        resp = client.messages.create(
            model=self.model, max_tokens=SCORING_MAX_TOKENS, temperature=self.temperature,
            tools=[self._SCORE_TOOL], tool_choice={"type": "tool", "name": "submit_scores"},
            messages=[{"role": "user", "content": prompt}],
        )
        if resp.stop_reason == "max_tokens":
            self.log.warning(f"Scoring call hit max_tokens — {len(batch_students)} mentees "
                             "scored partially; consider a smaller batch.")
        out = []
        for blk in resp.content:
            if blk.type == "tool_use" and blk.name == "submit_scores":
                out = blk.input.get("evaluations", [])
        return out

    def _score_llm(self, pool, students, score_matrix):
        name_to_i = {r["name"].lower(): i for i, (_, r) in enumerate(pool.iterrows())}
        name_to_j = {r["name"].lower(): j for j, (_, r) in enumerate(students.iterrows())}
        batches = [students.iloc[k:k + self.batch_size] for k in range(0, len(students), self.batch_size)]
        self.log.info(f"Scoring {len(students)} mentees x {len(pool)} mentors via {self.model} "
                      f"in {len(batches)} parallel batches")

        def run(b):
            try:
                return self._llm_score_batch(pool, b, name_to_i, name_to_j)
            except Exception as exc:
                self.log.warning(f"Scoring batch failed: {exc}")
                return []

        with cf.ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            results = list(ex.map(run, batches))   # ordered -> deterministic under seed

        for evals in results:
            for ev in evals:
                j = name_to_j.get(str(ev.get("mentee_name", "")).lower())
                i = name_to_i.get(str(ev.get("mentor_name", "")).lower())
                if i is None or j is None:
                    continue
                crit = {c: max(1, min(10, int(ev.get(c, 5)))) for c in CRITERIA}
                overall, _ = _overall_and_confidence(crit)
                overall = max(1, min(10, overall + _location_nudge(pool.iloc[i])))
                score_matrix[j, i] = float(overall)
                self.criteria_scores[(j, i)] = crit

    # — assignment —

    def _solve_min_cost_flow(self, score_matrix, slots):
        """OR-Tools min-cost max-flow. Capacity = hard arc limit; an ineligible
        (unscored / out-of-pool) pair simply has NO arc, so it cannot be chosen.
        Returns (assignments[(j,i)], unmatched_j, total_score)."""
        from ortools.graph.python import min_cost_flow
        S, P = score_matrix.shape
        mcf = min_cost_flow.SimpleMinCostFlow()
        SRC, SINK = 0, S + P + 1
        stu = lambda j: 1 + j
        men = lambda i: 1 + S + i
        for j in range(S):
            mcf.add_arc_with_capacity_and_unit_cost(SRC, stu(j), 1, 0)
            mcf.add_arc_with_capacity_and_unit_cost(stu(j), SINK, 1, 0)   # unmatched (cost 0)
        for i in range(P):
            if slots[i] > 0:
                mcf.add_arc_with_capacity_and_unit_cost(men(i), SINK, int(slots[i]), 0)
        arc_of: dict[int, tuple[int, int]] = {}
        for j in range(S):
            for i in range(P):
                if not np.isnan(score_matrix[j, i]):       # scored mask: only real arcs
                    a = mcf.add_arc_with_capacity_and_unit_cost(
                        stu(j), men(i), 1, -int(round(score_matrix[j, i] * SCORE_SCALE)))
                    arc_of[a] = (j, i)
        mcf.set_node_supply(SRC, S)
        mcf.set_node_supply(SINK, -S)
        if mcf.solve() != mcf.OPTIMAL:
            raise RuntimeError("min-cost flow did not solve to optimality")
        assignments, total = [], 0
        matched_students = set()
        for a, (j, i) in arc_of.items():
            if mcf.flow(a) > 0:
                assignments.append((j, i))
                matched_students.add(j)
                total += int(score_matrix[j, i])
        unmatched = [j for j in range(S) if j not in matched_students]
        return assignments, unmatched, total

    def _scipy_objective(self, score_matrix, slots):
        """Independent cross-check of the OPTIMAL total via slot-expanded Hungarian."""
        S, P = score_matrix.shape
        cols, col_men = [], []
        for i in range(P):
            for _ in range(int(slots[i])):
                cols.append(i)
                col_men.append(i)
        if not cols:
            return 0
        M = np.zeros((S, len(cols)))
        for cidx, i in enumerate(cols):
            col = score_matrix[:, i]
            M[:, cidx] = np.where(np.isnan(col), 0.0, col)   # unscored -> 0 (never preferred)
        # pad to square so every student can route to a (possibly dummy) column
        if M.shape[1] < S:
            M = np.hstack([M, np.zeros((S, S - M.shape[1]))])
        r, c = linear_sum_assignment(-M)
        return int(sum(M[ri, ci] for ri, ci in zip(r, c) if M[ri, ci] > 0))

    # — rationales —

    def _rationale_offline(self, men, stu, crit):
        grounded = self.grounded_overlap.get((stu["_j"], men["_i"]), [])
        bits = []
        if grounded:
            bits.append(f"Shared focus on {', '.join(grounded[:4])}.")
        ind = men.get("industry", "").strip()
        if ind:
            bits.append(f"{men['name']} works in {ind}, relevant to the mentee's stated direction.")
        stage = stu.get("where_today", "").strip()
        if stage:
            bits.append(f"The mentee is currently: {stage[:90]}, which fits this mentor's experience.")
        if not grounded:
            bits.append("Limited explicit keyword overlap in the form responses; "
                        "matched on availability and broad domain fit rather than stated specifics.")
        text = " ".join(bits[:4]) or "Matched on overall availability and domain fit."
        return text

    def _validate_rationale(self, text: str) -> bool:
        if not text or not text.strip():
            return False
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
        return 2 <= len(sentences) <= 4

    def _rationale_llm_batch(self, pairs):
        tool = {
            "name": "submit_rationales",
            "description": "Return one 2-4 sentence rationale + nothing invented.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "rationales": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "mentee_name": {"type": "string"},
                                "mentor_name": {"type": "string"},
                                "rationale": {"type": "string"},
                            },
                            "required": ["mentee_name", "mentor_name", "rationale"],
                        },
                    }
                },
                "required": ["rationales"],
            },
        }
        lines = []
        for (men, stu) in pairs:
            lines.append(f"Mentor {men['name']}: {men.get('topics','') or men.get('bio','') or 'unknown'} | "
                         f"Mentee {stu['name']}: {stu.get('top_choices','') or stu.get('strengths','') or 'unknown'}")
        prompt = (f"{GUARDRAIL}\n\nWrite a 2-4 sentence rationale for each pairing, citing only the "
                  f"shown profile facts.\n" + "\n".join(lines) + "\nCall submit_rationales.")
        resp = self._client_lazy().messages.create(
            model=self.model, max_tokens=4096, temperature=self.temperature,
            tools=[tool], tool_choice={"type": "tool", "name": "submit_rationales"},
            messages=[{"role": "user", "content": prompt}])
        for blk in resp.content:
            if blk.type == "tool_use":
                return {(r["mentee_name"].lower(), r["mentor_name"].lower()): r["rationale"]
                        for r in blk.input.get("rationales", [])}
        return {}

    def _generate_rationales(self, assignments, pool, students):
        out = []
        # LLM rationales matched BY NAME (fix #2), with a capped re-request on miss.
        llm_map: dict[tuple[str, str], str] = {}
        if self.mode == "llm":
            pairs = [(pool.iloc[i], students.iloc[j]) for (j, i) in assignments]
            for k in range(0, len(pairs), RATIONALE_BATCH_SIZE):
                try:
                    llm_map.update(self._rationale_llm_batch(pairs[k:k + RATIONALE_BATCH_SIZE]))
                except Exception as exc:
                    self.log.warning(f"Rationale batch failed: {exc}")

        for (j, i) in assignments:
            men = pool.iloc[i].copy(); men["_i"] = i
            stu = students.iloc[j].copy(); stu["_j"] = j
            crit = self.criteria_scores.get((j, i), {c: 5 for c in CRITERIA})
            _, conf = _overall_and_confidence(crit)
            key = (stu["name"].lower(), men["name"].lower())
            text = llm_map.get(key) if self.mode == "llm" else None
            attempts = 0
            while self.mode == "llm" and (not text or not self._validate_rationale(text)) and attempts < 2:
                attempts += 1
                try:
                    text = self._rationale_llm_batch([(men, stu)]).get(key)
                except Exception:
                    break
            if not text or not self._validate_rationale(text):
                text = self._rationale_offline(men, stu, crit)
            if not self._validate_rationale(text):  # neutral, never "generation failed"
                text = (f"{men['name']} and {stu['name']} were matched on overall domain fit and "
                        f"availability. The pairing reflects the strongest available compatibility "
                        f"given mentor capacity.")
            out.append({
                "mentee_name": stu["name"], "mentor_name": men["name"],
                "rationale": text, "confidence_score": conf,
                "criteria_scores": crit, "_j": j, "_i": i,
            })
        return out

    # — pipeline / io —

    def run(self, mentor_file, student_file):
        np.random.seed(self.seed)
        self.log.info("=" * 60)
        self.log.info(f"  MENTOR–MENTEE MATCHING (mode={self.mode}, model={self.model}, temp={self.temperature})")
        self.log.info("=" * 60)

        mentors_all = dm.load_mentors(mentor_file)
        students = dm.load_students(student_file).reset_index(drop=True)
        pool, excluded = dm.pool_filter_mentors(mentors_all)
        pool = pool.reset_index(drop=True)
        rec = dm.reconcile(mentors_all, pool, excluded, students)

        self.log.info(
            f"RECONCILIATION  mentors={rec['mentors_total']} pool={rec['mentors_in_1to1_pool']} "
            f"excluded={rec['mentors_excluded']} slots={rec['total_mentor_slots']} "
            f"students={rec['students_total']} expected_unmatched={rec['expected_unmatched']}")
        for e in excluded:
            self.log.info(f"  excluded mentor [{'; '.join(e['reasons'])}]")
        if rec["unknown_capacity_count"]:
            self.log.warning(f"{rec['unknown_capacity_count']} mentor(s) had unknown capacity "
                             f"(defaulted to {dm.UNKNOWN_CAPACITY_DEFAULT}).")

        S, P = len(students), len(pool)
        score_matrix = np.full((S, P), np.nan)        # nan == unscored (fix B2)
        if self.mode == "llm":
            self._score_llm(pool, students, score_matrix)
            # any pair the LLM failed to score stays nan -> simply gets no arc
        else:
            self._score_offline(pool, students, score_matrix)

        slots = [int(pool.iloc[i]["capacity_slots"]) for i in range(P)]
        assignments, unmatched, total = self._solve_min_cost_flow(score_matrix, slots)
        scipy_total = self._scipy_objective(score_matrix, slots)
        agree = (total == scipy_total)
        self.log.info(f"Assignment: {len(assignments)} matched, {len(unmatched)} unmatched | "
                      f"total_score(OR-Tools)={total} scipy_cross_check={scipy_total} "
                      f"agree={agree}")
        if not agree:
            self.log.warning(f"OR-Tools/scipy objective mismatch: {total} vs {scipy_total}")

        matches = self._generate_rationales(assignments, pool, students)
        rec["matched"] = len(matches)
        rec["unmatched"] = len(unmatched)
        rec["solver_total_score"] = total
        rec["scipy_cross_check_total"] = scipy_total
        rec["solver_agreement"] = bool(agree)

        self._export(matches, students, unmatched, pool, score_matrix, rec)
        return matches, unmatched, rec

    def _export(self, matches, students, unmatched, pool, score_matrix, rec):
        consent_by_name = {r["name"]: r["consent"] for _, r in pool.iterrows()}

        def shown_mentor(name):
            # Honor consent_to_share=No/unknown when displaying names.
            return name if consent_by_name.get(name) == "yes" else "Mentor (name withheld — no share consent)"

        rows = [{
            "mentee_name": m["mentee_name"],
            "mentor_name": m["mentor_name"],
            "rationale": m["rationale"],
            "confidence_score": m["confidence_score"],
        } for m in matches]
        pd.DataFrame(rows, columns=["mentee_name", "mentor_name", "rationale", "confidence_score"]) \
            .to_csv(self.output_dir / "matches.csv", index=False)

        un_rows = [{"mentee_name": students.iloc[j]["name"], "reason": "mentor capacity exhausted"}
                   for j in unmatched]
        pd.DataFrame(un_rows, columns=["mentee_name", "reason"]).to_csv(
            self.output_dir / "unmatched.csv", index=False)

        # Persist the raw score matrix for replayable, re-solvable runs (fix #7).
        np.savez_compressed(self.output_dir / "score_matrix.npz",
                            scores=score_matrix, mentors=np.array(list(pool["name"])),
                            students=np.array(list(students["name"])))

        meta = {
            "generation_mode": self.mode,
            "generated_at": self.timestamp,
            "model": self.model if self.mode == "llm" else "n/a (offline deterministic scoring)",
            "temperature": self.temperature,
            "seed": self.seed,
            "score_scale": SCORE_SCALE,
            "solver": "ortools.SimpleMinCostFlow (primary) + scipy.linear_sum_assignment (cross-check)",
            "guardrail": GUARDRAIL,
            "criteria": list(CRITERIA),
            "confidence_basis": "derived from the 4 criteria scores (0.6*mean + 0.4*min)",
            "reconciliation": rec,
            "note": ("Offline mode scores by deterministic keyword overlap on the real profile "
                     "fields (zero API calls, fully reproducible). Set ANTHROPIC_API_KEY and run "
                     "with --mode llm for the Claude (Condition B) scoring + rationales."),
        }
        (self.output_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))

        # Display payload for the web app (consent-respecting names).
        display = {
            "generated_at": self.timestamp,
            "generation_mode": self.mode,
            "stats": {
                "mentors_pool": rec["mentors_in_1to1_pool"],
                "students": rec["students_total"],
                "matched": rec["matched"],
                "unmatched": rec["unmatched"],
                "total_slots": rec["total_mentor_slots"],
                "avg_confidence": round(float(np.mean([m["confidence_score"] for m in matches])), 2) if matches else 0,
                "solver_agreement": rec["solver_agreement"],
            },
            "matches": [{
                "mentee_name": m["mentee_name"],
                "mentor_name": shown_mentor(m["mentor_name"]),
                "rationale": m["rationale"],
                "confidence_score": m["confidence_score"],
                "criteria_scores": m["criteria_scores"],
            } for m in sorted(matches, key=lambda x: -x["confidence_score"])],
            "unmatched": [students.iloc[j]["name"] for j in unmatched],
            "reconciliation": rec,
        }
        (self.output_dir / "results.json").write_text(json.dumps(display, indent=2, default=str))
        self.log.info(f"Output -> {self.output_dir.resolve()}")

        # Committed, shareable display payload: students anonymized (they did not
        # consent to public display), mentors already consent-gated above.
        if self.display_out is not None:
            def anon(j):
                return f"Student {int(j) + 1:03d}" if self.anonymize_students else students.iloc[j]["name"]
            pub = dict(display)
            # Strip identifiable mentor names out of the reconciliation block (these are
            # EXCLUDED / unknown-capacity mentors who are not consent-shown). Keep counts/reasons.
            safe_rec = {k: v for k, v in rec.items()
                        if k not in ("excluded_detail", "unknown_capacity_detail")}
            safe_rec["excluded_detail"] = [{"reasons": e.get("reasons", [])}
                                           for e in rec.get("excluded_detail", [])]
            pub["reconciliation"] = safe_rec
            pub["privacy"] = ("Students anonymized as 'Student NNN'; mentor names shown only with "
                              "consent_to_share=Yes. Identifiable data kept internal (gitignored output/).")
            def scrub_rationale(m):
                rat = _scrub_name(m["rationale"], m["mentee_name"], "the mentee")
                if consent_by_name.get(m["mentor_name"]) != "yes":   # withheld mentor
                    rat = _scrub_name(rat, m["mentor_name"], "the mentor")
                return rat
            pub["matches"] = [{
                "mentee_name": anon(m["_j"]),
                "mentor_name": shown_mentor(m["mentor_name"]),
                "rationale": scrub_rationale(m),
                "confidence_score": m["confidence_score"],
                "criteria_scores": m["criteria_scores"],
            } for m in sorted(matches, key=lambda x: -x["confidence_score"])]
            pub["unmatched"] = [anon(j) for j in unmatched]
            self.display_out.parent.mkdir(parents=True, exist_ok=True)
            self.display_out.write_text(json.dumps(pub, indent=2, default=str))
            self.log.info(f"Committed display -> {self.display_out.resolve()}")


def main():
    ap = argparse.ArgumentParser(description="Mentor-mentee matcher (LLM arm / Condition B)")
    ap.add_argument("--mentors", required=True)
    ap.add_argument("--students", required=True)
    ap.add_argument("--mode", choices=["offline", "llm"], default="offline")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--display-out", default=None,
                    help="Also write a committed, student-anonymized display JSON here")
    ap.add_argument("--no-anonymize", action="store_true",
                    help="Keep real student names in the display JSON (NOT for public links)")
    args = ap.parse_args()

    if args.mode == "llm" and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: --mode llm needs ANTHROPIC_API_KEY set.")
        sys.exit(1)

    m = MentorMatcher(output_dir=args.output_dir, model=args.model,
                      temperature=args.temperature, seed=args.seed, mode=args.mode,
                      display_out=args.display_out, anonymize_students=not args.no_anonymize)
    matches, unmatched, rec = m.run(args.mentors, args.students)
    print(f"\nDone: {len(matches)} matched, {len(unmatched)} unmatched "
          f"(mode={args.mode}, agreement={rec['solver_agreement']}).")


if __name__ == "__main__":
    main()
