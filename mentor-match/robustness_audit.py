#!/usr/bin/env python3
"""
Robustness audit for a solved matching — no LLM calls, works entirely from the
score matrix. Three probes:

  a) Forbid-each-edge: for every selected pair, delete its arc and re-solve.
     If the substantive optimum is unchanged, that pair was arbitrary among
     equally-optimal solutions — flagged `arbitrary`.
  b) Perturbation: add small seeded Gaussian noise to all scores (sigma scaled
     to the observed score spread), re-solve N times; record each selected
     pair's selection frequency and each student's unmatched frequency.
  c) Capacity sensitivity: re-solve under low (each mentor -1 slot, floor 1),
     base, and high (each +1) capacity scenarios.

A selected pair is `robust` iff it is selected in >= ROBUST_FREQ of
perturbation runs AND is not flagged arbitrary. All probes use
substantive-only costs (zero tie-break eps): noise itself breaks ties in (b),
and (a)/(c) compare substantive objectives.
"""

from __future__ import annotations

import numpy as np

from solver import COST_RESOLUTION, solve_min_cost_flow

N_PERTURBATIONS = 200
NOISE_SIGMA_FACTOR = 0.1     # sigma = 0.1 * std of the observed scores
ROBUST_FREQ = 0.8


def _solve_sub(scores, slots):
    eps = np.zeros(scores.shape, dtype=int)
    assignments, unmatched, sub_total, _ = solve_min_cost_flow(scores, slots, eps)
    return assignments, unmatched, sub_total


def perturbation_frequencies(scores: np.ndarray, slots, seed: int,
                             n_perturbations: int = N_PERTURBATIONS) -> dict:
    """The seeded perturbation experiment, separable so the matcher can use the
    SAME frequencies for its stability-weighted tie-break that the audit later
    reports — one experiment, two consumers."""
    S, P = scores.shape
    observed = scores[~np.isnan(scores)]
    sigma = float(NOISE_SIGMA_FACTOR * observed.std())
    rng = np.random.default_rng(seed)
    pair_counts: dict[tuple[int, int], int] = {}
    unmatched_counts = np.zeros(S, dtype=int)
    for _ in range(n_perturbations):
        noisy = scores + rng.normal(0.0, sigma, scores.shape)   # NaN + noise stays NaN
        assignments, unmatched, _ = _solve_sub(noisy, slots)
        for pair in assignments:
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
        for j in unmatched:
            unmatched_counts[j] += 1
    return {"pair_counts": pair_counts, "unmatched_counts": unmatched_counts,
            "sigma": sigma, "n": n_perturbations}


def run_audit(scores: np.ndarray, slots, selected: list[tuple[int, int]],
              seed: int, student_ids: list[str], mentor_ids: list[str],
              n_perturbations: int = N_PERTURBATIONS, log=None,
              precomputed_perturbation: dict | None = None) -> dict:
    """scores: (S, P) float utility matrix, NaN = no arc. selected: solved pairs."""
    S, P = scores.shape
    say = log or (lambda m: None)

    # ── a) forbid-each-edge ──────────────────────────────────────────────
    _, _, base_total = _solve_sub(scores, slots)
    forbid = []
    for (j, i) in selected:
        cut = scores.copy()
        cut[j, i] = np.nan
        _, _, t = _solve_sub(cut, slots)
        forbid.append({
            "student_id": student_ids[j], "mentor_id": mentor_ids[i],
            "objective_delta": (t - base_total) / COST_RESOLUTION,  # in score points
            "arbitrary": bool(t == base_total),
        })
    n_arbitrary = sum(1 for f in forbid if f["arbitrary"])
    say(f"Audit a) forbid-each-edge: {n_arbitrary}/{len(selected)} pairs are "
        f"arbitrary among equal optima")

    # ── b) seeded perturbation (reused from the tie-break pass if provided) ──
    pert = precomputed_perturbation or perturbation_frequencies(
        scores, slots, seed, n_perturbations)
    pair_counts, unmatched_counts = pert["pair_counts"], pert["unmatched_counts"]
    sigma, n_perturbations = pert["sigma"], pert["n"]
    sel_freq = {pair: pair_counts.get(pair, 0) / n_perturbations for pair in selected}
    say(f"Audit b) perturbation x{n_perturbations} (sigma={sigma:.3f}): "
        f"{sum(1 for f in sel_freq.values() if f >= ROBUST_FREQ)}/{len(selected)} "
        f"selected pairs appear in >= {ROBUST_FREQ:.0%} of runs")

    # ── c) capacity sensitivity ──────────────────────────────────────────
    # "base" describes the ACTUAL selected solution (with mass ties, an
    # eps-free re-solve can land on a different equally-optimal assignment,
    # which would make the vs-base deltas silently wrong for the real roster).
    # low/high are substantive-only re-solves under shifted capacities; their
    # deltas are computed against the actual selection's unmatched set.
    matched_js = {j for (j, _) in selected}
    base_um_ids = sorted(student_ids[j] for j in range(S) if j not in matched_js)
    base_unmatched_ids = set(base_um_ids)
    base_sel_total = sum(int(round(scores[j, i] * COST_RESOLUTION)) for (j, i) in selected)
    scenarios = {"base": {
        "total_slots": int(sum(slots)),
        "matched": len(selected),
        "unmatched": len(base_um_ids),
        "substantive_total": base_sel_total / COST_RESOLUTION,
        "unmatched_student_ids": base_um_ids,
        "note": "the published assignment (substantively optimal; equals the "
                "eps-free optimum objective)",
    }}
    assert base_sel_total == base_total, \
        f"published assignment is not substantively optimal: {base_sel_total} vs {base_total}"
    for name, sl in (("low", [max(1, s - 1) for s in slots]),
                     ("high", [s + 1 for s in slots])):
        assignments, unmatched, total = _solve_sub(scores, sl)
        um_ids = sorted(student_ids[j] for j in unmatched)
        scenarios[name] = {
            "total_slots": int(sum(sl)),
            "matched": len(assignments),
            "unmatched": len(unmatched),
            "substantive_total": total / COST_RESOLUTION,
            "unmatched_student_ids": um_ids,
            "note": "substantive-only re-solve under shifted capacities; among "
                    "equal optima the selection may differ from the published one",
            "newly_unmatched_vs_base": sorted(set(um_ids) - base_unmatched_ids),
            "newly_matched_vs_base": sorted(base_unmatched_ids - set(um_ids)),
        }
    say(f"Audit c) capacity: matched low={scenarios['low']['matched']} "
        f"base={scenarios['base']['matched']} high={scenarios['high']['matched']}")

    # ── verdicts ─────────────────────────────────────────────────────────
    arbitrary_pairs = {(f["student_id"], f["mentor_id"]) for f in forbid if f["arbitrary"]}
    robust: dict[tuple[int, int], bool] = {}
    for (j, i) in selected:
        key = (student_ids[j], mentor_ids[i])
        robust[(j, i)] = (sel_freq[(j, i)] >= ROBUST_FREQ) and (key not in arbitrary_pairs)

    report = {
        "seed": seed,
        "n_perturbations": n_perturbations,
        "noise_sigma": sigma,
        "noise_basis": f"{NOISE_SIGMA_FACTOR} * std of observed scores "
                       f"(std={sigma / NOISE_SIGMA_FACTOR:.3f})",
        "robust_rule": f"selected in >= {ROBUST_FREQ:.0%} of perturbation runs "
                       "AND not arbitrary under forbid-each-edge",
        "base_substantive_total": base_total / COST_RESOLUTION,
        "forbid_each_edge": forbid,
        "perturbation": {
            "pair_selection_frequency": [
                {"student_id": student_ids[j], "mentor_id": mentor_ids[i],
                 "frequency": sel_freq[(j, i)]}
                for (j, i) in selected],
            "student_unmatched_frequency": [
                {"student_id": student_ids[j],
                 "frequency": unmatched_counts[j] / n_perturbations}
                for j in range(S) if unmatched_counts[j] > 0],
        },
        "capacity_scenarios": scenarios,
        "summary": {
            "selected_pairs": len(selected),
            "arbitrary_pairs": n_arbitrary,
            "fragile_pairs": sum(1 for v in robust.values() if not v),
            "robust_pairs": sum(1 for v in robust.values() if v),
        },
    }
    return report, robust
