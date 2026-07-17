#!/usr/bin/env python3
"""
Shared assignment-solver layer: cost model, seeded tie-break, OR-Tools
min-cost max-flow, and the independent scipy Hungarian cross-check.

Cost model
==========
Pair utility is a FULL-PRECISION float (mean of the 3 measured criteria —
see matcher.UTILITY_CRITERIA). It is discretized only here:

    substantive arc cost = -round(1000 * utility)          (integer)
    composite  arc cost = substantive * TIE_SCALE + eps    (integer)

eps is a deterministic seeded tie-break in [0, TIE_EPS_MOD): with scores
clustered 8-10 there are many exactly-equal optima, and without an explicit
tie-break the winner is an artifact of solver arc ordering. The scaling
guarantees the tie-break can NEVER flip a substantive difference: the smallest
substantive gap costs TIE_SCALE, while the largest possible total eps across a
whole solution is n_students * (TIE_EPS_MOD - 1) < TIE_SCALE.

An ineligible pair (unscored, or removed by a hard constraint) is NaN in the
score matrix and simply has NO arc — it cannot be chosen at any price.
"""

from __future__ import annotations

import hashlib

import numpy as np
from scipy.optimize import linear_sum_assignment

COST_RESOLUTION = 1000       # utility points -> integer cost units
TIE_SCALE = 1_000_000        # substantive multiplier; > max students * max eps
TIE_EPS_MOD = 1000           # eps in [0, 999]


def tie_eps(seed: int, student_id: str, mentor_id: str) -> int:
    """Deterministic per-pair tie-break in [0, TIE_EPS_MOD), keyed on the run
    seed and the STABLE ids (never names)."""
    h = hashlib.sha256(f"{seed}:{student_id}:{mentor_id}".encode("utf-8")).hexdigest()
    return int(h, 16) % TIE_EPS_MOD


def _sub_cost(utility: float) -> int:
    return -int(round(float(utility) * COST_RESOLUTION))


def solve_min_cost_flow(score_matrix: np.ndarray, slots, eps: np.ndarray):
    """OR-Tools min-cost max-flow over composite integer costs.

    score_matrix : (S, P) float utility; NaN = no arc (unscored or infeasible)
    slots        : per-mentor capacities
    eps          : (S, P) int tie-break matrix (pass zeros for substantive-only)

    Returns (assignments [(j, i)], unmatched [j], substantive_total,
    composite_benefit_total). substantive_total is in COST_RESOLUTION units
    (sum of round(1000*utility) over chosen pairs); composite_benefit_total is
    the exact negated composite objective, comparable with the Hungarian check.
    """
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
            v = score_matrix[j, i]
            if not np.isnan(v):
                cost = _sub_cost(v) * TIE_SCALE + int(eps[j, i])
                a = mcf.add_arc_with_capacity_and_unit_cost(stu(j), men(i), 1, cost)
                arc_of[a] = (j, i)
    mcf.set_node_supply(SRC, S)
    mcf.set_node_supply(SINK, -S)
    if mcf.solve() != mcf.OPTIMAL:
        raise RuntimeError("min-cost flow did not solve to optimality")
    assignments, sub_total, comp_total = [], 0, 0
    matched = set()
    for a, (j, i) in arc_of.items():
        if mcf.flow(a) > 0:
            assignments.append((j, i))
            matched.add(j)
            sub_total += -_sub_cost(score_matrix[j, i])
            comp_total += -_sub_cost(score_matrix[j, i]) * TIE_SCALE - int(eps[j, i])
    unmatched = [j for j in range(S) if j not in matched]
    return assignments, unmatched, sub_total, comp_total


def hungarian_cross_check(score_matrix: np.ndarray, slots, eps: np.ndarray) -> int:
    """Independent optimal composite-benefit total via slot-expanded Hungarian.
    Must equal solve_min_cost_flow's composite_benefit_total exactly."""
    S, P = score_matrix.shape
    cols = [i for i in range(P) for _ in range(int(slots[i]))]
    if not cols:
        return 0
    M = np.zeros((S, len(cols)))
    for cidx, i in enumerate(cols):
        col = score_matrix[:, i]
        # benefit = -composite cost; a real arc always beats unmatched (benefit 0)
        benefit = np.round(col * COST_RESOLUTION) * TIE_SCALE - eps[:, i]
        M[:, cidx] = np.where(np.isnan(col), 0.0, benefit)
    if M.shape[1] < S:      # pad dummy columns so every student can go unmatched
        M = np.hstack([M, np.zeros((S, S - M.shape[1]))])
    r, c = linear_sum_assignment(-M)
    return int(sum(M[ri, ci] for ri, ci in zip(r, c) if M[ri, ci] > 0))
