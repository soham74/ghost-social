"""Cross-condition agreement metric used by the comparison view."""

import numpy as np


def cohens_kappa(ratings_a, ratings_b):
    """Cohen's kappa between two rating dicts {(mentee, mentor): int_score}.
    Scores are bucketed into low (1-3), medium (4-6), high (7-10)."""

    def bucket(s):
        return 0 if s <= 3 else (1 if s <= 6 else 2)

    pairs = set(ratings_a) & set(ratings_b)
    n = len(pairs)
    if n == 0:
        return 0.0

    matrix = np.zeros((3, 3))
    for p in pairs:
        matrix[bucket(ratings_a[p])][bucket(ratings_b[p])] += 1

    p_o = np.trace(matrix) / n
    p_e = sum(matrix.sum(axis=1)[i] * matrix.sum(axis=0)[i] for i in range(3)) / (n * n)
    return 1.0 if p_e == 1.0 else float((p_o - p_e) / (1 - p_e))
