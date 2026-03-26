"""Compatibility wrapper for linear sum assignment.

Prefers SciPy when available. Falls back to an exact dynamic-programming solver
for the small assignment problems used in this project.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment as _scipy_linear_sum_assignment
except Exception:  # pragma: no cover - exercised only when scipy is absent
    _scipy_linear_sum_assignment = None


def _dp_linear_sum_assignment(cost_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cost = np.asarray(cost_matrix, dtype=float)
    if cost.ndim != 2:
        raise ValueError("cost_matrix must be 2-dimensional")

    n_rows, n_cols = cost.shape
    if n_rows == 0 or n_cols == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    transposed = False
    if n_rows > n_cols:
        cost = cost.T
        n_rows, n_cols = cost.shape
        transposed = True

    @lru_cache(maxsize=None)
    def _solve(row: int, used_mask: int) -> tuple[float, tuple[int, ...]]:
        if row == n_rows:
            return 0.0, ()

        best_cost = float("inf")
        best_cols: tuple[int, ...] | None = None
        for col in range(n_cols):
            if used_mask & (1 << col):
                continue
            sub_cost, sub_cols = _solve(row + 1, used_mask | (1 << col))
            total = float(cost[row, col]) + sub_cost
            if total < best_cost:
                best_cost = total
                best_cols = (col,) + sub_cols

        assert best_cols is not None
        return best_cost, best_cols

    _, selected_cols = _solve(0, 0)
    row_ind = np.arange(n_rows, dtype=int)
    col_ind = np.array(selected_cols, dtype=int)

    if transposed:
        return col_ind, row_ind
    return row_ind, col_ind


def linear_sum_assignment(cost_matrix) -> tuple[np.ndarray, np.ndarray]:
    """Return the optimal minimum-cost bipartite assignment."""
    if _scipy_linear_sum_assignment is not None:
        return _scipy_linear_sum_assignment(cost_matrix)
    return _dp_linear_sum_assignment(np.asarray(cost_matrix, dtype=float))
