"""Projection utilities for coefficient updates."""

from __future__ import annotations

import numpy as np


def project_simplex(v: np.ndarray) -> np.ndarray:
    """Project ``v`` onto the probability simplex."""
    if v.ndim != 1:
        raise ValueError("input must be a 1-D array")
    if v.size == 0:
        return v.copy()

    # Shift and clip to enforce the simplex constraints.
    u = np.sort(np.maximum(v, 0.0))[::-1]
    cssv = np.cumsum(u)
    rho_candidates = u * (np.arange(1, v.size + 1)) > (cssv - 1)
    if not np.any(rho_candidates):
        rho = v.size - 1
    else:
        rho = np.nonzero(rho_candidates)[0][-1]
    theta = (cssv[rho] - 1.0) / float(rho + 1)
    return np.maximum(v - theta, 0.0)


def project_box(v: np.ndarray, low: float = 0.0, high: float = 1.0) -> np.ndarray:
    """Project ``v`` into the axis-aligned box ``[low, high]``."""
    if low > high:
        raise ValueError("lower bound must not exceed upper bound")
    return np.clip(v, low, high)

