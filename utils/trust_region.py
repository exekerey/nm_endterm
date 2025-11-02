"""Trust-region scheduling helpers."""

from __future__ import annotations

import numpy as np


def adjust_delta(
    delta: float,
    err: float,
    *,
    small_thresh: float,
    up: float,
    down: float,
    delta_min: float,
    delta_max: float,
) -> float:
    """Adaptively scale the trust-region radius based on the prediction error."""
    if abs(err) < small_thresh:
        delta *= up
    else:
        delta *= down
    return float(np.clip(delta, delta_min, delta_max))

