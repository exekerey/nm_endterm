"""Recursive least squares style updates with projection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Projector(Protocol):
    def __call__(self, v: np.ndarray) -> np.ndarray:
        ...


@dataclass
class RLSState:
    theta: np.ndarray
    P: np.ndarray


def rls_update(
    state: RLSState,
    features: np.ndarray,
    feedback: float,
    *,
    lam: float,
    project: Projector,
) -> tuple[RLSState, np.ndarray, float]:
    """Perform a recursive least squares update.

    Returns the updated state, projected coefficients, and instantaneous error.
    """
    phi = features.reshape(-1, 1)
    yhat = float(state.theta @ phi.ravel())
    error = feedback - yhat
    denom = lam + float(phi.T @ state.P @ phi)
    if denom <= 0:
        raise ValueError("RLS denominator became non-positive; adjust lambda or P.")
    gain = (state.P @ phi) / denom
    theta = state.theta + (gain.ravel() * error)
    P = (state.P - gain @ phi.T @ state.P) / lam
    projected = project(theta)
    return RLSState(theta, P), projected, float(error)

