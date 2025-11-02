"""Stochastic gradient-style coefficient updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np


class Projector(Protocol):
    def __call__(self, v: np.ndarray) -> np.ndarray:
        ...


@dataclass
class SGDState:
    coefficients: np.ndarray


def sgd_update(
    state: SGDState,
    features: np.ndarray,
    feedback: float,
    prediction: float,
    *,
    eta: float,
    project: Projector,
) -> tuple[SGDState, float]:
    """Perform a projected SGD step and return the updated state & error."""
    error = feedback - prediction
    new_coeffs = state.coefficients + eta * error * features
    projected = project(new_coeffs)
    return SGDState(projected), float(error)

