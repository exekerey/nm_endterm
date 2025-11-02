"""Synthetic environments for experimentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .base import Environment


@dataclass
class TimeAllocationEnvironment(Environment):
    """Allocate continuous resources with latent linear reward."""

    weights: np.ndarray
    noise_std: float = 0.0
    clip_range: Tuple[float, float] = (0.0, 10.0)
    rng: Optional[np.random.Generator] = None

    def __post_init__(self) -> None:
        self.n = self.weights.size
        self.p = self.n
        if self.rng is None:
            self.rng = np.random.default_rng()

    def observe(self, x: np.ndarray) -> float:
        noise = 0.0 if self.noise_std <= 0 else self.rng.normal(0.0, self.noise_std)
        value = float(self.weights @ x + noise)
        low, high = self.clip_range
        return float(np.clip(value, low, high))

    def features(self, x: np.ndarray) -> np.ndarray:
        return x.astype(float, copy=False)

    def extra_constraints(self, iteration: int):
        return None


@dataclass
class RoutingEnvironment(Environment):
    """Quadratic routing cost collapsed to a scalar signal."""

    base_cost: np.ndarray
    B: np.ndarray
    alpha: float
    noise_std: float = 0.0
    rng: Optional[np.random.Generator] = None

    def __post_init__(self) -> None:
        if self.rng is None:
            self.rng = np.random.default_rng()
        self.n = self.base_cost.size
        self.p = self.n
        if self.B.shape[1] != self.n:
            raise ValueError("matrix B must map flows to constraints (shape m x n)")

    def observe(self, x: np.ndarray) -> float:
        flow = self.B @ x
        quadratic = self.alpha * np.sum(flow**2)
        value = float(self.base_cost @ x + quadratic)
        if self.noise_std > 0:
            value += float(self.rng.normal(0.0, self.noise_std))
        return value

    def features(self, x: np.ndarray) -> np.ndarray:
        return x.astype(float, copy=False)

    def extra_constraints(self, iteration: int):
        return None

