"""Base types for feedback environments."""

from __future__ import annotations

from typing import Optional, Protocol, Tuple

import numpy as np


class Environment(Protocol):
    """Interactive environment that supplies feedback for chosen actions."""

    n: int
    p: int

    def observe(self, x: np.ndarray) -> float:
        """Return scalar feedback for the chosen decision vector ``x``."""

    def features(self, x: np.ndarray) -> np.ndarray:
        """Return feature representation of ``x`` compatible with the learner."""

    def extra_constraints(
        self, iteration: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return optional additional linear constraints for the LP."""

