"""Plotting utilities for monitoring metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from runner.loop import RoundRecord


def generate_plots(history: Iterable[RoundRecord], out_dir: str | Path) -> None:
    records = list(history)
    if not records:
        return
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    t = np.array([rec.t for rec in records], dtype=float)
    errors = np.array([abs(rec.err) for rec in records], dtype=float)
    coeffs = np.array([rec.c for rec in records], dtype=float)
    decisions = np.array([rec.x for rec in records], dtype=float)

    ema = _ema(errors, span=max(5, int(len(records) * 0.1)))
    plt.figure(figsize=(6, 4))
    plt.plot(t, errors, label="|Z - Zhat|")
    plt.plot(t, ema, label="EMA", linestyle="--")
    plt.xlabel("iteration")
    plt.ylabel("error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path / "error_curve.png", dpi=150)
    plt.close()

    if coeffs.size:
        plt.figure(figsize=(6, 4))
        for j in range(coeffs.shape[1]):
            plt.plot(t, coeffs[:, j], label=f"c{j+1}")
        plt.xlabel("iteration")
        plt.ylabel("coefficient value")
        plt.tight_layout()
        plt.savefig(out_path / "coefficients.png", dpi=150)
        plt.close()

    if len(records) > 1:
        diffs = np.linalg.norm(np.diff(decisions, axis=0), ord=1, axis=1)
        plt.figure(figsize=(6, 4))
        plt.plot(t[1:], diffs, label="|x_t - x_{t-1}|_1")
        plt.xlabel("iteration")
        plt.ylabel("decision change")
        plt.tight_layout()
        plt.savefig(out_path / "decision_change.png", dpi=150)
        plt.close()


def _ema(values: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1.0)
    ema = np.zeros_like(values)
    current = 0.0
    for idx, val in enumerate(values):
        if idx == 0:
            current = val
        else:
            current = alpha * val + (1 - alpha) * current
        ema[idx] = current
    return ema
