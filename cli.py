"""Command-line interface for adaptive LP experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from config import Config, load_config
from env import RoutingEnvironment, TimeAllocationEnvironment
from plots.metrics import generate_plots
from runner.loop import run_loop
from telemetry.writer import write_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adaptive LP runner")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory for history and plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    env = _make_environment(cfg)
    history = run_loop(cfg, env)

    out_dir = Path(args.out)
    write_history(out_dir / "history.jsonl", (record.to_dict() for record in history))
    generate_plots(history, out_dir / "plots")


def _make_environment(cfg: Config):
    env_cfg = cfg.environment or {}
    env_type = env_cfg.get("type", cfg.problem.env)
    rng = np.random.default_rng(env_cfg.get("seed", cfg.run.seed + 1))

    if env_type == "time_alloc":
        weights = _load_vector(
            env_cfg.get("weights") or env_cfg.get("weights_path"),
            cfg.problem.n,
            cfg.base_path,
        )
        noise_std = float(env_cfg.get("noise_std", 0.0))
        clip = env_cfg.get("clip_range", [0.0, 10.0])
        clip_low, clip_high = float(clip[0]), float(clip[1])
        return TimeAllocationEnvironment(
            weights=weights,
            noise_std=noise_std,
            clip_range=(clip_low, clip_high),
            rng=rng,
        )
    if env_type == "routing":
        base_cost = _load_vector(
            env_cfg.get("base_cost") or env_cfg.get("base_cost_path"),
            cfg.problem.n,
            cfg.base_path,
        )
        B = _load_matrix(
            env_cfg.get("B") or env_cfg.get("B_path"),
            cfg.base_path,
        )
        alpha = float(env_cfg.get("alpha", 1.0))
        noise_std = float(env_cfg.get("noise_std", 0.0))
        return RoutingEnvironment(
            base_cost=base_cost,
            B=B,
            alpha=alpha,
            noise_std=noise_std,
            rng=rng,
        )
    raise ValueError(f"unknown environment type: {env_type}")


def _load_vector(source: Any, expected: int, base: Path) -> np.ndarray:
    arr = _load_array(source, base)
    arr = np.asarray(arr, dtype=float).reshape(-1)
    if arr.size != expected:
        raise ValueError(f"vector length {arr.size} does not match expected {expected}")
    return arr


def _load_matrix(source: Any, base: Path) -> np.ndarray:
    arr = _load_array(source, base)
    return np.asarray(arr, dtype=float)


def _load_array(source: Any, base: Path) -> np.ndarray:
    if source is None:
        raise ValueError("environment parameter missing required array")
    if isinstance(source, str):
        path = (base / source).expanduser().resolve()
        if path.suffix == ".npy":
            return np.load(path)
        if path.suffix in {".csv", ".txt"}:
            return np.loadtxt(path, delimiter=",")
        raise ValueError(f"unsupported file type for {path}")
    return np.array(source, dtype=float)


if __name__ == "__main__":
    main()
