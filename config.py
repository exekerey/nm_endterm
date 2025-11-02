"""Configuration loading for the adaptive LP runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import yaml

from utils.projection import project_box, project_simplex


@dataclass
class ProblemConfig:
    n: int
    p: int
    A: np.ndarray
    b: np.ndarray
    feature_mode: str = "identity"
    env: str = "time_alloc"


@dataclass
class LearningConfig:
    algo: Literal["sgd", "rls"]
    eta: float = 0.1
    lam: float = 1.0
    projection: Literal["simplex", "box"] = "simplex"
    exploration: Literal["none", "epsilon", "cycle"] = "none"
    epsilon: float = 0.0
    k: int = 1
    box_low: float = 0.0
    box_high: Optional[float] = None


@dataclass
class StabilityConfig:
    trust_region: bool = False
    Delta0: float = 1.0
    Delta_min: float = 0.5
    Delta_max: float = 2.0
    up_factor: float = 1.0
    down_factor: float = 1.0
    err_small: float = 0.05


@dataclass
class RunConfig:
    T_max: int
    tol_c: float
    tol_err: float
    seed: int = 0
    logging: bool = True


@dataclass
class InitConfig:
    c: np.ndarray
    theta: np.ndarray
    P: np.ndarray


@dataclass
class Config:
    problem: ProblemConfig
    learning: LearningConfig
    stability: StabilityConfig
    run: RunConfig
    init: InitConfig
    base_path: Path
    environment: dict[str, Any]

    @property
    def projector(self):
        if self.learning.projection == "simplex":
            return project_simplex
        if self.learning.projection == "box":
            high = (
                self.learning.box_high
                if self.learning.box_high is not None
                else np.inf
            )
            return lambda v: project_box(v, self.learning.box_low, high)
        raise ValueError(f"unsupported projection mode: {self.learning.projection}")


def _load_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix in {".csv", ".txt"}:
        return np.loadtxt(path, delimiter=",")
    raise ValueError(f"unsupported matrix file type: {path}")


def _init_vector(
    spec: Any,
    size: int,
    rng: np.random.Generator,
    *,
    normalize: bool = False,
) -> np.ndarray:
    if isinstance(spec, str):
        if spec == "uniform":
            vec = np.ones(size, dtype=float)
            if normalize:
                vec /= vec.sum()
            return vec
        if spec == "random":
            vec = rng.random(size)
            if normalize:
                vec /= max(vec.sum(), 1e-12)
            return vec
        raise ValueError(f"unknown initializer string: {spec}")
    arr = np.asarray(spec, dtype=float)
    if arr.size != size:
        raise ValueError(f"initializer has size {arr.size}, expected {size}")
    return arr


def _init_matrix(
    spec: Any,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if isinstance(spec, (int, float)):
        return float(spec) * np.eye(size, dtype=float)
    if spec == "identity":
        return np.eye(size, dtype=float)
    if spec == "random":
        mat = rng.standard_normal((size, size))
        return mat @ mat.T
    arr = np.asarray(spec, dtype=float)
    if arr.shape != (size, size):
        raise ValueError(f"initializer matrix must be ({size},{size})")
    return arr


def load_config(path: str | Path) -> Config:
    cfg_path = Path(path).expanduser().resolve()
    with cfg_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    base = cfg_path.parent

    problem_raw = raw.get("problem") or {}
    learning_raw = raw.get("learning") or {}
    stability_raw = raw.get("stability") or {}
    run_raw = raw.get("run") or {}
    init_raw = raw.get("init") or {}
    environment_raw = raw.get("environment") or {}

    A = _load_array(base / problem_raw["A_path"])
    b = _load_array(base / problem_raw["b_path"]).reshape(-1)

    problem = ProblemConfig(
        n=int(problem_raw["n"]),
        p=int(problem_raw["p"]),
        A=np.asarray(A, dtype=float),
        b=np.asarray(b, dtype=float),
        feature_mode=str(problem_raw.get("feature_mode", "identity")),
        env=str(problem_raw.get("env", "time_alloc")),
    )

    learning = LearningConfig(
        algo=str(learning_raw["algo"]),
        eta=float(learning_raw.get("eta", 0.1)),
        lam=float(learning_raw.get("lambda", learning_raw.get("lam", 1.0))),
        projection=str(learning_raw.get("projection", "simplex")),
        exploration=str(learning_raw.get("exploration", "none")),
        epsilon=float(learning_raw.get("epsilon", 0.0)),
        k=int(learning_raw.get("k", 1)),
        box_low=float(learning_raw.get("box_low", 0.0)),
        box_high=learning_raw.get("box_high"),
    )
    if learning.box_high is not None:
        learning.box_high = float(learning.box_high)

    stability = StabilityConfig(
        trust_region=bool(stability_raw.get("trust_region", False)),
        Delta0=float(stability_raw.get("Delta0", 1.0)),
        Delta_min=float(stability_raw.get("Delta_min", 0.0)),
        Delta_max=float(stability_raw.get("Delta_max", float("inf"))),
        up_factor=float(stability_raw.get("up_factor", 1.0)),
        down_factor=float(stability_raw.get("down_factor", 1.0)),
        err_small=float(
            stability_raw.get("err_small", run_raw.get("tol_err", 0.05))
        ),
    )

    run = RunConfig(
        T_max=int(run_raw.get("T_max", 100)),
        tol_c=float(run_raw.get("tol_c", 1e-3)),
        tol_err=float(run_raw.get("tol_err", 1e-2)),
        seed=int(run_raw.get("seed", 0)),
        logging=bool(run_raw.get("logging", True)),
    )

    rng = np.random.default_rng(run.seed)

    c0 = _init_vector(init_raw.get("c", "uniform"), problem.p, rng, normalize=True)
    theta0 = _init_vector(init_raw.get("theta", c0.tolist()), problem.p, rng)
    P0 = _init_matrix(init_raw.get("P_scale", 1.0), problem.p, rng)

    init = InitConfig(c=c0, theta=theta0, P=P0)

    return Config(
        problem=problem,
        learning=learning,
        stability=stability,
        run=run,
        init=init,
        base_path=base,
        environment=environment_raw,
    )
