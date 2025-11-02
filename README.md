# Adaptive Linear Program Runner

This project iterates a vanilla linear program whose reward coefficients are unknown. Each round we solve an LP, apply the decision, observe scalar feedback, and update a projection-constrained model of the coefficients.

## Repository layout

- `cli.py` – entry point for running experiments (`python cli.py --config ... --out ...`)
- `config.py` – dataclasses and YAML loader (reads matrices, initializes learners)
- `env/` – environment protocol plus synthetic time-allocation and routing examples
- `solver/` – `linprog`-based LP wrapper with optional L1 trust region
- `learners/` – SGD and RLS coefficient updaters with projection hooks
- `runner/loop.py` – orchestration loop, exploration policies, trust-region scheduling
- `telemetry/` – JSONL writer
- `plots/` – matplotlib helpers for error, coefficient, and action-change curves
- `configs/` – ready-to-run YAML configs (see `configs/time_alloc.yaml`)
- `data/` – sample constraint matrices (`A.npy`, `b.npy`)
- `tests/` – unit and integration tests
- `app.py` – Streamlit front-end for interactive productivity planning

## Running an experiment

```bash
python cli.py --config configs/time_alloc.yaml --out runs/exp1
```

The command writes `runs/exp1/history.jsonl` and plots under `runs/exp1/plots/`. Inspect history with tools like `jq` or pandas.

## Plugging in your own environment

Implement the `Environment` protocol in `env/base.py`:

```python
class MyEnv:
    n = ...
    p = ...

    def observe(self, x: np.ndarray) -> float:
        # Evaluate your system with decision vector x and return scalar feedback.
        ...

    def features(self, x: np.ndarray) -> np.ndarray:
        # Provide feature vector (default: return x).
        return x

    def extra_constraints(self, iteration: int):
        # Optional: supply additional linear constraints (A_add, b_add).
        return None
```

Then adapt `cli.py` (or create a custom entrypoint) to instantiate `MyEnv` instead of the synthetic examples. All other components remain unchanged.

## Development

Install dependencies with `pip install -r requirements.txt`. Run the test suite via `pytest`. The codebase intentionally uses only numpy/scipy/matplotlib to keep deployment light.

### Streamlit planner

Launch the interactive interface with:

```bash
streamlit run app.py
```

Use it to define activities, generate daily plans, log outcomes, and let the learner update its coefficient estimates in real time.
