"""Telemetry writer producing JSON lines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping


def write_history(path: str | Path, records: Iterable[Mapping[str, object]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, default=_json_fallback))
            handle.write("\n")


def _json_fallback(obj):
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"object of type {type(obj)!r} is not JSON serializable")

