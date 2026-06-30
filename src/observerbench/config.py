"""Configuration loading for ObserverBench task YAML files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a task config file.

    The migration keeps config loading dependency-light, so ``.yaml`` files are
    JSON-compatible YAML. Full YAML support can be added later if a migrated
    task truly needs it.
    """

    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{config_path} must be JSON-compatible YAML"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(f"{config_path} must parse to a mapping")
    return data
