"""Configuration loading for placeholder ObserverBench task YAML files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a placeholder config file.

    The initial migration keeps base installs stdlib-only, so placeholder
    ``.yaml`` files are JSON-compatible YAML. Full YAML support can be added
    later if a migrated task truly needs it.
    """

    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{config_path} must be JSON-compatible YAML during the skeleton phase"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(f"{config_path} must parse to a mapping")
    return data
