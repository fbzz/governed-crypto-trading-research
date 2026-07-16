from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return config


def smoke_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a fast, deterministic offline variant of the main config."""
    result = deepcopy(config)
    result["data"]["source"] = "fixture"
    result["data"]["fixture_days"] = 520
    result["validation"]["folds"] = 2
    result["features"]["lookback"] = 32
    result["transformer"]["d_model"] = 32
    result["transformer"]["n_layers"] = 1
    result["transformer"]["epochs"] = 3
    result["transformer"]["early_stopping"] = 2
    result["output_dir"] = "artifacts/smoke"
    return result
