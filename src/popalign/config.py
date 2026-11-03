"""Config loading and normalization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .paths import project_path


Config = Dict[str, Any]


def load_config(path: str | Path | None) -> Config:
    """Load a YAML or JSON config file."""
    if path is None:
        return {}

    config_path = project_path(path)
    suffix = config_path.suffix.lower()
    with config_path.open("r", encoding="utf-8") as infile:
        if suffix == ".json":
            data = json.load(infile)
            if not isinstance(data, dict):
                raise ValueError(f"Config must contain a mapping at top level: {config_path}")
            return data
        if suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError("YAML configs require PyYAML. Install requirements.txt first.") from exc
            data = yaml.safe_load(infile)
            if data is None:
                data = {}
            if not isinstance(data, dict):
                raise ValueError(f"Config must contain a mapping at top level: {config_path}")
            return data

    raise ValueError(f"Unsupported config format: {config_path}")


def section(config: Config, name: str) -> Config:
    value = config.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{name}' must be a mapping.")
    return value


def require(config: Config, *keys: str) -> Any:
    """Read a nested key and raise a useful error when it is missing."""
    node: Any = config
    for key in keys:
        if not isinstance(node, dict) or key not in node or node[key] is None:
            dotted = ".".join(keys)
            raise ValueError(f"Missing required config value: {dotted}")
        node = node[key]
    return node
