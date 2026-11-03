"""JSONL and path IO utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

from .paths import project_path


JsonDict = Dict[str, Any]


def read_jsonl(path: str | Path) -> Iterator[JsonDict]:
    jsonl_path = project_path(path)
    with jsonl_path.open("r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_jsonl_list(path: str | Path) -> List[JsonDict]:
    return list(read_jsonl(path))


def write_jsonl(path: str | Path, rows: Iterable[JsonDict], append: bool = False) -> None:
    jsonl_path = project_path(path)
    if jsonl_path.parent != Path("."):
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if append else "w"
    with jsonl_path.open(mode, encoding="utf-8") as outfile:
        for row in rows:
            outfile.write(json.dumps(row, ensure_ascii=False) + "\n")


def jsonl_files(path: str | Path) -> List[Path]:
    resolved = project_path(path)
    if resolved.is_dir():
        files = sorted(resolved.glob("*.jsonl"))
        if not files:
            raise ValueError(f"No JSONL input files found in {resolved}")
        return files
    if not resolved.is_file():
        raise FileNotFoundError(f"Input file not found: {resolved}. See examples/README.md for sample inputs and docs/usage.md for input formats.")
    return [resolved]


def split_paths(value: str | Path | Iterable[str | Path]) -> List[Path]:
    if isinstance(value, (str, Path)):
        paths = [project_path(part.strip()) for part in str(value).split(",") if part.strip()]
    else:
        paths = [project_path(part) for part in value]
    if not paths:
        raise ValueError("At least one input file is required")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}. See examples/README.md for sample inputs and docs/usage.md for input formats.")
    return paths
