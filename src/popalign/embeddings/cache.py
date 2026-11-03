"""Embedding cache path helpers."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class EmbeddingCachePaths:
    query_embeddings: Path
    persona_embeddings: Path


def _as_paths(value: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(value, Path):
        return [value]
    if isinstance(value, str):
        return [Path(part.strip()) for part in value.split(",") if part.strip()]
    return [Path(part) for part in value]


def _path_metadata(value: str | Path | Iterable[str | Path]) -> list[dict]:
    metadata = []
    for path in _as_paths(value):
        expanded = path.expanduser()
        if expanded.exists():
            stat = expanded.stat()
            metadata.append(
                {
                    "path": str(expanded.resolve()),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
        else:
            metadata.append({"path": str(expanded), "missing": True})
    return metadata


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return slug[:80] or "embedding"


def embedding_cache_paths(
    cache_dir: str | Path,
    model_name: str,
    survey_type: str,
    persona_file_paths: str | Path | Iterable[str | Path],
) -> EmbeddingCachePaths:
    """Build cache paths scoped to model, survey type, and persona input files."""
    payload = {
        "model_name": model_name,
        "survey_type": survey_type,
        "persona_files": _path_metadata(persona_file_paths),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    prefix = f"{_slug(model_name)}_{_slug(survey_type)}_{digest}"
    root = Path(cache_dir).expanduser()
    return EmbeddingCachePaths(
        query_embeddings=root / f"{prefix}_query_embeddings.npy",
        persona_embeddings=root / f"{prefix}_persona_embeddings.npy",
    )
