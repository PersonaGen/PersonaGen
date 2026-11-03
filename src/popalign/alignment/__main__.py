"""CSV interface for the new, small-scale alignment reference implementation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from . import align


def read_csv(path, columns=None, id_column=None):
    """Read explicit numeric features without reordering/filtering input rows."""
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames
        if not headers or len(set(headers)) != len(headers):
            raise ValueError(f"{path}: missing or duplicate column names")
        if id_column and id_column not in headers:
            raise ValueError(f"{path}: missing ID column {id_column!r}")
        selected = list(columns) if columns is not None else [h for h in headers if h != id_column]
        if not selected or len(set(selected)) != len(selected) or any(h not in headers for h in selected):
            raise ValueError(f"{path}: requested feature columns are missing, empty, or duplicated")
        if id_column in selected:
            raise ValueError("ID column must not also be a feature column")
        values, ids = [], []
        for index, row in enumerate(reader):
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f"{path}: malformed CSV row {index + 2}")
            try:
                values.append([float(row[h]) for h in selected])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}: nonnumeric feature at CSV row {index + 2}") from exc
            ids.append(row[id_column] if id_column else str(index))
    if not values:
        raise ValueError(f"{path}: no data rows")
    return np.asarray(values, dtype=float), ids, selected


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=(
        "CPU alignment reference with camera-ready Eq. 3 top-fraction IS. "
        "Not the original experiment script. Defaults are demonstration settings."
    ))
    parser.add_argument("--persona-csv", required=True, type=Path)
    parser.add_argument("--human-csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New output directory (must not exist)")
    parser.add_argument("--columns", nargs="+", help="Ordered numeric feature names; default: all columns except ID")
    parser.add_argument("--id-column", help="Optional shared ID-column name; IDs are metadata, never features")
    parser.add_argument("--retain-fraction", type=float, default=0.7,
                        help="Retain floor(fraction * persona rows) by decreasing IS weight (default: 0.7)")
    parser.add_argument("--n-output", type=int, default=128)
    parser.add_argument("--bandwidth", type=float, default=0.5)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--feature-weights", nargs="+", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-cost-entries", type=int, default=2_000_000)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--max-iterations", type=int, default=5000)
    parser.add_argument("--transport-cost-convention", choices=["main-text", "appendix"],
                        default="main-text", help="Legacy mass-weighted cost or camera-ready Eq. 6 / Eq. 28 conditional cost")
    parser.add_argument("--save-plan", action="store_true", help="Also save dense cost and transport plan as NPZ")
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError(f"Output path already exists: {args.output}")
        persona, ids, columns = read_csv(args.persona_csv, args.columns, args.id_column)
        human, human_ids, human_columns = read_csv(args.human_csv, args.columns, args.id_column)
        if columns != human_columns:
            raise ValueError("Feature columns/order differ; pass --columns explicitly")
        parameters = {name: getattr(args, name) for name in [
            "retain_fraction", "n_output", "bandwidth", "epsilon", "temperature", "feature_weights",
            "seed", "max_cost_entries", "tolerance", "max_iterations", "transport_cost_convention"]}
        result = align(persona, human, **parameters)
        args.output.mkdir(parents=True, exist_ok=False)
        with (args.output / "selection.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["output_index", "persona_row", "persona_id", "candidate_index", "ot_probability"])
            for i, (row, candidate) in enumerate(zip(result.selected_rows, result.selected_candidates)):
                writer.writerow([i, int(row), ids[row], int(candidate), result.transport_probabilities[candidate]])
        with (args.output / "candidates.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["candidate_index", "persona_row", "persona_id", "is_probability", "transport_cost", "ot_probability"])
            for i, row in enumerate(result.candidate_rows):
                writer.writerow([i, int(row), ids[row], result.importance_probabilities[row],
                                 result.transport_costs[i], result.transport_probabilities[i]])
        with (args.output / "importance.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["persona_row", "persona_id", "is_probability"])
            writer.writerows((i, ids[i], p) for i, p in enumerate(result.importance_probabilities))
        metadata = {
            "implementation": "new_reference_implementation_not_original_experiment_script",
            "source": "Camera-ready Eq. 3 top-fraction IS; explicit legacy OT convention",
            "defaults": "IS retains the top 70%; other defaults are demonstration settings",
            "parameters": parameters,
            "effective_feature_weights": args.feature_weights or [1.0] * len(columns),
            "columns": columns, "id_column": args.id_column,
            "persona_rows": len(persona), "human_rows": len(human),
            "input_files": {kind: {"path": str(path.resolve()), "sha256": sha256(path)} for kind, path in
                            [("persona", args.persona_csv), ("human", args.human_csv)]},
            "indexing": "Zero-based data-row indices excluding the CSV header; candidate indices index the ranked IS subset",
            "replacement": {"importance_sampling": False, "ot_sampling": True},
            "is_selection": "top_fraction_by_density_ratio",
            "is_tie_break": "input_row_order",
            "n_candidates": len(result.candidate_rows),
            "transport_cost_definition": ("sum_j Gamma_ij * C_ij" if args.transport_cost_convention == "main-text"
                                          else "sum_j Gamma_ij * C_ij / a_i (a_i = 1 / n_candidates)"),
            "sinkhorn_iterations": result.sinkhorn_iterations,
            "maximum_absolute_marginal_error": result.marginal_error,
            "numpy_version": np.__version__,
            "python_version": sys.version.split()[0],
        }
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        if args.save_plan:
            np.savez_compressed(args.output / "transport.npz", cost=result.cost, plan=result.plan)
        print(f"Selected {len(result.selected_rows)} draws ({len(set(result.selected_rows))} unique personas) into {args.output}")
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"alignment: {exc}\n")


if __name__ == "__main__":
    main()
