#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WVS Evaluation Script

This script calculates distribution similarity metrics between LLM and Human responses
for WVS (World Values Survey) data across different subdomains.

Metrics computed:
- Avg_W1: Average Wasserstein distance across dimensions
- Frechet: Frechet distance between Gaussian distributions
- Sliced_W1: Sliced Wasserstein distance
- Energy_Dist: Energy distance

Output:
    - CSV files for each subdomain
    - Summary results file
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.stats import wasserstein_distance
from scipy.linalg import sqrtm


def frechet_gauss(X: np.ndarray, Y: np.ndarray) -> float:
    """Frechet distance compatible with 1-D / N-D."""
    if X.size == 0 or Y.size == 0:
        return np.nan
    mu1, mu2 = np.nanmean(X, 0), np.nanmean(Y, 0)
    c1, c2 = np.cov(X.T), np.cov(Y.T)

    if np.ndim(c1) == 0:  # 1-D
        if not np.isfinite(c1) or not np.isfinite(c2):
            return np.nan
        cov_prod = np.sqrt(c1 * c2).item()
        return float((mu1 - mu2) ** 2 + (c1 + c2 - 2 * cov_prod))

    if not np.all(np.isfinite(c1)) or not np.all(np.isfinite(c2)):
        return np.nan
    try:
        covm = sqrtm(c1 @ c2)
        covm = covm.real if np.iscomplexobj(covm) else covm
    except Exception:
        return np.nan
    diff = mu1 - mu2
    return float(diff.dot(diff) + np.trace(c1 + c2 - 2 * covm))


def sliced_w1(X: np.ndarray, Y: np.ndarray, n_proj: int = 100) -> float:
    """Compute sliced Wasserstein distance."""
    if X.size == 0 or Y.size == 0:
        return np.nan
    rng = np.random.default_rng(42)
    d = X.shape[1]
    vecs = rng.normal(size=(n_proj, d))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return np.mean([wasserstein_distance(X @ v, Y @ v) for v in vecs])


def energy_nd(X: np.ndarray, Y: np.ndarray, max_samples: int = 5000) -> float:
    """Compute energy distance."""
    if X.size == 0 or Y.size == 0:
        return np.nan
    n_x = min(len(X), max_samples)
    n_y = min(len(Y), max_samples)
    idx_x = np.random.choice(len(X), n_x, replace=False)
    idx_y = np.random.choice(len(Y), n_y, replace=False)
    X_s, Y_s = X[idx_x], Y[idx_y]
    return max(
        2 * cdist(X_s, Y_s).mean()
        - pdist(X_s).mean()
        - pdist(Y_s).mean(),
        0.0,
    )


def get_question_keys(subdomain_path: Path) -> Set[str]:
    """Extract question keys starting with 'Q' from any jsonl file in subdomain."""
    for fp in subdomain_path.glob("*.jsonl"):
        with fp.open() as f:
            first = json.loads(f.readline())
        return {k for k in first if k.startswith("Q")}
    return set()


def row_to_vec(row: Dict, q_keys: List[str]) -> np.ndarray | None:
    """Convert row to vector if it contains all questions with valid values (0-100)."""
    vals = []
    for k in q_keys:
        if k not in row:
            return None
        try:
            v = float(row[k])
        except Exception:
            return None
        if v < 0 or v > 100:
            return None
        vals.append(v)
    return np.array(vals, dtype=np.float32)


def load_matrix(jsonl_path: Path, q_keys: List[str]) -> np.ndarray:
    """Read jsonl, filter invalid rows, return (n, d) array."""
    vecs = []
    with jsonl_path.open() as f:
        for line in f:
            vec = row_to_vec(json.loads(line), q_keys)
            if vec is not None:
                vecs.append(vec)
    return np.vstack(vecs) if vecs else np.empty((0, len(q_keys)), dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="WVS Evaluation Script")
    parser.add_argument("--llm_root", type=str, required=True,
                       help="Path to LLM results directory")
    parser.add_argument("--human_root", type=str, required=True,
                       help="Path to human results directory")
    parser.add_argument("--output_root", type=str, required=True,
                       help="Path to output directory")

    args = parser.parse_args()

    # Setup paths
    LLM_ROOT = Path(args.llm_root)
    HUMAN_ROOT = Path(args.human_root)
    OUT_ROOT = Path(args.output_root)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Processing WVS evaluation...")
    print(f"LLM results: {LLM_ROOT}")
    print(f"Human results: {HUMAN_ROOT}")
    print(f"Output: {OUT_ROOT}")

    summary_lines: List[str] = []
    subdomain_dirs = sorted([d for d in LLM_ROOT.iterdir() if d.is_dir()])

    for sub_dir in subdomain_dirs:
        sub = sub_dir.name
        print(f"▶ Processing {sub} ...")
        q_keys = sorted(get_question_keys(sub_dir), key=lambda x: int(re.findall(r"\d+", x)[0]))
        if not q_keys:
            print(f"   ⚠ No question keys found, skipped.")
            continue

        # human matrix
        human_jsonl = HUMAN_ROOT / sub / f"wvs_{sub}_human.jsonl"
        if not human_jsonl.exists():
            print(f"   ⚠ Human jsonl not found, skipped.")
            continue
        Y = load_matrix(human_jsonl, q_keys)

        # Collect results for each method
        result_rows = []
        for fp in sorted(sub_dir.glob("wvs_*.jsonl")):
            method_name = fp.stem.split(f"wvs_{sub}_", 1)[-1]
            X = load_matrix(fp, q_keys)

            if X.shape[0] == 0 or Y.shape[0] == 0:
                metrics = [np.nan] * 4
            else:
                # Avg W1
                w1s = [
                    wasserstein_distance(X[:, i], Y[:, i]) for i in range(X.shape[1])
                ]
                avg_w1 = float(np.mean(w1s))

                frechet = frechet_gauss(X, Y)
                sw1 = sliced_w1(X, Y)
                edist = energy_nd(X, Y)
                metrics = [avg_w1, frechet, sw1, edist]

            avg_all = float(np.nanmean(metrics))
            result_rows.append([method_name, *metrics, avg_all])

        # Write per-subdomain CSV / TXT
        out_csv = OUT_ROOT / f"{sub}.csv"
        out_txt = OUT_ROOT / f"{sub}.txt"

        header = ["method", "Avg_W1", "Frechet", "Sliced_W1", "Energy_Dist", "avg"]
        with out_csv.open("w") as f_csv, out_txt.open("w") as f_txt:
            f_csv.write(",".join(header) + "\n")
            f_txt.write("\t".join(header) + "\n")
            for row in result_rows:
                csv_line = ",".join(f"{x}" for x in row) + "\n"
                txt_line = "\t".join(f"{x:>12}" if isinstance(x, float) else x for x in row) + "\n"
                f_csv.write(csv_line)
                f_txt.write(txt_line)

        # Add to summary
        summary_lines.append(sub)
        summary_lines.append(",".join(header))
        summary_lines.extend(
            [",".join(str(x) for x in row) for row in result_rows]
        )
        summary_lines.append("")  # Empty line
        summary_lines.append("")  # Second empty line
        summary_lines.append("")  # Third empty line

    # Write summary_result.csv
    summary_csv = OUT_ROOT / "summary_result.csv"
    with summary_csv.open("w") as f:
        f.write("\n".join(summary_lines))
    print("\n✅ All done. Results saved to", OUT_ROOT)


if __name__ == "__main__":
    main()
