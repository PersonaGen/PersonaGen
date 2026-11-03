#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YRBSS Evaluation Script

This script calculates distribution similarity metrics between LLM and Human responses
for YRBSS (Youth Risk Behavior Surveillance System) data.

Metrics computed:
- Avg_W1: Average Wasserstein distance across dimensions
- Frechet: Frechet distance between Gaussian distributions
- Sliced_W1: Sliced Wasserstein distance
- Energy_Dist: Energy distance

Features:
- Human data: Median imputation, using question-level features directly
- Method data: Read per-question, aligned with human data
- Compare each method file and compute metrics
- Results saved to CSV format
"""

import json
import argparse
import pathlib
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cdist, pdist
from scipy.linalg import sqrtm
from tqdm import tqdm


# Constants
NUM_RE = re.compile(r"[-+]?\d+(\.\d+)?")
SKIP_KEYS = {"idx", "persona_id", "persona_source", "persona_unique_id", "persona"}
METRICS = ["Avg_W1", "Frechet", "Sliced_W1", "Energy_Dist"]


def is_na(v):
    """Check if value is NA/None."""
    if v is None:
        return True
    if isinstance(v, str) and v.strip().lower() in {"na", "n/a", "nan", ""}:
        return True
    return False


def safe_float(v):
    """Extract first number from string and convert to float."""
    if is_na(v):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        if s and (s[0].isdigit() or (s[0] in "+-" and s[1:].lstrip().isdigit())):
            m = NUM_RE.match(s)
            if m:
                return float(m.group(0))
    return None


def load_human_imputed(path):
    """Load human data with median imputation."""
    rows = []
    with path.open(encoding="utf-8") as f:
        for ln in f:
            obj = json.loads(ln)
            rec = obj.get("responses", obj)   # Compatible with two formats
            clean = {
                k: (val if val is not None else np.nan)
                for k, v in rec.items()
                if k not in SKIP_KEYS
                for val in [safe_float(v)]
            }
            rows.append(clean)

    df_q = pd.DataFrame(rows)
    print(f"[human] Loaded {len(df_q)} records, {df_q.shape[1]} question columns")

    # Median imputation
    df_q_filled = df_q.fillna(df_q.median(numeric_only=True))
    print(f"[human] Remaining NaNs after imputation: {df_q_filled.isna().sum().sum()}")

    # Use question-level features directly
    print(f"[human] Using {df_q_filled.shape[1]} question-level features directly")
    return df_q_filled.astype(float)


def row_to_values(rec):
    """Extract values per question, maintaining same dimension as human data."""
    out = {}
    for k, v in rec.items():
        if k in SKIP_KEYS:
            continue
        val = safe_float(v)
        if val is not None:
            out[k] = val
    return out


def load_method_df(path):
    """Load method data."""
    rows = []
    with path.open(encoding="utf-8") as f:
        for ln in f:
            obj = json.loads(ln)
            rows.append(row_to_values(obj))
    return pd.DataFrame(rows).dropna(axis=1, how="all")


def frechet_gauss(X, Y):
    """Compute Frechet distance between Gaussian distributions."""
    mu1, mu2 = np.nanmean(X, 0), np.nanmean(Y, 0)
    C1, C2 = np.cov(X.T), np.cov(Y.T)

    if np.ndim(C1) == 0:           # 1-D special case
        if not np.isfinite(C1) or not np.isfinite(C2):
            return np.nan
        d = ((mu1 - mu2) ** 2 + C1 + C2 - 2 * np.sqrt(C1 * C2) + 1e-8) ** 0.5
    else:
        if not (np.all(np.isfinite(C1)) and np.all(np.isfinite(C2))):
            return np.nan
        try:
            prod = sqrtm(C1 @ C2)
            if np.iscomplexobj(prod):
                prod = prod.real
        except Exception:
            return np.nan
        diff = mu1 - mu2
        d = (diff @ diff + np.trace(C1 + C2 - 2 * prod) + 1e-8) ** 0.5
    return float(d)


def sliced_w1(X, Y, n_proj=100, seed=42):
    """Compute sliced Wasserstein distance."""
    rng = np.random.default_rng(seed)
    vecs = rng.normal(size=(n_proj, X.shape[1]))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return np.mean([wasserstein_distance(X @ v, Y @ v) for v in vecs])


def energy_nd(X, Y, max_samples=5000, seed=42):
    """Compute energy distance."""
    rng = np.random.default_rng(seed)
    X_s = X[rng.choice(len(X), min(len(X), max_samples), replace=False)]
    Y_s = Y[rng.choice(len(Y), min(len(Y), max_samples), replace=False)]
    return float(np.exp(0.5 * np.log(
        max(2 * cdist(X_s, Y_s).mean() - pdist(X_s).mean() - pdist(Y_s).mean(), 0.) + 1e-8
    )))


def compute_metrics(df_m, df_h):
    """Compute all metrics between method and human dataframes."""
    cols = df_m.columns.intersection(df_h.columns)
    if cols.empty:
        return {k: np.nan for k in METRICS + ["avg"]}

    A = df_m[cols].astype(float).values
    B = df_h[cols].astype(float).values

    # Remove columns where both A and B are all NaN
    keep = [i for i in range(A.shape[1])
            if not (np.all(np.isnan(A[:, i])) and np.all(np.isnan(B[:, i])))]
    if not keep:
        return {k: np.nan for k in METRICS + ["avg"]}
    A, B = A[:, keep], B[:, keep]

    # Per-column Wasserstein
    avg_w1 = np.mean([
        wasserstein_distance(
            A[:, i][~np.isnan(A[:, i])],
            B[:, i][~np.isnan(B[:, i])]
        ) for i in range(A.shape[1])
    ])

    # Multi-dimensional metrics
    non_nan_cols = ~(np.isnan(A).any(axis=0) | np.isnan(B).any(axis=0))
    A_nn, B_nn = A[:, non_nan_cols], B[:, non_nan_cols]
    if A_nn.shape[1] < 1 or len(A_nn) < 2 or len(B_nn) < 2:
        return {
            "Avg_W1": avg_w1,
            "Frechet": np.nan,
            "Sliced_W1": np.nan,
            "Energy_Dist": np.nan,
            "avg": np.nanmean([avg_w1])
        }

    A_nn = A_nn[~np.isnan(A_nn).any(axis=1)]
    B_nn = B_nn[~np.isnan(B_nn).any(axis=1)]
    if len(A_nn) < 2 or len(B_nn) < 2:
        return {
            "Avg_W1": avg_w1,
            "Frechet": np.nan,
            "Sliced_W1": np.nan,
            "Energy_Dist": np.nan,
            "avg": np.nanmean([avg_w1])
        }

    res = {
        "Avg_W1": avg_w1,
        "Frechet": frechet_gauss(A_nn, B_nn),
        "Sliced_W1": sliced_w1(A_nn, B_nn),
        "Energy_Dist": energy_nd(A_nn, B_nn),
    }
    res["avg"] = np.nanmean(list(res.values()))
    return res


def main():
    parser = argparse.ArgumentParser(description="YRBSS Evaluation Script")
    parser.add_argument("--method_dir", type=str, required=True,
                       help="Directory containing method result files")
    parser.add_argument("--human_file", type=str, required=True,
                       help="Path to human data JSONL file")
    parser.add_argument("--save_dir", type=str, required=True,
                       help="Directory to save results")

    args = parser.parse_args()

    # Setup paths
    METHOD_DIR = pathlib.Path(args.method_dir)
    HUMAN_FILE = pathlib.Path(args.human_file)
    SAVE_DIR = pathlib.Path(args.save_dir)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print("============== Loading Human Data with Median Imputation ==============")
    df_human = load_human_imputed(HUMAN_FILE)

    print("\n============== Processing Method Files ==============")
    results = {}
    method_files = sorted(METHOD_DIR.glob("*.jsonl"))
    for path in tqdm(method_files, desc="Processing Methods", ncols=100):
        stem = path.stem
        if stem.startswith("yrbss_prompts_"):
            stem = stem[len("yrbss_prompts_"):]
        stem = stem.split(".updated")[0]

        df_method = load_method_df(path)
        metrics = compute_metrics(df_method, df_human)
        results[stem] = metrics

    df_out = (
        pd.DataFrame.from_dict(results, orient="index")[METRICS + ["avg"]]
        .round(6)
        .rename_axis("Method")
        .sort_index()
    )

    csv_path = SAVE_DIR / "yrbss_metrics.csv"
    df_out.to_csv(csv_path)
    print("\n============== Done ==============")
    print(f"✅ Saved to: {csv_path.resolve()}")
    print(df_out)


if __name__ == "__main__":
    main()
