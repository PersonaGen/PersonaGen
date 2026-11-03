#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Big Five Persona Evaluation - Baseline Comparison (CLI)

This script computes distributional metrics between model-generated persona responses
and human reference distributions for the Big Five (EXT, AGR, CSN, EST, OPN).

It supports three categories of inputs:
- origin: your raw persona set (e.g., Ours/Raw)
- resampled: your resampled persona set (optional)
- baselines: one or more baseline files (comma-separated)

Outputs are saved into the specified output directory:
- baseline_metrics.csv
- baseline_metrics.txt

Example:
  python cal_llm_with_human/get_baseline_result_llama.py \
    --human_path path/to/big5_human.jsonl \
    --origin_path path/to/origin.jsonl \
    --resampled_jsonl path/to/resample.jsonl \
    --baselines path/to/b1.jsonl,path/to/b2.jsonl \
    --out_dir results/bigfive
"""

import os
import json
import pathlib
from itertools import combinations
from typing import Optional, List
import argparse

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance, entropy
from scipy.spatial.distance import pdist, cdist
from scipy.linalg import sqrtm
from sklearn.metrics import mutual_info_score
from popalign.evaluation.mmd import mmd_linear
from scipy.interpolate import interp1d

# ------------------------- Constants -------------------------

TRAITS = ["EXT", "AGR", "CSN", "EST", "OPN"]

# --------------------- Utility Functions ---------------------

def load_jsonl(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        return pd.DataFrame(json.loads(l) for l in f).dropna(subset=TRAITS)


def quantile_warp_csn(df_src, df_ref, col="CSN", n_bins=1000):
    qs = np.linspace(0, 1, n_bins)
    icdf_src = np.quantile(df_src[col], qs)
    icdf_ref = np.quantile(df_ref[col], qs)
    forward = interp1d(
        icdf_src,
        icdf_ref,
        bounds_error=False,
        fill_value=(icdf_ref[0], icdf_ref[-1]),
        assume_sorted=True,
    )
    df_src[col] = forward(df_src[col])
    return df_src


def _joint_hist(a: np.ndarray, b: np.ndarray, bins: int = 40):
    # Keep only occupied bins. A dense 40**5 histogram allocates over 800 MB
    # per population; omitted jointly empty bins contribute zero to all metrics.
    a, b = np.asarray(a), np.asarray(b)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("Inputs must be 2D arrays with matching trait dimensions.")
    edges = np.linspace(1.0, 5.0, bins + 1)

    def occupied(values):
        valid = np.all(np.isfinite(values) & (values >= 1) & (values <= 5), axis=1)
        indices = np.searchsorted(edges, values[valid], side="right") - 1
        indices[indices == bins] = bins - 1  # Include the rightmost edge, like histogramdd.
        return indices

    ia, ib = occupied(a), occupied(b)
    if not len(ia) or not len(ib):
        raise ValueError("Each population must have at least one finite row within [1, 5].")
    _, inverse = np.unique(np.vstack([ia, ib]), axis=0, return_inverse=True)
    count = int(inverse.max()) + 1
    Pa = np.bincount(inverse[:len(ia)], minlength=count).astype(float)
    Pb = np.bincount(inverse[len(ia):], minlength=count).astype(float)
    return Pa / Pa.sum(), Pb / Pb.sum()


def js_div(p: np.ndarray, q: np.ndarray, bins: int = 40) -> float:
    P, Q = _joint_hist(p, q, bins)
    M = 0.5 * (P + Q)
    return 0.5 * entropy(P, M) + 0.5 * entropy(Q, M)


def tv_dist(p: np.ndarray, q: np.ndarray, bins: int = 40) -> float:
    P, Q = _joint_hist(p, q, bins)
    return 0.5 * np.abs(P - Q).sum()


def hellinger_dist(p: np.ndarray, q: np.ndarray, bins: int = 40) -> float:
    P, Q = _joint_hist(p, q, bins)
    return (1.0 / np.sqrt(2.0)) * np.linalg.norm(np.sqrt(P) - np.sqrt(Q))


def frechet_gaussian(X: np.ndarray, Y: np.ndarray) -> float:
    """Frechet distance with robust fallback when sqrtm fails/returns NaN."""
    mu_x, mu_y = X.mean(axis=0), Y.mean(axis=0)
    cov_x, cov_y = np.cov(X.T), np.cov(Y.T)

    diff = mu_x - mu_y
    diff_sq = diff.dot(diff)

    try:
        cov_prod = sqrtm(cov_x.dot(cov_y))
        if np.iscomplexobj(cov_prod):
            cov_prod = cov_prod.real
        if np.isnan(cov_prod).any():
            raise ValueError("sqrtm produced NaNs")
    except Exception as e:
        print(f"sqrtm failed or produced NaN ({e}); fallback to cov_prod = 0")
        cov_prod = np.zeros_like(cov_x)

    trace_term = np.trace(cov_x + cov_y - 2 * cov_prod)

    frechet = diff_sq + trace_term
    return frechet


def sliced_wasserstein(X: np.ndarray, Y: np.ndarray, n_proj: int = 100,
                       random_state: int = 42) -> float:
    rng = np.random.default_rng(random_state)
    d = X.shape[1]
    sw = 0.0
    for _ in range(n_proj):
        v = rng.normal(size=d)
        v /= np.linalg.norm(v) + 1e-12
        Xp = X.dot(v)
        Yp = Y.dot(v)
        sw += wasserstein_distance(Xp, Yp)
    return sw / n_proj


def marg_w1(A: pd.DataFrame, B: pd.DataFrame):
    return {t: wasserstein_distance(A[t], B[t]) for t in TRAITS}


def cov_frobenius(df1: pd.DataFrame, df2: pd.DataFrame) -> float:
    return np.linalg.norm(np.cov(df1.T) - np.cov(df2.T), ord="fro")


def energy_distance_nd(
    X: np.ndarray,
    Y: np.ndarray,
    max_samples: int = 50_000,
    random_state: Optional[int] = 42,
) -> float:
    rng = np.random.default_rng(random_state)
    n_x = min(len(X), max_samples)
    n_y = min(len(Y), max_samples)
    idx_x = rng.choice(len(X), size=n_x, replace=False)
    idx_y = rng.choice(len(Y), size=n_y, replace=False)
    X_s = X[idx_x]
    Y_s = Y[idx_y]
    d_ab = cdist(X_s, Y_s, metric="euclidean")
    d_aa = pdist(X_s, metric="euclidean")
    d_bb = pdist(Y_s, metric="euclidean")
    return max(2.0 * d_ab.mean() - d_aa.mean() - d_bb.mean(), 0.0)


def mi_matrix(df: pd.DataFrame) -> np.ndarray:
    mat = np.zeros((5, 5))
    for i, j in combinations(range(5), 2):
        val = mutual_info_score(
            None,
            None,
            contingency=np.histogram2d(
                df.iloc[:, i], df.iloc[:, j], bins=15
            )[0],
        )
        mat[i, j] = mat[j, i] = val
    return mat


# ------------------ Compute Metrics for one DataFrame ------------------

def compute_metrics(df_x: pd.DataFrame, df_h: pd.DataFrame) -> dict:
    X = df_x[TRAITS].values.astype(np.float64)
    Xh = df_h[TRAITS].values.astype(np.float64)
    res = {}

    # 1) trait-wise W1
    w1 = marg_w1(df_x, df_h)
    for t in TRAITS:
        res[f"W1_{t}"] = w1[t]

    # 2) joint-distribution metrics
    res["JS"]        = js_div(X, Xh)
    res["TV"]        = tv_dist(X, Xh)
    res["Hellinger"] = hellinger_dist(X, Xh)
    res["Frechet"]   = frechet_gaussian(X, Xh)
    res["MMD"]       = mmd_linear(X, Xh)
    res["Sliced_W1"] = sliced_wasserstein(X, Xh)

    # 3) shape/dependency metrics
    res["Cov_Frob"]        = cov_frobenius(df_x[TRAITS], df_h[TRAITS])
    res["Energy_Distance"] = energy_distance_nd(X, Xh)
    res["MI_L2"]           = np.linalg.norm(mi_matrix(df_h[TRAITS]) - mi_matrix(df_x[TRAITS]))

    return res


# -------------------------- Main Routine --------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Big Five metrics between model personas and human data")
    parser.add_argument("--human_path", required=True, help="Path to human Big Five JSONL")
    parser.add_argument("--origin_path", required=True, help="Path to origin persona JSONL")
    parser.add_argument("--resampled_jsonl", default=None, help="Path to resampled persona JSONL (optional)")
    parser.add_argument("--baselines", default="", help="Comma-separated list of baseline JSONL files")
    parser.add_argument("--out_dir", required=True, help="Directory to save outputs")
    parser.add_argument("--quantile_warp_csn", action="store_true", help="Apply quantile warping on CSN vs human")
    args = parser.parse_args()

    pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    df_h = load_jsonl(args.human_path)

    rows = {}
    rows["origin"] = compute_metrics(load_jsonl(args.origin_path), df_h)

    if args.resampled_jsonl:
        df_r = load_jsonl(args.resampled_jsonl)
        if args.quantile_warp_csn:
            df_r = quantile_warp_csn(df_r, df_h, col="CSN")
        rows["resample"] = compute_metrics(df_r, df_h)

    baseline_paths: List[str] = [p for p in args.baselines.split(",") if p.strip()]
    for path in baseline_paths:
        df_b = load_jsonl(path)
        fname = pathlib.Path(path).stem
        if "persona_source" in df_b.columns:
            name = str(df_b.loc[0, "persona_source"]) or fname
        else:
            name = fname
        rows[name] = compute_metrics(df_b, df_h)

    df_metrics = pd.DataFrame.from_dict(rows, orient="index")
    csv_path = os.path.join(args.out_dir, "baseline_metrics.csv")
    txt_path = os.path.join(args.out_dir, "baseline_metrics.txt")

    df_metrics.to_csv(csv_path, index_label="model")

    pretty_df = df_metrics.apply(lambda col: col.map(lambda x: f"{x:.4f}"))
    print("\n=== All Metrics ===")
    print(pretty_df)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=== All Metrics ===\n")
        f.write(pretty_df.to_string())
        f.write("\n")

    print(f"\n✅ CSV saved to: {csv_path}")
    print(f"✅ TXT saved to: {txt_path}")


if __name__ == "__main__":
    main()
