#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Out-of-Domain (OOD) personality metric comparison (CLI)

Computes distributional metrics between human survey responses and agent responses
for multiple OOD tasks (e.g., CFCS, FBPS, Duckworth). Now parameterized by CLI
arguments with no hardcoded paths.
"""

import json, pathlib, re, argparse, numpy as np, pandas as pd
from collections import defaultdict
from itertools import combinations
from scipy.stats import wasserstein_distance, entropy
from scipy.spatial.distance import pdist, cdist
from scipy.linalg import sqrtm
from sklearn.metrics import mutual_info_score
from popalign.evaluation.mmd import mmd_linear

HUMAN_DIR = None
AGENT_DIR = None
SAVE_DIR = None

BASELINES = []

EX_KEYS = {"persona_id", "persona_source", "persona_unique_id", "persona"}

# -------------------- 工具 --------------------
NUM_RE = re.compile(r"[-+]?\d+(\.\d+)?")
PREF_RE = re.compile(r"^([A-Za-z]+)")

def safe_float(v):
    if v is None:
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

def row_to_prefix_means(rec):
    bucket = defaultdict(list)
    for k, v in rec.items():
        if k in EX_KEYS:
            continue
        val = safe_float(v)
        if val is None:
            continue
        m = PREF_RE.match(k)
        if m:
            bucket[m.group(1)].append(val)
    return {p: np.mean(vals) for p, vals in bucket.items() if vals}

def load_df(path, human=False):
    rows = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            d = json.loads(ln)
            d = d["responses"] if human else d
            rows.append(row_to_prefix_means(d))
    return pd.DataFrame(rows).dropna(axis=1, how="all")

# ---------------- marginal helpers ----------------
def _marginal(fn, A, B, bins=20):
    vals = []
    for i in range(A.shape[1]):
        a = A[:, i][~np.isnan(A[:, i])]
        b = B[:, i][~np.isnan(B[:, i])]
        if len(a) == 0 or len(b) == 0:
            continue
        edges = np.histogram_bin_edges(np.concatenate([a, b]), bins=bins)
        Pa, _ = np.histogram(a, bins=edges, density=True)
        Pb, _ = np.histogram(b, bins=edges, density=True)
        Pa /= Pa.sum() + 1e-12
        Pb /= Pb.sum() + 1e-12
        vals.append(fn(Pa, Pb))
    return np.mean(vals) if vals else np.nan

marg_js = lambda A, B: _marginal(
    lambda P, Q: 0.5 * (entropy(P, 0.5 * (P + Q)) + entropy(Q, 0.5 * (P + Q))), A, B
)
marg_tv = lambda A, B: _marginal(
    lambda P, Q: 0.5 * np.abs(P - Q).sum(), A, B
)
marg_hel = lambda A, B: _marginal(
    lambda P, Q: np.linalg.norm(np.sqrt(P) - np.sqrt(Q)) / np.sqrt(2), A, B
)

# ---------------- joint helpers ----------------
def joint_js_tv_hel(A, B, bins=10):
    rng = [
        (
            np.nanmin(np.r_[A[:, i], B[:, i]]),
            np.nanmax(np.r_[A[:, i], B[:, i]]),
        )
        for i in range(A.shape[1])
    ]
    if any(not np.isfinite(lo) or lo == hi for lo, hi in rng):
        raise ValueError("Invalid range for joint histogram.")
    Ha, _ = np.histogramdd(A, bins=bins, range=rng)
    Hb, _ = np.histogramdd(B, bins=bins, range=rng)
    Pa, Pb = Ha.ravel(), Hb.ravel()
    Pa /= Pa.sum() + 1e-12
    Pb /= Pb.sum() + 1e-12
    M = 0.5 * (Pa + Pb)
    JS = 0.5 * (entropy(Pa, M) + entropy(Pb, M))
    TV = 0.5 * np.abs(Pa - Pb).sum()
    Hel = np.linalg.norm(np.sqrt(Pa) - np.sqrt(Pb)) / np.sqrt(2)
    return JS, TV, Hel

# ---------------- other metrics ----------------
def frechet_gauss(X, Y):
    mu1, mu2 = np.nanmean(X, 0), np.nanmean(Y, 0)
    c1, c2 = np.cov(X.T), np.cov(Y.T)
    if np.ndim(c1) == 0:
        cov_prod = np.sqrt(c1 * c2).item()
        return float((mu1 - mu2) ** 2 + (c1 + c2 - 2 * cov_prod))
    covm = sqrtm(c1 @ c2)
    covm = covm.real if np.iscomplexobj(covm) else covm
    diff = mu1 - mu2
    return float(diff.dot(diff) + np.trace(c1 + c2 - 2 * covm))

def sliced_w1(X, Y, n_proj=100):
    rng = np.random.default_rng(42)
    d = X.shape[1]
    vecs = rng.normal(size=(n_proj, d))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return np.mean([wasserstein_distance(X @ v, Y @ v) for v in vecs])

def cov_frob(X, Y):
    """
    Frobenius norm of covariance difference
    """
    c1, c2 = np.cov(X.T), np.cov(Y.T)
    return abs(c1 - c2) if np.ndim(c1) == 0 else np.linalg.norm(c1 - c2, "fro")

def energy_nd(X, Y, max_samples=5000):
    n_x = min(len(X), max_samples)
    n_y = min(len(Y), max_samples)
    idx_x = np.random.choice(len(X), n_x, replace=False)
    idx_y = np.random.choice(len(Y), n_y, replace=False)
    X_s = X[idx_x]
    Y_s = Y[idx_y]
    return max(
        2 * cdist(X_s, Y_s).mean()
        - pdist(X_s).mean()
        - pdist(Y_s).mean(),
        0,
    )

def mi_matrix(df):
    n = df.shape[1]
    M = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        M[i, j] = M[j, i] = mutual_info_score(
            None,
            None,
            contingency=np.histogram2d(
                df.iloc[:, i], df.iloc[:, j], bins=15
            )[0],
        )
    return M

# ---------------- compute ----------------
def compute_metrics(df_m, df_h, use_joint):
    common = sorted(df_m.columns.intersection(df_h.columns))
    A, B = df_m[common].values.astype(float), df_h[common].values.astype(float)

    keep = [
        i
        for i in range(len(common))
        if not (np.all(np.isnan(A[:, i])) and np.all(np.isnan(B[:, i])))
    ]
    if not keep:
        return dict.fromkeys(
            [
                "Avg_W1",
                "JS",
                "TV",
                "Hellinger",
                "Frechet",
                "MMD",
                "Sliced_W1",
                "Cov_Frob",
                "Energy_Distance",
                "MI_L2",
                "AVG",
            ],
            np.nan,
        ), False

    A, B = A[:, keep], B[:, keep]
    cols = [common[i] for i in keep]

    # Avg W1
    res = {}
    res["Avg_W1"] = np.mean(
        [
            wasserstein_distance(
                A[:, i][~np.isnan(A[:, i])],
                B[:, i][~np.isnan(B[:, i])],
            )
            for i in range(A.shape[1])
        ]
    )

    # JS / TV / Hel
    if use_joint:
        try:
            JS, TV, Hel = joint_js_tv_hel(A, B, bins=10)
        except Exception as e:
            print(f"    [fallback → marginal] {type(e).__name__}: {e}")
            use_joint = False
    if not use_joint:
        JS, TV, Hel = marg_js(A, B), marg_tv(A, B), marg_hel(A, B)

    res["JS"], res["TV"], res["Hellinger"] = JS, TV, Hel

    # Other metrics
    res["Frechet"] = frechet_gauss(A, B)
    res["MMD"] = mmd_linear(A, B)
    res["Sliced_W1"] = sliced_w1(A, B)
    res["Cov_Frob"] = cov_frob(A, B)
    res["Energy_Distance"] = energy_nd(A, B)
    res["MI_L2"] = np.linalg.norm(
        mi_matrix(pd.DataFrame(A, columns=cols))
        - mi_matrix(pd.DataFrame(B, columns=cols))
    )
    res["AVG"] = np.nanmean([v for k, v in res.items() if k != "AVG"])
    return res, use_joint

# ---------------- main ----------------
def main():
    global HUMAN_DIR, AGENT_DIR, SAVE_DIR, BASELINES
    parser = argparse.ArgumentParser(description="Compute OOD metrics between human and agent responses")
    parser.add_argument("--human_dir", required=True, help="Directory with *_human.jsonl files")
    parser.add_argument("--agent_dir", required=True, help="Directory with agent jsonl files")
    parser.add_argument("--out_dir", required=True, help="Directory to save metrics")
    parser.add_argument("--baselines", required=True, help="Comma-separated baseline prefixes to evaluate")
    args = parser.parse_args()

    HUMAN_DIR = pathlib.Path(args.human_dir)
    AGENT_DIR = pathlib.Path(args.agent_dir)
    SAVE_DIR = pathlib.Path(args.out_dir)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    BASELINES = [b.strip() for b in args.baselines.split(',') if b.strip()]
    task_files = {
        p.stem.replace("_human", ""): p
        for p in HUMAN_DIR.glob("*_human.jsonl")
    }
    for task, hp in task_files.items():
        csv_out = SAVE_DIR / f"{task}_metrics.csv"
        if csv_out.exists():
            print(f"Task {task} … done, skip.")
            continue

        print(f"\n=== Task: {task} ===")
        df_h = load_df(hp, human=True)
        use_joint = True
        if df_h.empty:
            print("  Human data empty, skip.")
            continue

        rows = []
        for bl in BASELINES:
            bp = next(AGENT_DIR.glob(f"{task}_{bl}_*.jsonl"), None)
            if bp is None:
                rows.append(
                    dict.fromkeys(
                        [
                            "Avg_W1",
                            "JS",
                            "TV",
                            "Hellinger",
                            "Frechet",
                            "MMD",
                            "Sliced_W1",
                            "Cov_Frob",
                            "Energy_Distance",
                            "MI_L2",
                            "AVG",
                        ],
                        np.nan,
                    )
                )
                continue

            df_b = load_df(bp)
            if df_b.empty:
                rows.append(
                    dict.fromkeys(
                        [
                            "Avg_W1",
                            "JS",
                            "TV",
                            "Hellinger",
                            "Frechet",
                            "MMD",
                            "Sliced_W1",
                            "Cov_Frob",
                            "Energy_Distance",
                            "MI_L2",
                            "AVG",
                        ],
                        np.nan,
                    )
                )
                continue

            common = df_h.columns.intersection(df_b.columns)
            if common.empty:
                rows.append(
                    dict.fromkeys(
                        [
                            "Avg_W1",
                            "JS",
                            "TV",
                            "Hellinger",
                            "Frechet",
                            "MMD",
                            "Sliced_W1",
                            "Cov_Frob",
                            "Energy_Distance",
                            "MI_L2",
                            "AVG",
                        ],
                        np.nan,
                    )
                )
                continue

            metrics, use_joint = compute_metrics(
                df_b[common], df_h[common], use_joint
            )
            rows.append(metrics)

        df = pd.DataFrame(rows, index=BASELINES).round(6)
        df = df[
            [
                "Avg_W1",
                "JS",
                "TV",
                "Hellinger",
                "Frechet",
                "MMD",
                "Sliced_W1",
                "Cov_Frob",
                "Energy_Distance",
                "MI_L2",
                "AVG",
            ]
        ]

        df.to_csv(csv_out, index_label="Model")
        with open(SAVE_DIR / f"{task}_metrics.txt", "w", encoding="utf-8") as f:
            f.write(f"=== Metrics for task: {task} ===\n\n")
            f.write(df.to_string(justify="center"))
            f.write("\n")
        print(f"  -> saved: {csv_out}")

if __name__ == "__main__":
    main()
