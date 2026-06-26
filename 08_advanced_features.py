"""
08_advanced_features.py — Experiment 3: Advanced Feature Engineering (Runs D, E, F)

Compares 3 alternative feature engineering approaches against Run B (5D baseline):

  Run D — Statistical PRI Aggregation (Window-Level)
  Run E — Frequency Domain (FFT on ToA Sequence)
  Run F — UMAP Manifold Reduction on 13D Space

Each approach runs the full HDBSCAN param grid on all 5 scenarios, evaluates
against ground truth, and produces a comparison table + updated run_comparison.csv.

Usage:
  python 08_advanced_features.py
"""

import os
import sys
import time
import hashlib
import json
import gc
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from hdbscan import HDBSCAN
from umap import UMAP
from sklearn.metrics import (
    v_measure_score,
    adjusted_rand_score,
    adjusted_mutual_info_score,
    homogeneity_score,
    completeness_score,
    silhouette_score,
)

np.random.seed(42)

load_dotenv(dotenv_path=Path(__file__).parent / ".env")
BASE_DIR = Path(__file__).parent.resolve()
SCENARIOS_DIR = Path(os.getenv("TSRD_SCENARIOS_DIR", BASE_DIR / "scenarios"))
RESULTS_DIR = Path(os.getenv("TSRD_RESULTS_DIR", BASE_DIR / "results"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SCENARIO_NAMES = ["stare_low", "stare_high", "scan_low", "scan_high", "mixed"]

# ---------------------------------------------------------------------------
# Run B baseline (from results_runB_backup)
# ---------------------------------------------------------------------------
RUN_B_BEST = {
    "stare_low":  {"v_measure": 0.4987, "ari": 0.4313, "noise_ratio": 0.2458},
    "stare_high": {"v_measure": 0.9020, "ari": 0.9693, "noise_ratio": 0.0338},
    "scan_low":   {"v_measure": 0.6479, "ari": 0.5970, "noise_ratio": 0.0164},
    "scan_high":  {"v_measure": 0.8709, "ari": 0.8362, "noise_ratio": 0.0430},
    "mixed":      {"v_measure": 0.8097, "ari": 0.7499, "noise_ratio": 0.0270},
}

# Run C baseline (from results/best_params.json)
RUN_C_BEST = {
    "stare_low":  {"v_measure": 0.4914, "ari": 0.4295, "noise_ratio": 0.1799},
    "stare_high": {"v_measure": 0.4432, "ari": 0.3783, "noise_ratio": 0.2605},
    "scan_low":   {"v_measure": 0.5070, "ari": 0.4577, "noise_ratio": 0.0669},
    "scan_high":  {"v_measure": 0.6478, "ari": 0.4954, "noise_ratio": 0.1448},
    "mixed":      {"v_measure": 0.6147, "ari": 0.5219, "noise_ratio": 0.0532},
}

# ---------------------------------------------------------------------------
# HDBSCAN parameter grid (same as original)
# ---------------------------------------------------------------------------
PARAM_GRID = [
    {"min_cluster_size": 10, "min_samples": None, "cluster_selection_epsilon": 0.0, "cluster_selection_method": "eom", "metric": "euclidean"},
    {"min_cluster_size": 10, "min_samples": None, "cluster_selection_epsilon": 0.1, "cluster_selection_method": "eom", "metric": "euclidean"},
    {"min_cluster_size": 10, "min_samples": 10,   "cluster_selection_epsilon": 0.0, "cluster_selection_method": "eom", "metric": "euclidean"},
    {"min_cluster_size": 10, "min_samples": 10,   "cluster_selection_epsilon": 0.1, "cluster_selection_method": "eom", "metric": "euclidean"},
    {"min_cluster_size": 20, "min_samples": None, "cluster_selection_epsilon": 0.0, "cluster_selection_method": "eom", "metric": "euclidean"},
    {"min_cluster_size": 20, "min_samples": None, "cluster_selection_epsilon": 0.1, "cluster_selection_method": "eom", "metric": "euclidean"},
    {"min_cluster_size": 20, "min_samples": 20,   "cluster_selection_epsilon": 0.0, "cluster_selection_method": "eom", "metric": "euclidean"},
    {"min_cluster_size": 20, "min_samples": 20,   "cluster_selection_epsilon": 0.1, "cluster_selection_method": "eom", "metric": "euclidean"},
    {"min_cluster_size": 50, "min_samples": None, "cluster_selection_epsilon": 0.0, "cluster_selection_method": "eom", "metric": "euclidean"},
    {"min_cluster_size": 50, "min_samples": None, "cluster_selection_epsilon": 0.1, "cluster_selection_method": "eom", "metric": "euclidean"},
    {"min_cluster_size": 50, "min_samples": 50,   "cluster_selection_epsilon": 0.0, "cluster_selection_method": "eom", "metric": "euclidean"},
    {"min_cluster_size": 50, "min_samples": 50,   "cluster_selection_epsilon": 0.1, "cluster_selection_method": "eom", "metric": "euclidean"},
]

APPROACHES = {
    "D": {"label": "Run_D_PRIstat", "n_features": 8},
    "E": {"label": "Run_E_FFT",     "n_features": 8},
    "F": {"label": "Run_F_UMAP13D", "n_features": 3},
}

SILHOUETTE_APPROACHES = {"D", "E"}  # silhouette only valid on >8D


def param_to_hash(params):
    key = json.dumps(params, sort_keys=True)
    return hashlib.md5(key.encode()).hexdigest()[:8]


def param_to_label(params):
    ms = params["min_samples"] if params["min_samples"] is not None else "auto"
    return f"cs{params['min_cluster_size']}_ms{ms}_eps{params['cluster_selection_epsilon']}"


# ---------------------------------------------------------------------------
# Feature Engineering Functions
# ---------------------------------------------------------------------------

def add_pri_features(X_window):
    """Original 8 PRI-derived features (Run C) — for Approach 3 input."""
    n = len(X_window)
    ToA = X_window[:, 0]
    Freq = X_window[:, 1]
    PW = X_window[:, 2]
    lag1 = np.zeros(n); lead1 = np.zeros(n)
    lag2 = np.zeros(n); lead2 = np.zeros(n)
    lag3 = np.zeros(n); lead3 = np.zeros(n)
    delta_freq = np.zeros(n); delta_pw = np.zeros(n)
    lag1[1:] = ToA[1:] - ToA[:-1]
    lead1[:-1] = ToA[1:] - ToA[:-1]
    lag2[2:] = ToA[2:] - ToA[:-2]
    lead2[:-2] = ToA[2:] - ToA[:-2]
    lag3[3:] = ToA[3:] - ToA[:-3]
    lead3[:-3] = ToA[3:] - ToA[:-3]
    delta_freq[1:] = np.abs(Freq[1:] - Freq[:-1])
    delta_pw[1:] = np.abs(PW[1:] - PW[:-1])
    return np.column_stack([X_window, lag1, lead1, lag2, lead2, lag3, lead3, delta_freq, delta_pw])


def build_features_approach_D(X_window):
    """Run D: 5 PDW + 3 statistical PRI features (median, IQR, entropy)."""
    ToA = X_window[:, 0]
    pris = np.diff(ToA)
    median_pri = np.median(pris)
    iqr_pri = np.percentile(pris, 75) - np.percentile(pris, 25)
    hist, _ = np.histogram(pris, bins=50)
    hist = hist / (hist.sum() + 1e-12)
    pri_entropy = -np.sum(hist * np.log(hist + 1e-12))
    stat_features = np.tile([median_pri, iqr_pri, pri_entropy], (len(X_window), 1))
    return np.column_stack([X_window, stat_features])


def build_features_approach_E(X_window):
    """Run E: 5 PDW + 3 dominant FFT frequencies from detrended ToA."""
    ToA = X_window[:, 0]
    x = np.arange(len(ToA))
    coeffs = np.polyfit(x, ToA, 1)
    trend = np.polyval(coeffs, x)
    toa_detrended = ToA - trend
    fft_mag = np.abs(np.fft.rfft(toa_detrended))
    freqs = np.fft.rfftfreq(len(toa_detrended))
    fft_mag = fft_mag[1:]
    freqs = freqs[1:]
    n_top = min(3, len(freqs))
    top_indices = np.argsort(fft_mag)[-n_top:][::-1]
    dominant = np.zeros(3)
    for i, idx in enumerate(top_indices):
        dominant[i] = freqs[idx]
    fft_features = np.tile(dominant, (len(X_window), 1))
    return np.column_stack([X_window, fft_features])


def build_features_approach_F(X_window):
    """Run F: 13D (5 PDW + 8 PRI-derived) — UMAP reduction applied later."""
    return add_pri_features(X_window)


# ---------------------------------------------------------------------------
# Clustering and Evaluation
# ---------------------------------------------------------------------------

def cluster_and_evaluate(X, y_true, params):
    """Run HDBSCAN on feature matrix X and evaluate against y_true."""
    clusterer = HDBSCAN(**params)
    labels = clusterer.fit_predict(X)
    y_pred = np.array(labels)
    unique_pred = np.unique(y_pred)
    unique_true = np.unique(y_true)

    n_clusters = len([u for u in unique_pred if u != -1])
    n_noise = int((y_pred == -1).sum())

    if len(unique_pred) <= 1:
        metrics = {
            "v_measure": 0.0, "homogeneity": 1.0 if len(unique_pred) == 1 else 0.0,
            "completeness": 0.0, "ari": 0.0, "ami": 0.0,
            "silhouette": 0.0,
            "n_clusters_true": len(unique_true),
            "n_clusters_pred": n_clusters,
            "noise_ratio": float((y_pred == -1).mean()),
        }
    else:
        try:
            sil = 0.0
            mask = y_pred != -1
            if mask.sum() > 1 and len(np.unique(y_pred[mask])) > 1:
                sil = float(silhouette_score(X[mask], y_pred[mask]))
            metrics = {
                "v_measure": v_measure_score(y_true, y_pred),
                "homogeneity": homogeneity_score(y_true, y_pred),
                "completeness": completeness_score(y_true, y_pred),
                "ari": adjusted_rand_score(y_true, y_pred),
                "ami": adjusted_mutual_info_score(y_true, y_pred),
                "silhouette": sil,
                "n_clusters_true": len(unique_true),
                "n_clusters_pred": n_clusters,
                "noise_ratio": float((y_pred == -1).mean()),
            }
        except Exception:
            metrics = {
                "v_measure": 0.0, "homogeneity": 0.0, "completeness": 0.0,
                "ari": 0.0, "ami": 0.0, "silhouette": 0.0,
                "n_clusters_true": len(unique_true),
                "n_clusters_pred": n_clusters,
                "noise_ratio": float((y_pred == -1).mean()),
            }

    return metrics, y_pred.tolist()


def cache_path(approach_label, scenario, w_idx, param_hash):
    return RESULTS_DIR / f"{approach_label}_{scenario}_w{w_idx:04d}_p{param_hash}.json"


# ---------------------------------------------------------------------------
# Process One Approach + Scenario
# ---------------------------------------------------------------------------

def process_approach_scenario(approach_label, scenario_name, n_jobs=4):
    """Feature engineer, cluster, and evaluate for one approach + scenario."""
    data_path = SCENARIOS_DIR / f"{scenario_name}.npz"
    if not data_path.exists():
        print(f"    [SKIP] Scenario file not found: {data_path}")
        return None

    data = np.load(data_path, allow_pickle=True)
    X = data["X"]
    y_true = data["y"]
    data.close()
    n_windows = X.shape[0]
    n_params = len(PARAM_GRID)

    print(f"\n  [{approach_label}] Scenario: {scenario_name}")
    print(f"    Windows: {n_windows}  |  Params: {n_params}  |  Total fits: {n_windows * n_params}")

    # Build features
    builder = {
        "Run_D_PRIstat": build_features_approach_D,
        "Run_E_FFT":     build_features_approach_E,
        "Run_F_UMAP13D": build_features_approach_F,
    }[approach_label]

    # Determine feature dimension from a sample build
    sample_feat = builder(X[0])
    n_feat_in = sample_feat.shape[1]
    X_feat = np.zeros((n_windows, X.shape[1], n_feat_in))
    for w_idx in range(n_windows):
        X_feat[w_idx] = builder(X[w_idx])

    n_feat = n_feat_in if approach_label != "Run_F_UMAP13D" else 3
    print(f"    Feature matrix: ({X_feat.shape[0]}, {X_feat.shape[1]}, {X_feat.shape[2]})")

    # Per-scenario normalization
    feat_mean = X_feat.mean(axis=(0, 1))
    feat_std = X_feat.std(axis=(0, 1))
    feat_std[feat_std == 0] = 1.0
    print(f"    Normalization: {n_feat} features normalized")

    completed_fits = 0

    for w_idx in range(n_windows):
        X_norm = (X_feat[w_idx] - feat_mean) / feat_std

        # For Run F: UMAP reduce 13D -> 3D before HDBSCAN
        if approach_label == "Run_F_UMAP13D":
            umap = UMAP(n_components=3, metric="euclidean", n_neighbors=15,
                        min_dist=0.1, random_state=42 + w_idx, n_jobs=1)
            X_cluster = umap.fit_transform(X_norm)
        else:
            X_cluster = X_norm

        for p_idx, params in enumerate(PARAM_GRID):
            phash = param_to_hash(params)
            cpath = cache_path(approach_label, scenario_name, w_idx, phash)
            if cpath.exists():
                continue
            try:
                metrics, labels = cluster_and_evaluate(X_cluster, y_true[w_idx], params)
                result = {
                    "labels": labels,
                    "n_clusters": metrics["n_clusters_pred"],
                    "n_noise": int(metrics["noise_ratio"] * len(labels)),
                    "n_total": len(labels),
                    "metrics": {k: v for k, v in metrics.items() if k != "n_clusters_pred"},
                }
                with open(cpath, "w") as f:
                    json.dump(result, f)
                completed_fits += 1
            except Exception as e:
                print(f"    [WARN] Window {w_idx}, param {phash} failed: {e}")
                continue

    print(f"    Completed fits this run: {completed_fits}")
    return completed_fits


# ---------------------------------------------------------------------------
# Evaluate a full approach across all scenarios
# ---------------------------------------------------------------------------

def evaluate_approach(approach_label):
    """Evaluate cached results for one approach, return summary DataFrame."""
    print(f"\n{'=' * 60}")
    print(f"Evaluating {approach_label}")
    print(f"{'=' * 60}")

    all_rows = []
    for scenario_name in SCENARIO_NAMES:
        data_path = SCENARIOS_DIR / f"{scenario_name}.npz"
        if not data_path.exists():
            continue
        data = np.load(data_path, allow_pickle=True)
        y_all = data["y"]
        n_windows = y_all.shape[0]
        data.close()

        for w_idx in range(n_windows):
            for params in PARAM_GRID:
                phash = param_to_hash(params)
                cpath = cache_path(approach_label, scenario_name, w_idx, phash)
                if not cpath.exists():
                    continue
                with open(cpath) as f:
                    result = json.load(f)
                m = result.get("metrics", {})
                row = {
                    "scenario": scenario_name,
                    "param_label": param_to_label(params),
                    "param_hash": phash,
                    "window_idx": w_idx,
                    "v_measure": m.get("v_measure", 0),
                    "homogeneity": m.get("homogeneity", 0),
                    "completeness": m.get("completeness", 0),
                    "ari": m.get("ari", 0),
                    "ami": m.get("ami", 0),
                    "silhouette": m.get("silhouette", 0),
                    "n_clusters_pred": result.get("n_clusters", 0),
                    "noise_ratio": m.get("noise_ratio", 1),
                }
                all_rows.append(row)

    if not all_rows:
        print(f"  [WARN] No cached results for {approach_label}")
        return None

    df = pd.DataFrame(all_rows)

    agg = df.groupby(["scenario", "param_label"]).agg(
        v_measure_mean=("v_measure", "mean"),
        v_measure_std=("v_measure", "std"),
        homogeneity_mean=("homogeneity", "mean"),
        completeness_mean=("completeness", "mean"),
        ari_mean=("ari", "mean"),
        ami_mean=("ami", "mean"),
        silhouette_mean=("silhouette", "mean"),
        n_clusters_pred_mean=("n_clusters_pred", "mean"),
        noise_ratio_mean=("noise_ratio", "mean"),
        n_windows=("window_idx", "count"),
    ).reset_index()

    agg["v_measure_rank"] = agg.groupby("scenario")["v_measure_mean"].rank(ascending=False)
    agg = agg.sort_values(["scenario", "v_measure_rank"])

    csv_path = RESULTS_DIR / f"summary_{approach_label.lower()}.csv"
    agg.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"  Saved summary: {csv_path}")

    # Best params per scenario
    best = agg.loc[agg.groupby("scenario")["v_measure_rank"].idxmin()]
    best_params = {}
    for _, row in best.iterrows():
        best_params[row["scenario"]] = {
            "param_label": row["param_label"],
            "v_measure": round(row["v_measure_mean"], 4),
            "ari": round(row["ari_mean"], 4),
            "noise_ratio": round(row["noise_ratio_mean"], 4),
            "silhouette": round(row["silhouette_mean"], 4),
            "n_clusters": round(row["n_clusters_pred_mean"], 2),
        }

    json_path = RESULTS_DIR / f"best_params_{approach_label.lower()}.json"
    with open(json_path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"  Saved best params: {json_path}")

    return best_params


# ---------------------------------------------------------------------------
# Main: Run all approaches
# ---------------------------------------------------------------------------

def run_all():
    print("=" * 60)
    print("Experiment 3: Advanced Feature Engineering")
    print("=" * 60)
    print(f"  Approaches: D (PRI statistics), E (FFT), F (UMAP 13D->3D)")
    print(f"  Scenarios:  {', '.join(SCENARIO_NAMES)}")
    print(f"  Params:     {len(PARAM_GRID)} combinations each")
    print(f"  Results in: {RESULTS_DIR}")
    print()

    start_total = time.time()

    # Phase 1: Cluster each approach
    approach_labels = [v["label"] for v in APPROACHES.values()]
    for approach_label in approach_labels:
        print(f"\n{'=' * 60}")
        print(f"Running {approach_label}")
        print(f"{'=' * 60}")
        total = 0
        for name in SCENARIO_NAMES:
            n = process_approach_scenario(approach_label, name, n_jobs=4)
            if n is not None:
                total += n
            gc.collect()
        print(f"  Total new fits for {approach_label}: {total}")

    # Phase 2: Evaluate all approaches
    all_best = {}
    for approach_label in approach_labels:
        best = evaluate_approach(approach_label)
        if best is not None:
            all_best[approach_label] = best

    # Phase 3: Comparison table
    print(f"\n{'=' * 150}")
    print("COMPARISON: Best V-measure per Scenario")
    print(f"{'=' * 150}")
    header = f"{'Scenario':<12}"
    for label in approach_labels:
        header += f" {label:>18}"
    header += f" {'Run_B_5D':>10} {'Run_C_13D':>10}"
    print(header)
    print("-" * 150)

    for s in SCENARIO_NAMES:
        row = f"{s:<12}"
        for label in approach_labels:
            bp = all_best.get(label, {}).get(s, {})
            row += f" {bp.get('v_measure', 0):>18.4f}"
        row += f" {RUN_B_BEST[s]['v_measure']:>10.4f} {RUN_C_BEST[s]['v_measure']:>10.4f}"
        print(row)

    print("-" * 150)

    # Phase 4: Delta from Run B
    print(f"\n{'=' * 150}")
    print("DELTA FROM RUN B BASELINE (V-measure % change)")
    print(f"{'=' * 150}")
    header = f"{'Scenario':<12}"
    for label in approach_labels:
        header += f" {label:>18}"
    header += f" {'Run_B':>10} {'Run_C':>10}"
    print(header)
    print("-" * 150)

    for s in SCENARIO_NAMES:
        row = f"{s:<12}"
        base_b = RUN_B_BEST[s]["v_measure"]
        base_c = RUN_C_BEST[s]["v_measure"]
        for label in approach_labels:
            bp = all_best.get(label, {}).get(s, {})
            v = bp.get("v_measure", 0)
            delta = (v - base_b) / base_b * 100 if base_b > 0 else 0
            row += f" {delta:>+17.1f}%"
        row += f" {'—':>10} {(base_c - base_b) / base_b * 100:>+9.1f}%"
        print(row)

    # Phase 5: Silhouette comparison (only for D, E)
    print(f"\n{'=' * 80}")
    print("SILHOUETTE SCORES (best params, non-noise points)")
    print(f"{'=' * 80}")
    print(f"{'Scenario':<12} {'Run_D_PRIstat':>15} {'Run_E_FFT':>15}")
    for s in SCENARIO_NAMES:
        d_sil = all_best.get("Run_D_PRIstat", {}).get(s, {}).get("silhouette", 0)
        e_sil = all_best.get("Run_E_FFT", {}).get(s, {}).get("silhouette", 0)
        print(f"{s:<12} {d_sil:>15.4f} {e_sil:>15.4f}")

    # Phase 6: Cluster count comparison
    print(f"\n{'=' * 150}")
    print("CLUSTER COUNTS (best params, mean predicted clusters)")
    print(f"{'=' * 150}")
    header = f"{'Scenario':<12}"
    for label in approach_labels:
        header += f" {label:>18}"
    header += f" {'Run_B':>10} {'Run_C':>10}"
    print(header)
    print("-" * 150)
    for s in SCENARIO_NAMES:
        row = f"{s:<12}"
        for label in approach_labels:
            bp = all_best.get(label, {}).get(s, {})
            row += f" {bp.get('n_clusters', 0):>18.1f}"
        row += f" {'—':>10} {'—':>10}"
        print(row)

    # Phase 7: Update run_comparison.csv
    print(f"\n{'=' * 60}")
    print("Updating run_comparison.csv")
    print(f"{'=' * 60}")

    comp_path = RESULTS_DIR / "run_comparison.csv"
    if comp_path.exists():
        comp_df = pd.read_csv(comp_path)
    else:
        comp_df = pd.DataFrame()

    # Add new columns for Runs D, E, F
    for label in approach_labels:
        v_col = f"{label}_v"
        ari_col = f"{label}_ari"
        noise_col = f"{label}_noise"
        param_col = f"{label}_param"

        comp_df[v_col] = np.nan
        comp_df[ari_col] = np.nan
        comp_df[noise_col] = np.nan
        comp_df[param_col] = ""

        for _, row in comp_df.iterrows():
            s = row["scenario"]
            bp = all_best.get(label, {}).get(s, {})
            comp_df.loc[comp_df["scenario"] == s, v_col] = bp.get("v_measure", np.nan)
            comp_df.loc[comp_df["scenario"] == s, ari_col] = bp.get("ari", np.nan)
            comp_df.loc[comp_df["scenario"] == s, noise_col] = bp.get("noise_ratio", np.nan)
            comp_df.loc[comp_df["scenario"] == s, param_col] = bp.get("param_label", "")

    comp_df.to_csv(comp_path, index=False, float_format="%.4f")
    print(f"  Updated: {comp_path}")

    # Final verdict
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    for label in approach_labels:
        wins = sum(1 for s in SCENARIO_NAMES
                   if RUN_B_BEST[s]["v_measure"] <= all_best.get(label, {}).get(s, {}).get("v_measure", 0))
        print(f"  {label}: {wins}/5 scenarios match or beat Run B baseline")

    elapsed = time.time() - start_total
    mins, secs = divmod(elapsed, 60)
    print(f"\n  Total time: {int(mins)}m {int(secs)}s")


if __name__ == "__main__":
    run_all()
