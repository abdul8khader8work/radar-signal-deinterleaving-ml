"""
06_run_umap_hdbscan.py — Experiment 2A: UMAP + HDBSCAN on Run B features

Compares:
  - Run B baseline (5D HDBSCAN) — from backup best_params.json
  - UMAP 2D + HDBSCAN
  - UMAP 3D + HDBSCAN

Each window: normalize → UMAP reduce → HDBSCAN cluster → evaluate
Results saved to results_experiment2/
"""

import os, sys, json, time, gc, hashlib
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from hdbscan import HDBSCAN
from umap import UMAP
from sklearn.metrics import v_measure_score, adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score, completeness_score

load_dotenv(dotenv_path=Path(__file__).parent / ".env")
BASE_DIR = Path(__file__).parent.resolve()
SCENARIOS_DIR = Path(os.getenv("TSRD_SCENARIOS_DIR", BASE_DIR / "scenarios"))
RESULTS_DIR = BASE_DIR / "results_experiment2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

# ── Run B best params per scenario ──
BEST_PARAMS = {
    "stare_low":  {"min_cluster_size": 50, "min_samples": 50, "cluster_selection_epsilon": 0.0, "cluster_selection_method": "eom", "metric": "euclidean"},
    "stare_high": {"min_cluster_size": 10, "min_samples": 10, "cluster_selection_epsilon": 0.1, "cluster_selection_method": "eom", "metric": "euclidean"},
    "scan_low":   {"min_cluster_size": 10, "min_samples": 10, "cluster_selection_epsilon": 0.1, "cluster_selection_method": "eom", "metric": "euclidean"},
    "scan_high":  {"min_cluster_size": 10, "min_samples": 10, "cluster_selection_epsilon": 0.1, "cluster_selection_method": "eom", "metric": "euclidean"},
    "mixed":      {"min_cluster_size": 10, "min_samples": 10, "cluster_selection_epsilon": 0.1, "cluster_selection_method": "eom", "metric": "euclidean"},
}

# ── Run B baseline V-measure (from results_runB_backup/summary_metrics.csv) ──
RUN_B_BASELINE = {
    "stare_low":  0.4987,
    "stare_high": 0.9020,
    "scan_low":   0.6479,
    "scan_high":  0.8709,
    "mixed":      0.8097,
}


def param_hash(params):
    return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]


def evaluate(y_true, y_pred):
    y_pred = np.array(y_pred)
    unique_pred = np.unique(y_pred)
    unique_true = np.unique(y_true)
    if len(unique_pred) <= 1:
        return {"v_measure": 0.0, "homogeneity": 1.0 if len(unique_pred) == 1 else 0.0, "completeness": 0.0, "ari": 0.0, "ami": 0.0, "noise_ratio": float((y_pred == -1).mean())}
    try:
        return {"v_measure": v_measure_score(y_true, y_pred), "homogeneity": homogeneity_score(y_true, y_pred), "completeness": completeness_score(y_true, y_pred), "ari": adjusted_rand_score(y_true, y_pred), "ami": adjusted_mutual_info_score(y_true, y_pred), "noise_ratio": float((y_pred == -1).mean())}
    except:
        return {"v_measure": 0.0, "homogeneity": 0.0, "completeness": 0.0, "ari": 0.0, "ami": 0.0, "noise_ratio": float((y_pred == -1).mean())}


def process_window(scenario, w_idx, X_norm, y_true, ndim):
    result_path = RESULTS_DIR / f"{scenario}_umap{ndim}d_w{w_idx:04d}.json"
    if result_path.exists():
        return None
    umap = UMAP(n_components=ndim, metric="euclidean", n_neighbors=15, random_state=42 + w_idx, n_jobs=1)
    X_reduced = umap.fit_transform(X_norm)
    params = BEST_PARAMS[scenario]
    clusterer = HDBSCAN(**params)
    labels = clusterer.fit_predict(X_reduced)
    result = {"labels": labels.tolist(), "n_clusters": len(set(labels)) - (1 if -1 in labels else 0), "n_noise": int((labels == -1).sum()), "n_total": len(labels)}
    with open(result_path, "w") as f:
        json.dump(result, f)
    return result


def run_scenario(scenario_name):
    print(f"\n  Scenario: {scenario_name}")
    data_path = SCENARIOS_DIR / f"{scenario_name}.npz"
    if not data_path.exists():
        print(f"    [SKIP] File not found")
        return
    data = np.load(data_path, allow_pickle=True)
    X, y_true = data["X"], data["y"]
    data.close()
    n_windows = X.shape[0]

    # Per-scenario normalization (Run B approach)
    feat_mean = X.mean(axis=(0, 1))
    feat_std = X.std(axis=(0, 1))
    feat_std[feat_std == 0] = 1.0

    for ndim in [2, 3]:
        print(f"    UMAP {ndim}D: {n_windows} windows")
        n_completed = 0
        for w_idx in range(n_windows):
            result_path = RESULTS_DIR / f"{scenario_name}_umap{ndim}d_w{w_idx:04d}.json"
            if result_path.exists():
                n_completed += 1
                continue
            X_norm = (X[w_idx] - feat_mean) / feat_std
            result = process_window(scenario_name, w_idx, X_norm, y_true[w_idx], ndim)
            if result is not None:
                n_completed += 1
        print(f"      Completed: {n_completed}/{n_windows}")

    # ── Evaluate ──
    rows = []
    for ndim in [2, 3]:
        for w_idx in range(n_windows):
            result_path = RESULTS_DIR / f"{scenario_name}_umap{ndim}d_w{w_idx:04d}.json"
            if not result_path.exists():
                continue
            with open(result_path) as f:
                pred = json.load(f)
            metrics = evaluate(y_true[w_idx], pred["labels"])
            metrics["scenario"] = scenario_name
            metrics["method"] = f"UMAP_{ndim}D"
            metrics["window"] = w_idx
            rows.append(metrics)

    return rows


if __name__ == "__main__":
    print("=" * 60)
    print("Experiment 2A: UMAP + HDBSCAN")
    print("=" * 60)

    start = time.time()
    scenario_names = ["stare_low", "stare_high", "scan_low", "scan_high", "mixed"]
    all_rows = []

    for name in scenario_names:
        rows = run_scenario(name)
        if rows:
            all_rows.extend(rows)

    # ── Summary table ──
    df = pd.DataFrame(all_rows)
    summary = df.groupby(["scenario", "method"]).agg(v_measure=("v_measure", "mean"), ari=("ari", "mean"), noise_ratio=("noise_ratio", "mean"), n_windows=("window", "count")).reset_index()

    print(f"\n{'=' * 120}")
    print(f"{'Scenario':<12} {'Method':<14} {'V-measure':>10} {'ARI':>10} {'Noise%':>8} {'Windows':>8}")
    print(f"{'─' * 120}")
    for _, r in summary.iterrows():
        print(f"{r['scenario']:<12} {r['method']:<14} {r['v_measure']:10.4f} {r['ari']:10.4f} {r['noise_ratio']*100:7.1f}% {r['n_windows']:8.0f}")
    # Also add Run B baseline
    print(f"{'─' * 120}")
    for s in scenario_names:
        print(f"{s:<12} {'Run_B_5D':<14} {RUN_B_BASELINE[s]:10.4f} {'':>10} {'':>8} {'':>8}")

    # ── Delta table: change from Run B baseline ──
    print(f"\n{'=' * 80}")
    print("Delta from Run B baseline (V-measure)")
    print(f"{'=' * 80}")
    print(f"{'Scenario':<12} {'UMAP_2D':>10} {'UMAP_3D':>10}")
    for s in scenario_names:
        base = RUN_B_BASELINE[s]
        v2 = summary.loc[(summary["scenario"] == s) & (summary["method"] == "UMAP_2D"), "v_measure"].values
        v3 = summary.loc[(summary["scenario"] == s) & (summary["method"] == "UMAP_3D"), "v_measure"].values
        d2 = v2[0] - base if len(v2) else 0
        d3 = v3[0] - base if len(v3) else 0
        print(f"{s:<12} {d2:+10.4f} ({d2/base*100:+5.1f}%) {d3:+10.4f} ({d3/base*100:+5.1f}%)")

    # ── Save ──
    csv_path = RESULTS_DIR / "summary_umap_hdbscan.csv"
    summary.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\nSaved: {csv_path}")

    elapsed = time.time() - start
    mins, secs = divmod(elapsed, 60)
    print(f"\nTime: {int(mins)}m {int(secs)}s")
