"""
06_evaluate.py — Evaluate clustering results against ground truth

What this script does:
1. Reads all cached clustering results from step 05
2. For each (scenario, param combination):
   a. Loads true emitter labels from .npz file
   b. Loads predicted labels from .json cache
   c. Computes 7 evaluation metrics
   d. Averages across all windows in that scenario
3. Saves a summary CSV: results/summary_metrics.csv
4. Identifies the best parameter combo per scenario (by V-measure)

Metrics glossary:
  - V-measure:        Harmonic mean of homogeneity & completeness (primary metric, 0-1)
  - Homogeneity:      Each cluster contains only one emitter class (0-1)
  - Completeness:     All pulses of an emitter go to the same cluster (0-1)
  - ARI:              Adjusted Rand Index, pairwise agreement corrected for chance (0-1)
  - AMI:              Adjusted Mutual Information, information overlap (0-1)
  - N_clusters:       How many clusters HDBSCAN found
  - Noise_ratio:      Fraction of pulses labeled as noise (-1) by HDBSCAN

Run: python 06_evaluate.py
"""

import os
import sys
import json
import time
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import (
    v_measure_score,
    adjusted_rand_score,
    adjusted_mutual_info_score,
    homogeneity_score,
    completeness_score,
)

# Load .env
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

BASE_DIR = Path(__file__).parent.resolve()
SCENARIOS_DIR = Path(os.getenv("TSRD_SCENARIOS_DIR", BASE_DIR / "scenarios"))
RESULTS_DIR = Path(os.getenv("TSRD_RESULTS_DIR", BASE_DIR / "results"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# STEP 1: Parameter definitions (same as step 05)
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


def param_to_hash(params):
    import hashlib, json
    key = json.dumps(params, sort_keys=True)
    return hashlib.md5(key.encode()).hexdigest()[:8]


def param_to_label(params):
    ms = params["min_samples"] if params["min_samples"] is not None else "auto"
    return f"cs{params['min_cluster_size']}_ms{ms}_eps{params['cluster_selection_epsilon']}"


# ---------------------------------------------------------------------------
# STEP 2: Evaluate one window's predictions
# ---------------------------------------------------------------------------

def evaluate_window(y_true, y_pred_dict):
    """
    Compute all metrics for one window.
    
    y_true: numpy array of true emitter labels (1024,)
    y_pred_dict: dict from cache with 'labels' list
    
    Returns dict of metric values.
    """
    y_pred = np.array(y_pred_dict["labels"])

    # If all predictions are noise or all same label, some metrics fail
    unique_pred = np.unique(y_pred)
    unique_true = np.unique(y_true)

    # Handle edge cases
    if len(unique_pred) <= 1:
        # Only noise or only one cluster — homogeneity = 1, completeness = 0
        return {
            "v_measure": 0.0,
            "homogeneity": 1.0 if len(unique_pred) == 1 else 0.0,
            "completeness": 0.0,
            "ari": 0.0,
            "ami": 0.0,
            "n_clusters_true": len(unique_true),
            "n_clusters_pred": len([u for u in unique_pred if u != -1]),
            "noise_ratio": float((y_pred == -1).mean()),
        }

    try:
        metrics = {
            "v_measure": v_measure_score(y_true, y_pred),
            "homogeneity": homogeneity_score(y_true, y_pred),
            "completeness": completeness_score(y_true, y_pred),
            "ari": adjusted_rand_score(y_true, y_pred),
            "ami": adjusted_mutual_info_score(y_true, y_pred),
            "n_clusters_true": len(unique_true),
            "n_clusters_pred": len([u for u in unique_pred if u != -1]),
            "noise_ratio": float((y_pred == -1).mean()),
        }
    except Exception as e:
        # Some edge cases can cause sklearn errors
        metrics = {
            "v_measure": 0.0,
            "homogeneity": 0.0,
            "completeness": 0.0,
            "ari": 0.0,
            "ami": 0.0,
            "n_clusters_true": len(unique_true),
            "n_clusters_pred": len([u for u in unique_pred if u != -1]),
            "noise_ratio": float((y_pred == -1).mean()),
        }

    return metrics


# ---------------------------------------------------------------------------
# STEP 3: Evaluate a full scenario
# ---------------------------------------------------------------------------

def evaluate_scenario(scenario_name, p_idx, params):
    """
    For one scenario + one param set:
    - Load all cached window results
    - Compute metrics per window
    - Return as DataFrame rows
    """
    import hashlib, json
    param_hash = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]
    param_label = param_to_label(params)

    # Load ground truth
    data_path = SCENARIOS_DIR / f"{scenario_name}.npz"
    if not data_path.exists():
        return None
    data = np.load(data_path, allow_pickle=True)
    y_all = data["y"]
    n_windows = y_all.shape[0]
    data.close()

    rows = []
    for w_idx in range(n_windows):
        result_path = RESULTS_DIR / f"{scenario_name}_w{w_idx:04d}_p{param_hash}.json"
        if not result_path.exists():
            continue

        with open(result_path) as f:
            pred_dict = json.load(f)

        metrics = evaluate_window(y_all[w_idx], pred_dict)
        metrics["scenario"] = scenario_name
        metrics["param_label"] = param_label
        metrics["param_hash"] = param_hash
        metrics["window_idx"] = w_idx
        rows.append(metrics)

    return rows


# ---------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Evaluating clustering results")
    print("=" * 60)

    scenario_names = ["stare_low", "stare_high", "scan_low", "scan_high", "mixed"]

    all_rows = []
    total = len(scenario_names) * len(PARAM_GRID)

    start = time.time()

    with tqdm(total=total, desc="Evaluating", unit="combo") as pbar:
        for name in scenario_names:
            for p_idx, params in enumerate(PARAM_GRID):
                rows = evaluate_scenario(name, p_idx, params)
                if rows:
                    all_rows.extend(rows)
                pbar.update(1)

    if len(all_rows) == 0:
        print("[ERROR] No results found. Run 05_run_hdbscan.py first.")
        sys.exit(1)

    # Convert to DataFrame
    df = pd.DataFrame(all_rows)

    # Compute mean ± std grouped by scenario + param
    agg = df.groupby(["scenario", "param_label"]).agg(
        v_measure_mean=("v_measure", "mean"),
        v_measure_std=("v_measure", "std"),
        homogeneity_mean=("homogeneity", "mean"),
        completeness_mean=("completeness", "mean"),
        ari_mean=("ari", "mean"),
        ami_mean=("ami", "mean"),
        n_clusters_pred_mean=("n_clusters_pred", "mean"),
        noise_ratio_mean=("noise_ratio", "mean"),
        n_windows=("window_idx", "count"),
    ).reset_index()

    # Sort by scenario then v-measure (descending)
    agg["v_measure_rank"] = agg.groupby("scenario")["v_measure_mean"].rank(ascending=False)
    agg = agg.sort_values(["scenario", "v_measure_rank"])

    # Save
    csv_path = RESULTS_DIR / "summary_metrics.csv"
    agg.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\n  Saved summary: {csv_path}")

    # Find best params per scenario
    best = agg.loc[agg.groupby("scenario")["v_measure_rank"].idxmin()]
    best_params = {}
    for _, row in best.iterrows():
        best_params[row["scenario"]] = {
            "param_label": row["param_label"],
            "v_measure": round(row["v_measure_mean"], 4),
            "ari": round(row["ari_mean"], 4),
            "noise_ratio": round(row["noise_ratio_mean"], 4),
        }

    json_path = RESULTS_DIR / "best_params.json"
    with open(json_path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"  Saved best params: {json_path}")

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"BEST PARAMETERS PER SCENARIO")
    print(f"{'=' * 60}")
    for scenario, bp in best_params.items():
        print(f"  {scenario:15s}: {bp['param_label']:20s}  "
              f"V={bp['v_measure']:.3f}  ARI={bp['ari']:.3f}  Noise={bp['noise_ratio']:.1%}")

    elapsed = time.time() - start
    print(f"\n  Evaluated {len(all_rows)} window-param combinations in {elapsed:.1f}s")
    print(f"  Next: python 07_visualize.py")
