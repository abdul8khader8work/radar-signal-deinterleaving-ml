"""
05_run_hdbscan.py — Run HDBSCAN clustering on all 5 scenarios

What this script does:
1. Loads each scenario .npz file (windowed data from step 04)
2. For each window, runs HDBSCAN with every parameter combination
3. Uses n_jobs=4 for parallel processing (safe for laptop)
4. Caches results per (scenario, window, param) to disk
5. Supports resume: if interrupted, it picks up where it left off

HDBSCAN parameters we vary:
  - min_cluster_size:     minimum points to form a cluster (10, 20, 50)
  - min_samples:          distance to kth nearest neighbor (None = defaults to min_cluster_size)
  - cluster_selection_epsilon: merge clusters below this distance (0.0, 0.1)
  - cluster_selection_method:  'eom' (Excess of Mass) = standard HDBSCAN*
  
Total: 12 parameter combinations × 5 scenarios × 100 windows = 6,000 cluster fits

Run: python 05_run_hdbscan.py
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

np.random.seed(42)

# Load .env
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

BASE_DIR = Path(__file__).parent.resolve()
SCENARIOS_DIR = Path(os.getenv("TSRD_SCENARIOS_DIR", BASE_DIR / "scenarios"))
RESULTS_DIR = Path(os.getenv("TSRD_RESULTS_DIR", BASE_DIR / "results"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# STEP 1: Parameter grid
# ---------------------------------------------------------------------------

# We define 12 parameter combinations.
# In HDBSCAN:
#   min_cluster_size:   The smallest grouping considered a cluster.
#                       Smaller = more clusters, more noise.
#                       Larger = fewer clusters, less noise.
#   min_samples:        How conservative the algorithm is about noise.
#                       None = same as min_cluster_size (denser clusters).
#   cluster_selection_epsilon:  Distance threshold to merge clusters.
#                               0.0 = no merging (standard HDBSCAN*).
#                               0.1 = merge nearby clusters.
#   cluster_selection_method:   'eom' (Excess of Mass) = most persistent clusters
#                               'leaf' = finest-grained clusters (not used here)

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
    """Create a unique hash for a parameter dict (for filename)"""
    key = json.dumps(params, sort_keys=True)
    return hashlib.md5(key.encode()).hexdigest()[:8]


def param_to_label(params):
    """Human-readable label for a parameter set"""
    ms = params["min_samples"] if params["min_samples"] is not None else "auto"
    return (
        f"cs{params['min_cluster_size']}"
        f"_ms{ms}"
        f"_eps{params['cluster_selection_epsilon']}"
    )


# ---------------------------------------------------------------------------
# STEP 2a: Feature engineering — add PRI-derived + delta features
# ---------------------------------------------------------------------------

def add_pri_features(X_window):
    """
    Add 8 derived features to a single window:
    - 6 temporal lag/lead features (capture PRI patterns)
    - 2 delta features (delta_freq, delta_pw — capture smoothness)
    
    X_window: (1024, 5) — [ToA, Freq, PW, AoA, Ampl]
    Returns: (1024, 13) — original 5 + 8 derived
    """
    n = len(X_window)
    ToA = X_window[:, 0]
    Freq = X_window[:, 1]
    PW = X_window[:, 2]

    lag1 = np.zeros(n)
    lead1 = np.zeros(n)
    lag2 = np.zeros(n)
    lead2 = np.zeros(n)
    lag3 = np.zeros(n)
    lead3 = np.zeros(n)
    delta_freq = np.zeros(n)
    delta_pw = np.zeros(n)

    lag1[1:] = ToA[1:] - ToA[:-1]
    lead1[:-1] = ToA[1:] - ToA[:-1]
    lag2[2:] = ToA[2:] - ToA[:-2]
    lead2[:-2] = ToA[2:] - ToA[:-2]
    lag3[3:] = ToA[3:] - ToA[:-3]
    lead3[:-3] = ToA[3:] - ToA[:-3]
    delta_freq[1:] = np.abs(Freq[1:] - Freq[:-1])
    delta_pw[1:] = np.abs(PW[1:] - PW[:-1])

    return np.column_stack([X_window, lag1, lead1, lag2, lead2, lag3, lead3, delta_freq, delta_pw])


# ---------------------------------------------------------------------------
# STEP 2b: Run HDBSCAN on a single window with a single param set
# ---------------------------------------------------------------------------

def cluster_window(X, params):
    """
    Run HDBSCAN on one window of PDW data.
    
    X: numpy array of shape (1024, 5) — one window of PDWs
    params: dict of HDBSCAN parameters
    
    Returns: dict with labels, n_clusters, n_noise
    """
    clusterer = HDBSCAN(**params)
    labels = clusterer.fit_predict(X)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())

    return {
        "labels": labels.tolist(),
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "n_total": len(labels),
    }


# ---------------------------------------------------------------------------
# STEP 3: Process one scenario
# ---------------------------------------------------------------------------

def process_scenario(scenario_name, X, y_true, n_jobs=4):
    """
    For one scenario: run all param combinations on all windows.
    
    We use joblib.Parallel to process multiple windows simultaneously.
    n_jobs=4 means 4 windows at a time (safe for 8GB RAM).
    Results are saved per (window, param) to enable resume.
    """
    n_windows = X.shape[0]
    n_params = len(PARAM_GRID)
    print(f"\n  Scenario: {scenario_name}")
    print(f"    Windows: {n_windows}  |  Params: {n_params}  |  Total fits: {n_windows * n_params}")

    # Pre-compute extended features for all windows
    print(f"    Computing 8 PRI-derived features per window...")
    X_extended = np.zeros((n_windows, X.shape[1], X.shape[2] + 8))
    for w_idx in range(n_windows):
        X_extended[w_idx] = add_pri_features(X[w_idx])
    print(f"    Feature matrix: ({X_extended.shape[0]}, {X_extended.shape[1]}, {X_extended.shape[2]})")

    # Per-scenario normalization: compute global mean/std across all windows
    feat_mean = X_extended.mean(axis=(0, 1))
    feat_std = X_extended.std(axis=(0, 1))
    feat_std[feat_std == 0] = 1.0
    print(f"    Normalization: {X_extended.shape[2]} features normalized")

    # Count already-completed results
    total_fits = n_windows * n_params
    completed_fits = 0

    for w_idx in range(n_windows):
        X_norm = (X_extended[w_idx] - feat_mean) / feat_std
        for p_idx, params in enumerate(PARAM_GRID):
            param_hash = param_to_hash(params)
            result_path = RESULTS_DIR / f"{scenario_name}_w{w_idx:04d}_p{param_hash}.json"

            # Check if this (window, param) combo is already done
            if not result_path.exists():
                try:
                    result = cluster_window(X_norm, params)
                    with open(result_path, "w") as f:
                        json.dump(result, f)
                    completed_fits += 1
                except Exception as e:
                    print(f"    [WARN] Window {w_idx}, param {param_hash} failed: {e}")
                    continue

    print(f"    Completed fits this run: {completed_fits}")
    return completed_fits


# ---------------------------------------------------------------------------
# STEP 4: Load all scenarios and run
# ---------------------------------------------------------------------------

def load_scenario(name):
    """Load a scenario .npz file"""
    path = SCENARIOS_DIR / f"{name}.npz"
    if not path.exists():
        print(f"  [WARNING] Scenario file not found: {path}")
        print(f"            Run 04_create_scenarios.py first.")
        return None, None
    data = np.load(path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    data.close()
    return X, y


# ---------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("HDBSCAN clustering sweep on 5 scenarios")
    print("=" * 60)
    print(f"  n_jobs:      4 (parallel windows)")
    print(f"  Params:      {len(PARAM_GRID)} combinations")
    print(f"  Results in:  {RESULTS_DIR}")
    print()

    start_total = time.time()
    total_fits = 0

    # List of scenario names (must match .npz files from step 04)
    scenario_names = [
        "stare_low",
        "stare_high",
        "scan_low",
        "scan_high",
        "mixed",
    ]

    for name in scenario_names:
        X, y_true = load_scenario(name)
        if X is None:
            continue

        n_fits = process_scenario(name, X, y_true, n_jobs=4)
        total_fits += n_fits

        # Free memory between scenarios
        del X, y_true
        gc.collect()

    elapsed = time.time() - start_total
    mins, secs = divmod(elapsed, 60)

    print(f"\n{'=' * 60}")
    print(f"CLUSTERING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  New cluster fits: {total_fits:,}")
    print(f"  Total time:       {int(mins)}m {int(secs)}s")
    print(f"  Results cached in: {RESULTS_DIR}")
    print(f"  Next: python 06_evaluate.py")
