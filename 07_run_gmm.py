"""
07_run_gmm.py — Experiment 2B: Gaussian Mixture Model baseline (fast version)

Optimizations:
  - n_init=1, max_iter=100 (BIC selects best K, not best init)
  - K range: 2-20 (covers 2-30 emitters per window)
  - joblib parallelism for windows
  - Checkpointing per window (resume-safe)
"""

import os, sys, json, time
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.mixture import GaussianMixture
from sklearn.metrics import v_measure_score, adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score, completeness_score

load_dotenv(dotenv_path=Path(__file__).parent / ".env")
BASE_DIR = Path(__file__).parent.resolve()
SCENARIOS_DIR = Path(os.getenv("TSRD_SCENARIOS_DIR", BASE_DIR / "scenarios"))
RESULTS_DIR = BASE_DIR / "results_experiment2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

SCENARIOS = ["stare_low", "stare_high", "scan_low", "scan_high", "mixed"]

RUN_B_BASELINE = {"stare_low": 0.4987, "stare_high": 0.9020, "scan_low": 0.6479, "scan_high": 0.8709, "mixed": 0.8097}

# K from 2 to 20 — covers typical 2-30 emitters in a window
K_RANGE = list(range(2, 21))


def fit_gmm_window(scenario_name, w_idx, X_norm, y_true_w):
    result_path = RESULTS_DIR / f"{scenario_name}_gmm_w{w_idx:04d}.json"
    if result_path.exists():
        with open(result_path) as f:
            return json.load(f)

    best_bic = float("inf")
    best_labels = None
    best_k = None

    for k in K_RANGE:
        gmm = GaussianMixture(n_components=k, random_state=42 + w_idx + k, n_init=1, max_iter=100)
        gmm.fit(X_norm)
        bic = gmm.bic(X_norm)
        labels = gmm.predict(X_norm)
        if bic < best_bic:
            best_bic = bic
            best_labels = labels
            best_k = k

    # Evaluate
    y_pred = np.array(best_labels)
    u = np.unique(y_pred)
    if len(u) <= 1:
        metrics = {"v_measure": 0.0, "homogeneity": 1.0 if len(u) == 1 else 0.0, "completeness": 0.0, "ari": 0.0, "ami": 0.0, "noise_ratio": 0.0}
    else:
        try:
            metrics = {"v_measure": v_measure_score(y_true_w, y_pred), "homogeneity": homogeneity_score(y_true_w, y_pred), "completeness": completeness_score(y_true_w, y_pred), "ari": adjusted_rand_score(y_true_w, y_pred), "ami": adjusted_mutual_info_score(y_true_w, y_pred), "noise_ratio": 0.0}
        except:
            metrics = {"v_measure": 0.0, "homogeneity": 0.0, "completeness": 0.0, "ari": 0.0, "ami": 0.0, "noise_ratio": 0.0}

    entry = {"scenario": scenario_name, "method": "GMM", "window": w_idx, "best_k": best_k, "best_bic": float(best_bic), **metrics}
    with open(result_path, "w") as f:
        json.dump(entry, f)
    return entry


def run_scenario(scenario_name):
    print(f"\n  Scenario: {scenario_name}")
    data_path = SCENARIOS_DIR / f"{scenario_name}.npz"
    if not data_path.exists():
        print(f"    [SKIP] File not found")
        return []
    data = np.load(data_path, allow_pickle=True)
    X, y_true = data["X"], data["y"]
    data.close()
    n_windows = X.shape[0]

    # Per-scenario normalization
    feat_mean = X.mean(axis=(0, 1))
    feat_std = X.std(axis=(0, 1))
    feat_std[feat_std == 0] = 1.0

    # Pre-normalize all windows
    X_norm_all = (X - feat_mean) / feat_std

    # Check which windows need processing
    pending = []
    for w_idx in range(n_windows):
        rp = RESULTS_DIR / f"{scenario_name}_gmm_w{w_idx:04d}.json"
        if not rp.exists():
            pending.append(w_idx)

    # Process pending windows in parallel
    if pending:
        print(f"    Processing {len(pending)}/{n_windows} windows (n_jobs=4)...")
        results = Parallel(n_jobs=4)(
            delayed(fit_gmm_window)(scenario_name, w_idx, X_norm_all[w_idx], y_true[w_idx])
            for w_idx in pending
        )
    else:
        print(f"    All {n_windows} windows already cached")

    # Load all results
    rows = []
    for w_idx in range(n_windows):
        rp = RESULTS_DIR / f"{scenario_name}_gmm_w{w_idx:04d}.json"
        if rp.exists():
            with open(rp) as f:
                rows.append(json.load(f))
    return rows


if __name__ == "__main__":
    print("=" * 60)
    print("Experiment 2B: GMM baseline (optimized)")
    print("=" * 60)

    start = time.time()
    all_rows = []

    for name in SCENARIOS:
        rows = run_scenario(name)
        if rows:
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    summary = df.groupby("scenario").agg(v_measure=("v_measure", "mean"), ari=("ari", "mean"), best_k_mean=("best_k", "mean"), n_windows=("window", "count")).reset_index()

    print(f"\n{'=' * 110}")
    print(f"{'Scenario':<12} {'Method':<10} {'V-measure':>10} {'ARI':>10} {'Avg K':>8} {'Windows':>8}")
    print("-" * 110)
    for _, r in summary.iterrows():
        print(f"{r['scenario']:<12} {'GMM':<10} {r['v_measure']:10.4f} {r['ari']:10.4f} {r['best_k_mean']:7.1f}  {r['n_windows']:8.0f}")
    print("-" * 110)
    for s in SCENARIOS:
        print(f"{s:<12} {'Run_B':<10} {RUN_B_BASELINE[s]:10.4f} {'':>10} {'':>8} {'':>8}")
    print("=" * 110)

    print(f"\n{'=' * 80}")
    print("Delta: GMM vs Run B baseline (V-measure)")
    print("=" * 80)
    print(f"{'Scenario':<12} {'GMM':>10} {'Run_B':>10} {'Delta':>10}")
    for s in SCENARIOS:
        v = summary.loc[summary["scenario"] == s, "v_measure"].values[0]
        base = RUN_B_BASELINE[s]
        d = v - base
        p = d / base * 100 if base else 0
        print(f"{s:<12} {v:10.4f} {base:10.4f} {d:+9.4f} ({p:+5.1f}%)")

    csv_path = RESULTS_DIR / "summary_gmm.csv"
    summary.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\nSaved: {csv_path}")

    elapsed = time.time() - start
    mins, secs = divmod(elapsed, 60)
    print(f"\nTime: {int(mins)}m {int(secs)}s")
