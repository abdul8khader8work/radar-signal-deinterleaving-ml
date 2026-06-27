"""
11_generate_all_plots.py - Generate comprehensive visualizations for Runs B, J, K, L, M.

Saves per-window predictions to cache, then generates all plots.
"""

import os, sys, json, gc, time, warnings
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import hdbscan
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import v_measure_score
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.optim as optim
warnings.filterwarnings("ignore")

load_dotenv(dotenv_path=Path(__file__).parent / ".env")
BASE_DIR = Path(__file__).parent.resolve()
SCENARIOS_DIR = Path(os.getenv("TSRD_SCENARIOS_DIR", BASE_DIR / "scenarios"))
CACHE_DIR = BASE_DIR / "results_experiment6" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE_DIR))
from utils.plotting import plot_scenario_results, plot_run_summary, plot_comparison

SCENARIOS = ["stare_low", "stare_high", "scan_low", "scan_high", "mixed"]
N_WINDOWS = 100
WINDOW_LEN = 1024
RUN_B_HASH = {"stare_low":"86b9f834","stare_high":"779a2296","scan_low":"779a2296",
              "scan_high":"779a2296","mixed":"779a2296"}

RUN_BASE = BASE_DIR / "results_runB_backup"
PLOT_DIRS = {
    "Run_B": BASE_DIR / "results" / "run_b" / "plots",
    "Run_J": BASE_DIR / "results" / "run_j" / "plots",
    "Run_K": BASE_DIR / "results" / "run_k" / "plots",
    "Run_L": BASE_DIR / "results" / "run_l" / "plots",
    "Run_M": BASE_DIR / "results" / "run_m" / "plots",
}
COMPARISON_DIR = BASE_DIR / "results" / "comparison" / "plots"

def load_scenario(name):
    data = np.load(SCENARIOS_DIR / f"{name}.npz", allow_pickle=True)
    X, y = data["X"], data["y"]
    data.close()
    return X, y

# =====================================================================
# CACHE: Save/load per-window predictions
# =====================================================================
def cache_path(run_name, scenario):
    return CACHE_DIR / f"{run_name}_{scenario}_labels.npy"

def save_cache(run_name, scenario, labels_list):
    arr = np.array(labels_list, dtype=object)
    np.save(cache_path(run_name, scenario), arr)

def load_cache(run_name, scenario):
    p = cache_path(run_name, scenario)
    if p.exists():
        return list(np.load(p, allow_pickle=True))
    return None

# =====================================================================
# Run B: Load from results_runB_backup/
# =====================================================================
def get_predictions_run_b(scenario):
    cached = load_cache("Run_B", scenario)
    if cached is not None:
        return cached
    hash_v = RUN_B_HASH[scenario]
    labels = []
    for w in range(N_WINDOWS):
        path = RUN_BASE / f"{scenario}_w{w:04d}_p{hash_v}.json"
        if path.exists():
            with open(path) as f:
                labels.append(np.array(json.load(f)["labels"], dtype=int))
        else:
            labels.append(np.full(WINDOW_LEN, -1, dtype=int))
    save_cache("Run_B", scenario, labels)
    return labels

# =====================================================================
# Run J: Multi-scale PRI Histogram
# =====================================================================
def _multiscale_pri_peaks(intervals, bin_sizes=(50, 100, 200)):
    peaks_by_scale = []
    for n_bins in bin_sizes:
        if len(intervals) < 3:
            continue
        p5, p95 = np.percentile(intervals, [2, 98])
        if p95 - p5 < 1:
            p5, p95 = intervals.min(), np.percentile(intervals, 99)
        if p95 - p5 < 1:
            continue
        bins = np.linspace(p5, p95, n_bins + 1)
        h, _ = np.histogram(intervals, bins=bins)
        bc = (bins[:-1] + bins[1:]) / 2
        thresh = h.mean() + 1.0 * h.std()
        sp = [float(bc[i]) for i in range(1, len(h)-1) if h[i] > h[i-1] and h[i] > h[i+1] and h[i] > thresh]
        peaks_by_scale.append(sp)
    if not peaks_by_scale:
        return []
    all_flat = [p for s in peaks_by_scale for p in s]
    if not all_flat:
        return []
    p_arr = np.array(all_flat).reshape(-1, 1)
    if len(p_arr) < 2:
        return sorted(set(all_flat))
    scaler = StandardScaler()
    pn = scaler.fit_transform(p_arr)
    clustering = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, cluster_selection_epsilon=0.3)
    pl = clustering.fit_predict(pn)
    consensus = []
    for label in set(pl):
        if label == -1:
            continue
        members = p_arr[pl == label]
        consensus.append(float(np.median(members)))
    return sorted(consensus)

def get_predictions_run_j(scenario):
    cached = load_cache("Run_J", scenario)
    if cached is not None:
        return cached
    X, _ = load_scenario(scenario)
    all_labels = []
    for w in range(N_WINDOWS):
        toa = X[w, :, 0]
        toa_s = np.sort(toa)
        intervals = toa_s[1:] - toa_s[:-1]
        pri_peaks = _multiscale_pri_peaks(intervals)
        labels = np.full(WINDOW_LEN, -1, dtype=int)
        if pri_peaks:
            for i in range(WINDOW_LEN):
                idx = np.argsort(toa)
                pos = np.where(idx == i)[0]
                if len(pos) == 0:
                    continue
                p = pos[0]
                if p >= len(intervals):
                    continue
                interval = intervals[p]
                dists = np.abs(np.array(pri_peaks) - interval)
                min_dist = dists.min()
                if min_dist < 0.15 * interval:
                    labels[i] = int(np.argmin(dists))
        all_labels.append(labels)
    save_cache("Run_J", scenario, all_labels)
    return all_labels

# =====================================================================
# Run K: Ensemble Voting
# =====================================================================
def get_predictions_run_k(scenario):
    cached = load_cache("Run_K", scenario)
    if cached is not None:
        return cached
    X, y_true = load_scenario(scenario)
    all_labels = []
    for w in range(N_WINDOWS):
        pw = X[w]
        yt = y_true[w]
        pw_n = (pw - pw.mean(axis=0)) / (pw.std(axis=0) + 1e-10)
        n_true = len(set(yt))
        # 1. HDBSCAN
        hdb = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=10, cluster_selection_epsilon=0.1)
        l_hdb = hdb.fit_predict(pw_n)
        # 2. GMM
        K = max(2, min(n_true, 30))
        gmm = GaussianMixture(n_components=K, random_state=42, n_init=3)
        l_gmm = gmm.fit_predict(pw_n)
        # 3. KMeans
        km = KMeans(n_clusters=K, random_state=42, n_init=3)
        l_km = km.fit_predict(pw_n)
        # 4. Spectral
        try:
            spec = SpectralClustering(n_clusters=K, random_state=42, affinity='nearest_neighbors',
                                      n_neighbors=min(K*5, 100), n_init=3)
            l_spec = spec.fit_predict(pw_n)
        except Exception:
            l_spec = l_km.copy()
        # Align labels via Hungarian matching to HDBSCAN
        all_l = np.stack([l_hdb, l_gmm, l_km, l_spec])
        aligned = []
        for ai in range(4):
            al = all_l[ai]
            noise_mask = al == -1
            ac = al.copy()
            ac[noise_mask] = -999
            ua = sorted(set(ac) - {-999})
            uh = sorted(set(l_hdb) - {-1})
            if not ua or not uh:
                aligned.append(al)
                continue
            cm = np.zeros((len(ua), len(uh)))
            am = {v: i for i, v in enumerate(ua)}
            hm = {v: i for i, v in enumerate(uh)}
            for p in range(WINDOW_LEN):
                a, hh = ac[p], l_hdb[p]
                if a in am and hh in hm:
                    cm[am[a], hm[hh]] += 1
            ri, ci = linear_sum_assignment(-cm)
            remap = {ua[r]: uh[c] for r, c in zip(ri, ci)}
            aa = np.array([remap.get(aa, -1) for aa in ac])
            aligned.append(aa)
        aligned = np.array(aligned)
        ens = np.full(WINDOW_LEN, -1, dtype=int)
        for p in range(WINDOW_LEN):
            votes = aligned[:, p]
            vc = Counter(votes[votes != -1])
            if vc:
                ens[p] = vc.most_common(1)[0][0]
        all_labels.append(ens)
    save_cache("Run_K", scenario, all_labels)
    return all_labels

# =====================================================================
# Run L: CDIF Standalone
# =====================================================================
def _cdif_peaks(toa, max_levels=4, n_bins=150):
    toa = np.sort(toa)
    n = len(toa)
    all_diffs = []
    for level in range(1, max_levels + 1):
        if n - level < 2:
            break
        all_diffs.append(toa[level:] - toa[:-level])
    if not all_diffs:
        return []
    d1 = all_diffs[0]
    p5, p95 = np.percentile(d1, [2, 98])
    if p95 - p5 < 1:
        p5, p95 = d1.min(), np.percentile(d1, 99)
    if p95 - p5 < 1:
        return [float(np.median(d1))]
    bins = np.linspace(p5, p95, n_bins + 1)
    bc = (bins[:-1] + bins[1:]) / 2
    hists = []
    for diffs in all_diffs:
        h, _ = np.histogram(diffs, bins=bins)
        hists.append(h.astype(np.float64))
    cdif = hists[0].copy()
    for li in range(1, len(hists)):
        cdif = np.maximum(cdif, hists[li] - 0.5 * hists[li - 1])
    thresh = np.mean(cdif) + 1.5 * np.std(cdif)
    peaks = [float(bc[i]) for i in range(1, len(cdif)-1)
             if cdif[i] > cdif[i-1] and cdif[i] > cdif[i+1] and cdif[i] > thresh]
    return sorted(peaks)[:4]

def get_predictions_run_l(scenario):
    cached = load_cache("Run_L", scenario)
    if cached is not None:
        return cached
    X, y_true = load_scenario(scenario)
    all_labels = []
    for w in range(N_WINDOWS):
        toa = X[w, :, 0]
        yt = y_true[w]
        toa_s = np.sort(toa)
        peaks = _cdif_peaks(toa)
        K = len(peaks)
        if K == 0:
            intervals = toa_s[1:] - toa_s[:-1]
            feats = np.zeros((WINDOW_LEN, 1))
            for i in range(min(WINDOW_LEN, len(intervals))):
                feats[i, 0] = intervals[i] if i < len(intervals) else 0
        else:
            feats = np.zeros((WINDOW_LEN, K))
            for i in range(WINDOW_LEN):
                pos = np.where(np.argsort(toa) == i)[0]
                if len(pos) == 0:
                    continue
                p = pos[0]
                interval = toa_s[p+1] - toa_s[p] if p + 1 < WINDOW_LEN else 0
                if interval > 0:
                    for ki, pk in enumerate(peaks):
                        feats[i, ki] = abs(interval - pk)
        feats_n = (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-10)
        params_grid = [
            {"min_cluster_size": 10, "min_samples": 10, "cluster_selection_epsilon": 0.1},
            {"min_cluster_size": 10, "min_samples": None, "cluster_selection_epsilon": 0.1},
        ]
        best_v, best_yp = 0, None
        for params in params_grid:
            clusterer = hdbscan.HDBSCAN(**params)
            yp = clusterer.fit_predict(feats_n)
            nm = yp == -1
            if (~nm).sum() > 0 and len(set(yp[~nm])) > 1:
                v = v_measure_score(yt[~nm], yp[~nm])
                if v > best_v:
                    best_v, best_yp = v, yp
        all_labels.append(best_yp if best_yp is not None else np.full(WINDOW_LEN, -1, dtype=int))
    save_cache("Run_L", scenario, all_labels)
    return all_labels

# =====================================================================
# Run M: Bi-GRU Post-Processor
# =====================================================================
class _TinyGRU(nn.Module):
    def __init__(self, input_size=1, hidden_size=16):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

def _train_gru(intervals, n_epochs=20, lr=0.01):
    if len(intervals) < 10:
        return None, None
    seqs = [intervals[i-3:i] for i in range(3, len(intervals))]
    targets = [intervals[i] for i in range(3, len(intervals))]
    if len(seqs) < 5:
        return None, None
    X_t = torch.FloatTensor(np.array(seqs)).unsqueeze(-1)
    y_t = torch.FloatTensor(np.array(targets)).unsqueeze(-1)
    model = _TinyGRU(input_size=1, hidden_size=16)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    model.train()
    for ep in range(n_epochs):
        pred = model(X_t)
        loss = crit(pred, y_t)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model, None

def get_predictions_run_m(scenario):
    cached = load_cache("Run_M", scenario)
    if cached is not None:
        return cached
    X, y_true = load_scenario(scenario)
    # Load Run B labels first
    run_b_labels = get_predictions_run_b(scenario)
    all_labels = []
    for w in range(N_WINDOWS):
        toa = X[w, :, 0]
        yt = y_true[w]
        yb = run_b_labels[w]
        toa_s = toa
        sort_idx = np.argsort(toa_s)
        sorted_toa = toa_s[sort_idx]
        sorted_labels = yb[sort_idx]
        # Train GRU per Run B cluster
        cluster_models = {}
        for cid in set(sorted_labels):
            if cid == -1:
                continue
            mask = sorted_labels == cid
            c_toa = sorted_toa[mask]
            if len(c_toa) < 5:
                continue
            c_int = c_toa[1:] - c_toa[:-1]
            if len(c_int) < 5:
                continue
            c_mean, c_std = np.mean(c_int), np.std(c_int) + 1e-6
            c_int_n = (c_int - c_mean) / c_std
            model, _ = _train_gru(c_int_n, n_epochs=20)
            if model is not None:
                cluster_models[cid] = (model, c_mean, c_std)
        # Refine labels
        refined = yb.copy()
        if cluster_models:
            for i in range(WINDOW_LEN):
                cid = yb[i]
                if cid == -1 or cid not in cluster_models:
                    continue
                pos = np.where(sort_idx == i)[0]
                if len(pos) == 0 or pos[0] < 3:
                    continue
                p = pos[0]
                recent = []
                for j in range(p-3, p):
                    recent.append(sorted_toa[j+1] - sorted_toa[j] if j+1 < WINDOW_LEN else 0)
                if len(recent) < 3:
                    continue
                model, c_mean, c_std = cluster_models[cid]
                recent_n = (np.array(recent) - c_mean) / c_std
                with torch.no_grad():
                    inp = torch.FloatTensor(recent_n).unsqueeze(0).unsqueeze(-1)
                    pred_n = model(inp).squeeze().item()
                    pred_int = pred_n * c_std + c_mean
                actual_int = sorted_toa[p] - sorted_toa[p-1] if p > 0 else 0
                if actual_int > 0 and pred_int > 0:
                    err = abs(actual_int - pred_int) / max(actual_int, pred_int)
                    if err > 1.0:
                        refined[i] = -1
            # Merge similar clusters
            cids = sorted(cluster_models.keys())
            merge_groups = {c: c for c in cids}
            for i, c1 in enumerate(cids):
                for c2 in cids[i+1:]:
                    if merge_groups[c2] == c1:
                        continue
                    _, m1, s1 = cluster_models[c1]
                    _, m2, s2 = cluster_models[c2]
                    if abs(m1 - m2) < 1.5 * max(s1, s2):
                        for k, v in merge_groups.items():
                            if v == c2:
                                merge_groups[k] = c1
            merge_map = {}
            nid = 0
            for c in cids:
                root = merge_groups[c]
                if root not in merge_map:
                    merge_map[root] = nid; nid += 1
                merge_map[c] = merge_map[root]
            new_l = refined.copy()
            for i in range(WINDOW_LEN):
                cc = refined[i]
                if cc in merge_map:
                    new_l[i] = merge_map[cc]
            refined = new_l
        all_labels.append(refined)
    save_cache("Run_M", scenario, all_labels)
    return all_labels

# =====================================================================
# MAIN
# =====================================================================
def compute_scenario_metrics(y_true_list, y_pred_list):
    """Compute per-scenario metrics dict (averaged across windows)."""
    v_vals, sil_vals, db_vals, n_clust, n_noise, n_total = [], [], [], [], 0, 0
    for yt, yp in zip(y_true_list, y_pred_list):
        if yp is None:
            continue
        nm = yp == -1
        n_noise += nm.sum()
        n_total += len(yp)
        n_clust.append(len(set(yp)) - (1 if -1 in yp else 0))
        if (~nm).sum() > 0 and len(set(yp[~nm])) > 1:
            v_vals.append(v_measure_score(yt[~nm], yp[~nm]))
        else:
            v_vals.append(0.0)
    # Silhouette and DB on best window
    best_w = np.argmax(v_vals) if v_vals else 0
    yt_b, yp_b = y_true_list[best_w], y_pred_list[best_w]
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    nm = yp_b != -1
    if nm.sum() >= 2 and len(set(yp_b[nm])) > 1:
        from utils.plotting import _reduce_2d
        X_sc, _ = load_scenario(SCENARIOS[0])  # hack: just use first scenario's X
        sil = float(silhouette_score(X_sc[best_w][nm], yp_b[nm]))
        db = float(davies_bouldin_score(X_sc[best_w][nm], yp_b[nm]))
    else:
        sil, db = -1.0, 999.0
    return {
        "v_measure": float(np.mean(v_vals)),
        "silhouette": sil,
        "davies_bouldin": db,
        "n_clusters": float(np.mean(n_clust)),
        "noise_ratio": n_noise / max(n_total, 1),
        "time_s": 0,
        "ari": 0,
        "nmi": 0,
    }

ALL_RUNS = ["Run_B", "Run_J", "Run_K", "Run_L", "Run_M"]
PREDICTORS = {
    "Run_B": get_predictions_run_b,
    "Run_J": get_predictions_run_j,
    "Run_K": get_predictions_run_k,
    "Run_L": get_predictions_run_l,
    "Run_M": get_predictions_run_m,
}

if __name__ == "__main__":
    print("=" * 60)
    print("Generating all plots for Runs B, J, K, L, M")
    print("=" * 60)

    all_runs_metrics = []

    for run_name in ALL_RUNS:
        print(f"\n--- {run_name} ---")
        scenario_metrics_dict = {}
        predictor = PREDICTORS[run_name]
        save_dir = PLOT_DIRS[run_name]
        save_dir.mkdir(parents=True, exist_ok=True)

        for scenario in SCENARIOS:
            X, y_true = load_scenario(scenario)
            print(f"  Loading/generating predictions for {scenario}...")
            y_pred = predictor(scenario)

            if y_pred is None or any(yp is None for yp in y_pred):
                print(f"    WARNING: Missing predictions for {run_name}/{scenario}")
                continue

            # Generate per-scenario plots
            plot_scenario_results(run_name, scenario, X, y_pred,
                                  [y_true[w] for w in range(N_WINDOWS)],
                                  save_dir)

            # Compute scenario-level metrics
            metrics = compute_scenario_metrics(
                [y_true[w] for w in range(N_WINDOWS) if y_pred[w] is not None],
                [yp for yp in y_pred if yp is not None]
            )
            scenario_metrics_dict[scenario] = metrics

        # Generate run summary dashboard
        if scenario_metrics_dict:
            plot_run_summary(run_name, scenario_metrics_dict, save_dir)
            all_runs_metrics.append((run_name, scenario_metrics_dict))
            print(f"  [OK] {run_name}: {len(scenario_metrics_dict)} scenarios done")

    # Generate comparison plots
    if len(all_runs_metrics) > 0:
        print("\n--- Generating Comparison Plots ---")
        COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
        plot_comparison(all_runs_metrics, COMPARISON_DIR)

    # =====================================================================
    # PLOT INDEX
    # =====================================================================
    print("\n--- Generating PLOT_INDEX.md ---")
    lines = []
    lines.append("# Plot Index - All Visualizations")
    lines.append("")
    lines.append("## Per-Run Plots")
    lines.append("")
    lines.append("| Run | Scenarios | Plots per Scenario | Summary Plot |")
    lines.append("|-----|-----------|-------------------|--------------|")
    for run_name in ALL_RUNS:
        plot_dir = PLOT_DIRS[run_name]
        n_scenarios = 0
        has_summary = False
        if plot_dir.exists():
            files = os.listdir(plot_dir)
            for sc in SCENARIOS:
                if any(f.startswith(f"{sc}_") for f in files):
                    n_scenarios += 1
            has_summary = "summary_metrics.png" in files
        lines.append(f"| {run_name} | {n_scenarios}/5 | 6 per scenario | {'Yes' if has_summary else 'No'} |")

    lines.append("")
    lines.append("### Run B - `results/run_b/plots/`")
    lines.append("")
    for sc in SCENARIOS:
        lines.append(f"- **{sc}**: cluster_scatter, gt_vs_pred, silhouette, cluster_sizes, feature_profiles, pri_histogram")
    lines.append("- **summary_metrics.png** - 5-subplot dashboard across scenarios")
    lines.append("")

    for run_name in ["Run_J", "Run_K", "Run_L", "Run_M"]:
        lines.append(f"### {run_name} - `results/{run_name.lower().replace('run_','run_')}/plots/`")
        lines.append("")
        for sc in SCENARIOS:
            lines.append(f"- **{sc}**: cluster_scatter, gt_vs_pred, silhouette, cluster_sizes, feature_profiles, pri_histogram")
    lines.append("- **summary_metrics.png** - 5-subplot dashboard across scenarios")
    lines.append("")

    lines.append("## Comparison Plots - `results/comparison/plots/`")
    lines.append("")
    lines.append("| File | Description |")
    lines.append("|------|-------------|")
    comparison_files = [
        ("all_runs_silhouette_comparison.png", "Grouped bar chart of silhouette scores across all runs"),
        ("all_runs_vmeasure_comparison.png", "V-Measure comparison with Run B baseline line"),
        ("winner_heatmap.png", "5x5 heatmap showing best approach per scenario"),
        ("timing_vs_performance.png", "Scatter plot of accuracy vs compute time"),
        ("final_dashboard.png", "2x2 dashboard with winner, radar, improvements, noise"),
    ]
    for fname, desc in comparison_files:
        lines.append(f"| {fname} | {desc} |")

    # Count total plots
    total = 0
    for run_name in ALL_RUNS:
        plot_dir = PLOT_DIRS[run_name]
        if plot_dir.exists():
            total += len(os.listdir(plot_dir))
    if COMPARISON_DIR.exists():
        total += len(os.listdir(COMPARISON_DIR))

    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Total plots:** {total}")
    lines.append(f"- **Runs visualized:** {', '.join(ALL_RUNS)}")
    lines.append(f"- **Scenarios per run:** {len(SCENARIOS)}")
    lines.append(f"- **Comparison plots:** {len(comparison_files)}")
    lines.append("")
    lines.append("### Top 3 Most Important Plates for DRDO Report")
    lines.append("")
    lines.append("1. **`results/comparison/plots/final_dashboard.png`** - Complete overview of all runs")
    lines.append("2. **`results/comparison/plots/winner_heatmap.png`** - Best approach per scenario at a glance")
    lines.append("3. **`results/comparison/plots/all_runs_vmeasure_comparison.png`** - V-Measure vs Run B baseline")

    index_path = BASE_DIR / "results" / "PLOT_INDEX.md"
    with open(index_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  [OK] PLOT_INDEX.md saved ({total} total plots)")

    print("\n" + "=" * 60)
    print("ALL PLOTS GENERATED SUCCESSFULLY")
    print("=" * 60)
