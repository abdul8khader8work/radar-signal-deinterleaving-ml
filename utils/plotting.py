"""
utils/plotting.py — Reusable plotting utilities for TSRD clustering experiments.

Provides:
  - plot_scenario_results(run_name, scenario, X, y_pred, y_true, save_dir)
  - plot_run_summary(run_name, metrics_dict, save_dir)
  - plot_comparison(all_runs_metrics, save_dir)
"""

import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import StandardScaler
import umap
import warnings
warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-darkgrid")
COLORS = plt.cm.tab10(np.linspace(0, 1, 30))
NOISE_COLOR = (0.4, 0.4, 0.4, 0.6)  # gray
SCENARIOS = ["stare_low", "stare_high", "scan_low", "scan_high", "mixed"]
FEATURE_NAMES = ["ToA", "Freq", "PW", "AoA", "Ampl"]


def _reduce_2d(X_window, random_state=42):
    """Reduce a single window (1024, 5) to 2D using UMAP."""
    scaler = StandardScaler()
    X_n = scaler.fit_transform(X_window)
    reducer = umap.UMAP(n_components=2, random_state=random_state, n_neighbors=30,
                        min_dist=0.1, verbose=False)
    return reducer.fit_transform(X_n)


def _get_colors(labels):
    """Map label values to consistent colors. -1 gets gray."""
    unique = sorted(set(l for l in labels if l >= 0))
    colormap = {}
    for i, u in enumerate(unique):
        colormap[u] = COLORS[i % len(COLORS)]
    colormap[-1] = NOISE_COLOR
    return colormap, len(unique)


def _plot_cluster_scatter(X_2d, labels, ax, title, show_centroids=True):
    colormap, n_clusters = _get_colors(labels)
    for lbl in sorted(set(labels)):
        mask = labels == lbl
        c = colormap[lbl]
        marker = "x" if lbl == -1 else "o"
        s = 8 if lbl == -1 else 12
        alpha = 0.4 if lbl == -1 else 0.7
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[c], s=s, marker=marker,
                   alpha=alpha, label=f"Cluster {lbl}" if lbl >= 0 else "Noise",
                   edgecolors="none")
    if show_centroids and n_clusters > 0:
        for lbl in set(l for l in labels if l >= 0):
            mask = labels == lbl
            if mask.sum() > 0:
                cx, cy = X_2d[mask].mean(axis=0)
                ax.scatter(cx, cy, marker="X", s=200, c="red", edgecolors="black",
                           linewidths=1.5, zorder=5)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("UMAP-1", fontsize=10)
    ax.set_ylabel("UMAP-2", fontsize=10)
    ax.legend(fontsize=7, loc="best", ncol=2)


# =====================================================================
# PUBLIC API
# =====================================================================

def plot_scenario_results(run_name, scenario, X, y_pred, y_true, save_dir):
    """
    Generate 6 per-scenario plots and save to save_dir.

    Parameters:
      run_name: str (e.g., "Run_B")
      scenario: str (e.g., "stare_low")
      X: np.ndarray (100, 1024, 5) — full scenario PDW data
      y_pred: list of np.ndarray — per-window predicted labels
      y_true: list of np.ndarray — per-window ground truth labels
      save_dir: Path — plots directory for this run
    """
    print(f"  Generating plots for {run_name}/{scenario}...")
    n_windows = X.shape[0]

    # Pick the best window (highest V-measure) for detailed plots
    best_win = 0
    best_v = -1
    for w in range(n_windows):
        yt_w = y_true[w]
        yp_w = y_pred[w]
        if yp_w is not None:
            n_mask = yp_w != -1
            if n_mask.sum() > 0 and len(set(yp_w[n_mask])) > 1:
                from sklearn.metrics import v_measure_score
                v = v_measure_score(yt_w[n_mask], yp_w[n_mask])
                if v > best_v:
                    best_v = v
                    best_win = w

    X_win = X[best_win]
    yp_win = y_pred[best_win]
    yt_win = y_true[best_win]

    if yp_win is None:
        print(f"    Skipping {scenario} — no predictions available")
        return

    # Reduce to 2D
    X_2d = _reduce_2d(X_win)

    # --- 1. Cluster Scatter ---
    fig, ax = plt.subplots(figsize=(10, 8))
    _plot_cluster_scatter(X_2d, yp_win, ax,
                          f"Run {run_name.replace('Run_','')} — {scenario} — Cluster Distribution (UMAP 2D)")
    n_clust = len(set(yp_win)) - (1 if -1 in yp_win else 0)
    n_noise = (yp_win == -1).sum()
    ax.text(0.02, 0.98, f"Clusters: {n_clust} | Noise: {n_noise} ({n_noise/len(yp_win):.1%})",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{scenario}_cluster_scatter.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # --- 2. GT vs Predicted ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    _plot_cluster_scatter(X_2d, yt_win, ax1,
                          f"Ground Truth — {scenario}", show_centroids=False)
    _plot_cluster_scatter(X_2d, yp_win, ax2,
                          f"Predicted — {run_name.replace('Run_','')} — {scenario}", show_centroids=False)
    plt.suptitle(f"GT vs Predicted: {scenario}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{scenario}_gt_vs_pred.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # --- 3. Silhouette Analysis ---
    from sklearn.metrics import silhouette_samples, silhouette_score
    mask = yp_win != -1
    if mask.sum() >= 2 and len(set(yp_win[mask])) > 1:
        sil_vals = silhouette_samples(X_2d[mask], yp_win[mask])
        sil_avg = silhouette_score(X_2d[mask], yp_win[mask])
        fig, ax = plt.subplots(figsize=(10, 6))
        y_lower = 10
        unique_labels = sorted(set(yp_win[mask]))
        for i, lbl in enumerate(unique_labels):
            i_mask = yp_win[mask] == lbl
            i_sil = sil_vals[i_mask]
            i_sil.sort()
            size = len(i_sil)
            y_upper = y_lower + size
            color = COLORS[i % len(COLORS)]
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, i_sil,
                             facecolor=color, alpha=0.7, label=f"Cluster {lbl}")
            ax.text(-0.05, y_lower + 0.5 * size, f"C{lbl}", fontsize=8)
            y_lower = y_upper + 10
        ax.axvline(x=sil_avg, color="red", linestyle="--",
                   label=f"Avg: {sil_avg:.3f}")
        ax.set_title(f"Run {run_name.replace('Run_','')} — {scenario} — Silhouette Analysis (avg={sil_avg:.3f})",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Silhouette Coefficient", fontsize=11)
        ax.set_ylabel("Cluster", fontsize=11)
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{scenario}_silhouette.png"),
                    dpi=150, bbox_inches="tight")
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Silhouette analysis not available\n(insufficient valid clusters)",
                ha="center", va="center", fontsize=14, transform=ax.transAxes)
        ax.set_title(f"Run {run_name.replace('Run_','')} — {scenario} — Silhouette Analysis (N/A)",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{scenario}_silhouette.png"),
                    dpi=150, bbox_inches="tight")
    plt.close()

    # --- 4. Cluster Size Distribution ---
    fig, ax = plt.subplots(figsize=(10, 6))
    counts = Counter(yp_win)
    sorted_labels = sorted([l for l in counts if l >= 0], key=lambda x: -counts[x])
    labels_str = [f"C{l}" for l in sorted_labels]
    sizes = [counts[l] for l in sorted_labels]
    noise_sz = counts.get(-1, 0)
    bars = ax.bar(range(len(sorted_labels)), sizes, color=[COLORS[i % len(COLORS)] for i in range(len(sorted_labels))])
    if noise_sz > 0:
        ax.bar(len(sorted_labels), noise_sz, color=NOISE_COLOR, label=f"Noise ({noise_sz})")
        labels_str.append("Noise")
        sizes.append(noise_sz)
    ax.set_xticks(range(len(labels_str)))
    ax.set_xticklabels(labels_str, fontsize=9)
    for i, (bar, sz) in enumerate(zip(bars, sizes[:len(bars)])):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(sz), ha="center", fontsize=9, fontweight="bold")
    ax.set_title(f"Run {run_name.replace('Run_','')} — {scenario} — Cluster Size Distribution",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Cluster", fontsize=11)
    ax.set_ylabel("Number of Pulses", fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{scenario}_cluster_sizes.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # --- 5. Feature Profiles (Box plots) ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    cluster_labels = sorted(set(yp_win))
    for fi in range(5):
        ax = axes[fi]
        data = []
        labels = []
        for lbl in cluster_labels:
            mask = yp_win == lbl
            if mask.sum() > 5:
                data.append(X_win[mask, fi])
                labels.append(f"C{lbl}" if lbl >= 0 else "Noise")
        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
            for patch, lbl in zip(bp["boxes"], cluster_labels):
                c = COLORS[cluster_labels.index(lbl) % len(COLORS)] if lbl >= 0 else NOISE_COLOR
                patch.set_facecolor(c)
                patch.set_alpha(0.6)
        ax.set_title(FEATURE_NAMES[fi], fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", labelsize=8)
    axes[5].axis("off")
    fig.suptitle(f"Run {run_name.replace('Run_','')} — {scenario} — Feature Profiles by Cluster",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{scenario}_feature_profiles.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # --- 6. PRI Histogram ---
    fig, ax = plt.subplots(figsize=(10, 6))
    toa = X_win[:, 0]
    sort_idx = np.argsort(toa)
    intervals = toa[sort_idx[1:]] - toa[sort_idx[:-1]]
    intervals = intervals[intervals > 0]
    if len(intervals) > 0:
        p5, p95 = np.percentile(intervals, [1, 99])
        intervals_clip = intervals[(intervals >= p5) & (intervals <= p95)]
        if len(intervals_clip) > 0:
            cluster_labels = sorted(set(yp_win))
            for lbl in cluster_labels:
                # Get intervals where consecutive pulses are in the same cluster
                c_mask = np.zeros(len(intervals), dtype=bool)
                for i in range(len(intervals)):
                    if (yp_win[sort_idx[i]] == lbl and yp_win[sort_idx[i + 1]] == lbl):
                        c_mask[i] = True
                c_intervals = intervals[c_mask]
                c_intervals = c_intervals[(c_intervals >= p5) & (c_intervals <= p95)]
                if len(c_intervals) > 10:
                    c = COLORS[cluster_labels.index(lbl) % len(COLORS)] if lbl >= 0 else NOISE_COLOR
                    ax.hist(c_intervals, bins=50, alpha=0.5, color=c,
                            label=f"Cluster {lbl}" if lbl >= 0 else "Noise", density=True)
    ax.set_title(f"Run {run_name.replace('Run_','')} — {scenario} — PRI Distribution by Cluster",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Pulse Repetition Interval (ToA diff)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{scenario}_pri_histogram.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    print(f"    Saved 6 plots for {run_name}/{scenario}")


def plot_run_summary(run_name, scenario_metrics, save_dir):
    """
    Generate a 5-subplot dashboard for a single run across all scenarios.

    Parameters:
      run_name: str
      scenario_metrics: dict {scenario: {v_measure, silhouette, davies_bouldin, n_clusters, noise_ratio}}
      save_dir: Path
    """
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    metrics_keys = ["v_measure", "silhouette", "davies_bouldin", "n_clusters", "noise_ratio"]
    titles = ["V-Measure", "Silhouette", "Davies-Bouldin", "Cluster Count", "Noise Ratio"]

    for idx, (mk, title) in enumerate(zip(metrics_keys, titles)):
        ax = axes[idx]
        scenarios_list = list(scenario_metrics.keys())
        values = [scenario_metrics[s].get(mk, 0) for s in scenarios_list]
        colors_bar = [plt.cm.Set2(i / len(scenarios_list)) for i in range(len(scenarios_list))]
        bars = ax.bar(scenarios_list, values, color=colors_bar, edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.02,
                    f"{val:.3f}", ha="center", fontsize=8, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        if mk in ("noise_ratio",):
            ax.set_ylim(0, max(values) * 1.3 if max(values) > 0 else 0.1)

    fig.suptitle(f"Run {run_name.replace('Run_','')} — Summary Metrics Across All Scenarios",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "summary_metrics.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved summary dashboard for {run_name}")


def plot_comparison(all_runs_data, save_dir):
    """
    Generate 5 comparison plots across all runs.

    Parameters:
      all_runs_data: list of (run_name, scenario_metrics_dict)
      save_dir: Path to results/comparison/plots/
    """
    run_names = [d[0] for d in all_runs_data]
    n_runs = len(run_names)

    # Build a DataFrame for easy plotting
    rows = []
    for run_name, s_metrics in all_runs_data:
        for scenario, metrics in s_metrics.items():
            row = metrics.copy()
            row["run"] = run_name.replace("Run_", "")
            row["scenario"] = scenario
            rows.append(row)
    df = pd.DataFrame(rows)

    colors_run = [plt.cm.tab10(i / max(n_runs, 1)) for i in range(n_runs)]
    run_labels_short = [r.replace("Run_", "") for r in run_names]

    # --- 1. Silhouette Comparison ---
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(SCENARIOS))
    w = 0.8 / n_runs
    for ri, (run_name, s_metrics) in enumerate(all_runs_data):
        vals = [s_metrics.get(s, {}).get("silhouette", 0) for s in SCENARIOS]
        bars = ax.bar(x + ri * w - 0.4 + w / 2, vals, w, label=run_labels_short[ri],
                      color=colors_run[ri], edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.2f}", ha="center", fontsize=7, rotation=45)
    ax.set_xticks(x)
    ax.set_xticklabels(SCENARIOS, fontsize=10)
    ax.set_title("Silhouette Score Comparison Across All Runs", fontsize=14, fontweight="bold")
    ax.set_ylabel("Silhouette Score", fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "all_runs_silhouette_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # --- 2. V-Measure Comparison ---
    fig, ax = plt.subplots(figsize=(12, 6))
    run_b_avg = 0
    for ri, (run_name, s_metrics) in enumerate(all_runs_data):
        vals = [s_metrics.get(s, {}).get("v_measure", 0) for s in SCENARIOS]
        bars = ax.bar(x + ri * w - 0.4 + w / 2, vals, w, label=run_labels_short[ri],
                      color=colors_run[ri], edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.2f}", ha="center", fontsize=7, rotation=45)
        if run_name == "Run_B":
            run_b_avg = np.mean(vals)
    ax.axhline(y=run_b_avg, color="red", linestyle="--", linewidth=1.5,
               label=f"Run B Avg: {run_b_avg:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(SCENARIOS, fontsize=10)
    ax.set_title("V-Measure Comparison (Baseline: Run B)", fontsize=14, fontweight="bold")
    ax.set_ylabel("V-Measure", fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "all_runs_vmeasure_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # --- 3. Winner Heatmap ---
    fig, ax = plt.subplots(figsize=(10, 6))
    heatmap_data = np.zeros((len(SCENARIOS), n_runs))
    for ri, (run_name, s_metrics) in enumerate(all_runs_data):
        for si, scenario in enumerate(SCENARIOS):
            heatmap_data[si, ri] = s_metrics.get(scenario, {}).get("v_measure", 0)
    im = ax.imshow(heatmap_data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    for si in range(len(SCENARIOS)):
        for ri in range(n_runs):
            val = heatmap_data[si, ri]
            txt_color = "white" if val > 0.5 else "black"
            ax.text(ri, si, f"{val:.3f}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=txt_color)
    # Highlight best per row
    for si in range(len(SCENARIOS)):
        best_col = np.argmax(heatmap_data[si])
        ax.add_patch(plt.Rectangle((best_col - 0.5, si - 0.5), 1, 1,
                                   fill=False, edgecolor="blue", linewidth=3))
    ax.set_xticks(range(n_runs))
    ax.set_xticklabels(run_labels_short, fontsize=9, rotation=30)
    ax.set_yticks(range(len(SCENARIOS)))
    ax.set_yticklabels(SCENARIOS, fontsize=9)
    ax.set_title("V-Measure Heatmap — Best Approach Per Scenario", fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "winner_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- 4. Performance vs Time Tradeoff ---
    fig, ax = plt.subplots(figsize=(10, 8))
    for ri, (run_name, s_metrics) in enumerate(all_runs_data):
        avg_v = np.mean([s_metrics.get(s, {}).get("v_measure", 0) for s in SCENARIOS])
        avg_time = np.mean([s_metrics.get(s, {}).get("time_s", 0) for s in SCENARIOS])
        ax.scatter(avg_time, avg_v, s=200, c=[colors_run[ri]], edgecolors="black",
                   linewidths=1.5, zorder=5)
        ax.annotate(run_labels_short[ri], (avg_time, avg_v),
                    textcoords="offset points", xytext=(10, 10), fontsize=11, fontweight="bold")
    # Quadrant lines
    med_time = np.median([np.mean([s_metrics.get(s, {}).get("time_s", 0) for s in SCENARIOS])
                          for _, s_metrics in all_runs_data])
    med_perf = np.median([np.mean([s_metrics.get(s, {}).get("v_measure", 0) for s in SCENARIOS])
                          for _, s_metrics in all_runs_data])
    ax.axvline(x=med_time, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(y=med_perf, color="gray", linestyle=":", alpha=0.5)
    ax.fill_between([0, med_time], med_perf, 1.1, alpha=0.1, color="green", label="Fast & Good")
    ax.set_xscale("log")
    ax.set_xlabel("Average Runtime per Scenario (seconds, log scale)", fontsize=11)
    ax.set_ylabel("Average V-Measure Across All Scenarios", fontsize=11)
    ax.set_title("Performance vs Computational Cost", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(left=max(0.1, min([np.mean([s_metrics.get(s, {}).get("time_s", 0) for s in SCENARIOS])
                                    for _, s_metrics in all_runs_data])) * 0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "timing_vs_performance.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- 5. Final Dashboard ---
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Top-left: Winner per scenario
    ax1 = fig.add_subplot(gs[0, 0])
    best_per_scenario = {}
    for s in SCENARIOS:
        best_v = -1
        best_r = ""
        for run_name, s_metrics in all_runs_data:
            v = s_metrics.get(s, {}).get("v_measure", 0)
            if v > best_v:
                best_v = v
                best_r = run_name.replace("Run_", "")
        best_per_scenario[s] = (best_r, best_v)
    sc_list = list(best_per_scenario.keys())
    best_names = [best_per_scenario[s][0] for s in sc_list]
    best_vals = [best_per_scenario[s][1] for s in sc_list]
    bar_colors = []
    for n in best_names:
        idx = run_labels_short.index(n) if n in run_labels_short else 0
        bar_colors.append(colors_run[idx])
    bars = ax1.bar(range(len(sc_list)), best_vals, color=bar_colors, edgecolor="black", linewidth=0.8)
    for i, (bar, val, name) in enumerate(zip(bars, best_vals, best_names)):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{name}\n({val:.3f})", ha="center", fontsize=9, fontweight="bold")
    ax1.set_xticks(range(len(sc_list)))
    ax1.set_xticklabels(sc_list, fontsize=9, rotation=20)
    ax1.set_ylabel("V-Measure", fontsize=11)
    ax1.set_title("Winner Per Scenario", fontsize=13, fontweight="bold")

    # Top-right: Average metrics radar chart
    ax2 = fig.add_subplot(gs[0, 1], projection="polar")
    metrics_for_radar = ["v_measure", "silhouette", "ari", "nmi"]
    radar_labels = ["V-Measure", "Silhouette", "ARI", "NMI"]
    n_radar = len(radar_labels)
    angles = np.linspace(0, 2 * np.pi, n_radar, endpoint=False).tolist()
    angles += angles[:1]
    for ri, (run_name, s_metrics) in enumerate(all_runs_data):
        vals = [np.mean([s_metrics.get(s, {}).get(m, 0) for s in SCENARIOS]) for m in metrics_for_radar]
        vals += vals[:1]
        ax2.plot(angles, vals, "o-", linewidth=2, label=run_labels_short[ri],
                 color=colors_run[ri])
        ax2.fill(angles, vals, alpha=0.1, color=colors_run[ri])
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(radar_labels, fontsize=9)
    ax2.set_title("Average Metrics — Radar", fontsize=13, fontweight="bold", pad=20)
    ax2.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.3, 1.1))

    # Bottom-left: Improvement over Run B
    ax3 = fig.add_subplot(gs[1, 0])
    run_b_metrics = None
    for run_name, s_metrics in all_runs_data:
        if run_name == "Run_B":
            run_b_metrics = s_metrics
            break
    if run_b_metrics:
        improvements = {}
        for run_name, s_metrics in all_runs_data:
            if run_name == "Run_B":
                continue
            imps = []
            for s in SCENARIOS:
                bv = run_b_metrics.get(s, {}).get("v_measure", 0)
                v = s_metrics.get(s, {}).get("v_measure", 0)
                imp = ((v - bv) / max(bv, 0.001)) * 100
                imps.append(imp)
            improvements[run_name.replace("Run_", "")] = imps
        x = np.arange(len(SCENARIOS))
        w2 = 0.8 / max(len(improvements), 1)
        for i, (rn, imps) in enumerate(improvements.items()):
            bars = ax3.bar(x + i * w2 - 0.4 + w2 / 2, imps, w2, label=rn,
                           color=colors_run[run_labels_short.index(rn)],
                           edgecolor="black", linewidth=0.5)
            for bar, imp in zip(bars, imps):
                ax3.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 1 if imp >= 0 else bar.get_height() - 5,
                         f"{imp:.0f}%", ha="center", fontsize=7)
        ax3.axhline(y=0, color="red", linestyle="-", linewidth=1)
        ax3.set_xticks(x)
        ax3.set_xticklabels(SCENARIOS, fontsize=9, rotation=20)
        ax3.set_ylabel("Improvement over Run B (%)", fontsize=11)
        ax3.set_title("Improvement Over Run B Per Scenario", fontsize=13, fontweight="bold")
        ax3.legend(fontsize=8)

    # Bottom-right: Noise comparison
    ax4 = fig.add_subplot(gs[1, 1])
    for ri, (run_name, s_metrics) in enumerate(all_runs_data):
        vals = [s_metrics.get(s, {}).get("noise_ratio", 0) * 100 for s in SCENARIOS]
        bars = ax4.bar(x + ri * w - 0.4 + w / 2, vals, w, label=run_labels_short[ri],
                       color=colors_run[ri], edgecolor="black", linewidth=0.5)
    ax4.set_xticks(x)
    ax4.set_xticklabels(SCENARIOS, fontsize=9, rotation=20)
    ax4.set_ylabel("Noise Percentage (%)", fontsize=11)
    ax4.set_title("Noise Percentage Comparison", fontsize=13, fontweight="bold")
    ax4.legend(fontsize=8)

    # Find overall winner
    best_avg = -1
    best_rn = ""
    for run_name, s_metrics in all_runs_data:
        avg_v = np.mean([s_metrics.get(s, {}).get("v_measure", 0) for s in SCENARIOS])
        if avg_v > best_avg:
            best_avg = avg_v
            best_rn = run_name.replace("Run_", "")
    fig.suptitle(f"FINAL RESULTS DASHBOARD — Overall Winner: {best_rn} (V={best_avg:.4f})",
                 fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "final_dashboard.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved 5 comparison plots to {save_dir}")
