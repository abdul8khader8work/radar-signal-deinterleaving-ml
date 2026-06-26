"""
07_visualize.py — Generate all comparison plots

What this script creates (7 plots):
1. scenario_overview.png       — Bar chart: pulses & emitters per scenario
2. pdw_scatter_best_each.png   — ToA vs Freq with best HDBSCAN predictions
3. metrics_comparison.png      — Grouped bars: V-measure, ARI, AMI per scenario
4. param_sensitivity.png       — Heatmap: min_cluster_size vs epsilon
5. noise_analysis.png          — Noise ratio vs scenario + emitter density
6. cluster_counts.png          — Predicted vs true clusters per scenario
7. summary_table.png           — Formatted metrics table as image

Run: python 07_visualize.py
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

# Use Agg backend for headless plotting (no GUI needed)
matplotlib.use("Agg")

# Load .env
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

BASE_DIR = Path(__file__).parent.resolve()
SCENARIOS_DIR = Path(os.getenv("TSRD_SCENARIOS_DIR", BASE_DIR / "scenarios"))
RESULTS_DIR = Path(os.getenv("TSRD_RESULTS_DIR", BASE_DIR / "results"))
PLOTS_DIR = Path(os.getenv("TSRD_PLOTS_DIR", BASE_DIR / "plots"))
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Color palette
PALETTE = sns.color_palette("Set2", 5)
SCENARIO_COLORS = {name: PALETTE[i] for i, name in enumerate(
    ["stare_low", "stare_high", "scan_low", "scan_high", "mixed"])}

# Style settings
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})


# ---------------------------------------------------------------------------
# STEP 1: Load data
# ---------------------------------------------------------------------------

def load_metrics():
    """Load the summary metrics CSV from step 06"""
    csv_path = RESULTS_DIR / "summary_metrics.csv"
    if not csv_path.exists():
        print("[ERROR] summary_metrics.csv not found. Run 06_evaluate.py first.")
        sys.exit(1)
    df = pd.read_csv(csv_path)
    # Only keep scenarios that have actual data
    all_scenarios = ["stare_low", "stare_high", "scan_low", "scan_high", "mixed"]
    present = [s for s in all_scenarios if s in df["scenario"].values]
    df["scenario"] = pd.Categorical(df["scenario"], categories=present, ordered=True)
    return df


def load_best_params():
    """Load best params JSON"""
    json_path = RESULTS_DIR / "best_params.json"
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)
    return {}


def load_scenario(name):
    """Load .npz for a scenario (for scatter plots)"""
    path = SCENARIOS_DIR / f"{name}.npz"
    if path.exists():
        data = np.load(path, allow_pickle=True)
        X, y = data["X"], data["y"]
        data.close()
        return X, y
    return None, None


# ---------------------------------------------------------------------------
# STEP 2: Plot 1 — Scenario overview
# ---------------------------------------------------------------------------

def plot_scenario_overview():
    """Bar chart comparing pulses and emitters across 5 scenarios"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    names = ["stare_low", "stare_high", "scan_low", "scan_high", "mixed"]

    n_pulses = []
    n_emitters = []

    for name in names:
        X, y = load_scenario(name)
        if X is not None:
            n_pulses.append(X.shape[0] * X.shape[1])
            n_emitters.append(len(np.unique(y)))
        else:
            n_pulses.append(0)
            n_emitters.append(0)

    # Plot pulses
    colors = [SCENARIO_COLORS[n] for n in names]
    axes[0].bar(names, n_pulses, color=colors, edgecolor="white", linewidth=0.5)
    axes[0].set_title("Total pulses collected")
    axes[0].set_ylabel("Pulse count")
    axes[0].tick_params(axis="x", rotation=30)

    # Add value labels
    for i, v in enumerate(n_pulses):
        axes[0].text(i, v + max(n_pulses) * 0.01, f"{v:,}", ha="center", fontsize=8)

    # Plot emitters
    axes[1].bar(names, n_emitters, color=colors, edgecolor="white", linewidth=0.5)
    axes[1].set_title("Unique emitters per scenario")
    axes[1].set_ylabel("Emitter count")
    axes[1].tick_params(axis="x", rotation=30)

    for i, v in enumerate(n_emitters):
        axes[1].text(i, v + 0.3, str(v), ha="center", fontsize=9)

    plt.tight_layout()
    save_path = PLOTS_DIR / "scenario_overview.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {save_path.name}")


# ---------------------------------------------------------------------------
# STEP 3: Plot 2 — PDW scatter with best predictions
# ---------------------------------------------------------------------------

def load_cached_prediction(scenario_name, w_idx, param_hash):
    """Load a single cached prediction"""
    path = RESULTS_DIR / f"{scenario_name}_w{w_idx:04d}_p{param_hash}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def plot_pdw_scatter():
    """
    For each scenario, show a 2-panel plot:
    Left:  Ground truth labels (ToA vs Frequency)
    Right: Best HDBSCAN prediction
    """
    best_params = load_best_params()
    if not best_params:
        print("  [SKIP] No best_params.json found. Run 06_evaluate.py.")
        return

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    names = ["stare_low", "stare_high", "scan_low", "scan_high", "mixed"]

    for col, name in enumerate(names):
        X, y_true = load_scenario(name)
        if X is None:
            continue

        # Use first window
        x_plot = X[0, :, 0]  # ToA
        y_plot = X[0, :, 1]  # Frequency

        # --- Top row: ground truth ---
        ax_t = axes[0, col]
        unique_true = np.unique(y_true[0])
        colors_t = plt.cm.tab10(np.linspace(0, 1, len(unique_true)))
        for i, label in enumerate(unique_true):
            mask = y_true[0] == label
            ax_t.scatter(x_plot[mask], y_plot[mask], s=2, alpha=0.6,
                        color=colors_t[i])
        ax_t.set_title(f"{name}\n(true, {len(unique_true)} emitters)", fontsize=9)
        ax_t.set_xlabel("ToA (us)" if col == 0 else "")
        ax_t.set_ylabel("Freq (MHz)" if col == 0 else "")

        # --- Bottom row: best prediction ---
        ax_b = axes[1, col]
        if name in best_params:
            bp = best_params[name]
            param_label = bp["param_label"]
            # Find matching param hash
            for params in [
                {"min_cluster_size": 10, "min_samples": None, "cluster_selection_epsilon": 0.0, "cluster_selection_method": "eom", "metric": "euclidean"},
                {"min_cluster_size": 10, "min_samples": None, "cluster_selection_epsilon": 0.1, "cluster_selection_method": "eom", "metric": "euclidean"},
                {"min_cluster_size": 10, "min_samples": 10, "cluster_selection_epsilon": 0.0, "cluster_selection_method": "eom", "metric": "euclidean"},
                {"min_cluster_size": 10, "min_samples": 10, "cluster_selection_epsilon": 0.1, "cluster_selection_method": "eom", "metric": "euclidean"},
                {"min_cluster_size": 20, "min_samples": None, "cluster_selection_epsilon": 0.0, "cluster_selection_method": "eom", "metric": "euclidean"},
                {"min_cluster_size": 20, "min_samples": None, "cluster_selection_epsilon": 0.1, "cluster_selection_method": "eom", "metric": "euclidean"},
                {"min_cluster_size": 20, "min_samples": 20, "cluster_selection_epsilon": 0.0, "cluster_selection_method": "eom", "metric": "euclidean"},
                {"min_cluster_size": 20, "min_samples": 20, "cluster_selection_epsilon": 0.1, "cluster_selection_method": "eom", "metric": "euclidean"},
                {"min_cluster_size": 50, "min_samples": None, "cluster_selection_epsilon": 0.0, "cluster_selection_method": "eom", "metric": "euclidean"},
                {"min_cluster_size": 50, "min_samples": None, "cluster_selection_epsilon": 0.1, "cluster_selection_method": "eom", "metric": "euclidean"},
                {"min_cluster_size": 50, "min_samples": 50, "cluster_selection_epsilon": 0.0, "cluster_selection_method": "eom", "metric": "euclidean"},
                {"min_cluster_size": 50, "min_samples": 50, "cluster_selection_epsilon": 0.1, "cluster_selection_method": "eom", "metric": "euclidean"},
            ]:
                import hashlib, json
                ph = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]
                pred = load_cached_prediction(name, 0, ph)
                if pred is not None and param_label in str(ph):
                    break

            if pred is not None:
                y_pred = np.array(pred["labels"])
                unique_pred = np.unique(y_pred)
                noise_mask = y_pred == -1

                # Plot noise in gray
                if noise_mask.any():
                    ax_b.scatter(x_plot[noise_mask], y_plot[noise_mask],
                               s=2, alpha=0.4, color="lightgray")

                # Plot clusters
                cluster_labels = [u for u in unique_pred if u != -1]
                colors_p = plt.cm.Set2(np.linspace(0, 1, len(cluster_labels) + 1))
                for i, label in enumerate(cluster_labels):
                    mask = y_pred == label
                    ax_b.scatter(x_plot[mask], y_plot[mask], s=2, alpha=0.6,
                               color=colors_p[i % len(colors_p)])
                n_noise = int(noise_mask.sum())
                ax_b.set_title(f"HDBSCAN V={bp['v_measure']:.2f}\n"
                             f"{len(cluster_labels)} cls, {n_noise} noise", fontsize=9)
            else:
                ax_b.set_title("No cached prediction", fontsize=9)
        else:
            ax_b.set_title("No best params", fontsize=9)

        ax_b.set_xlabel("ToA (us)" if col == 0 else "")

    plt.tight_layout()
    save_path = PLOTS_DIR / "pdw_scatter_best_each.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {save_path.name}")


# ---------------------------------------------------------------------------
# STEP 4: Plot 3 — Metrics comparison
# ---------------------------------------------------------------------------

def plot_metrics_comparison(df):
    """Grouped bar chart of V-measure, ARI, AMI per scenario"""
    # For each scenario, take the best (highest V-measure) params
    best_idx = df.groupby("scenario", observed=True)["v_measure_mean"].idxmax()
    best_df = df.loc[best_idx]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(best_df))
    width = 0.25

    metrics = ["v_measure_mean", "ari_mean", "ami_mean"]
    labels = ["V-measure", "ARI", "AMI"]

    for i, (metric, label) in enumerate(zip(metrics, labels)):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, best_df[metric], width, label=label,
                      color=sns.color_palette("husl", 3)[i], edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(best_df["scenario"], rotation=20)
    ax.set_ylabel("Score")
    ax.set_title("Best clustering metrics per scenario")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1)

    plt.tight_layout()
    save_path = PLOTS_DIR / "metrics_comparison.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {save_path.name}")


# ---------------------------------------------------------------------------
# STEP 5: Plot 4 — Parameter sensitivity heatmap
# ---------------------------------------------------------------------------

def plot_param_sensitivity(df):
    """
    Heatmap: min_cluster_size (x) vs cluster_selection_epsilon (y) → V-measure
    Shown for each scenario as a subplot.
    """
    scenarios = df["scenario"].unique()
    n = len(scenarios)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    for idx, scenario in enumerate(scenarios):
        ax = axes[idx]
        subset = df[df["scenario"] == scenario].copy()

        # Parse min_cluster_size and epsilon into numeric
        subset["cs"] = subset["param_label"].str.extract(r"cs(\d+)").astype(int)
        subset["eps"] = subset["param_label"].str.extract(r"eps([\d.]+)").astype(float)

        pivot = subset.pivot_table(
            index="eps", columns="cs", values="v_measure_mean", aggfunc="mean"
        )

        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd",
                   ax=ax, cbar_kws={"label": "V-measure"},
                   linewidths=0.5, linecolor="white")
        ax.set_title(f"{scenario}")
        ax.set_xlabel("min_cluster_size")
        ax.set_ylabel("epsilon")

    # Hide unused subplots
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    save_path = PLOTS_DIR / "param_sensitivity.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {save_path.name}")


# ---------------------------------------------------------------------------
# STEP 6: Plot 5 — Noise analysis
# ---------------------------------------------------------------------------

def plot_noise_analysis(df):
    """
    Bar chart: noise ratio per scenario, with best and worst params.
    Shows how much data HDBSCAN discards as noise.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    scenarios = df["scenario"].unique()

    # Get noise ratio range per scenario
    noise_data = []
    for scenario in scenarios:
        subset = df[df["scenario"] == scenario]
        min_noise = subset["noise_ratio_mean"].min()
        max_noise = subset["noise_ratio_mean"].max()
        best_v_idx = subset["v_measure_mean"].idxmax()
        best_noise = subset.loc[best_v_idx, "noise_ratio_mean"]
        noise_data.append({
            "scenario": scenario,
            "min_noise": min_noise,
            "max_noise": max_noise,
            "best_noise": best_noise,
        })

    ndf = pd.DataFrame(noise_data)
    x = np.arange(len(scenarios))
    width = 0.3

    ax.bar(x - width/2, ndf["min_noise"], width, label="Min noise", color="steelblue", alpha=0.7)
    ax.bar(x + width/2, ndf["max_noise"], width, label="Max noise", color="coral", alpha=0.7)
    ax.scatter(x, ndf["best_noise"], s=80, color="black", zorder=5,
              label="Best param (by V-measure)", marker="D")

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=20)
    ax.set_ylabel("Noise ratio")
    ax.set_title("HDBSCAN noise ratio per scenario (min / max / best)")
    ax.legend()

    plt.tight_layout()
    save_path = PLOTS_DIR / "noise_analysis.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {save_path.name}")


# ---------------------------------------------------------------------------
# STEP 7: Plot 6 — Predicted vs true clusters
# ---------------------------------------------------------------------------

def plot_cluster_counts(df):
    """
    Scatter plot: predicted clusters vs true clusters.
    Points on the diagonal = perfect count estimation.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    scenarios = df["scenario"].unique()

    for scenario in scenarios:
        subset = df[df["scenario"] == scenario]
        ax.scatter(
            subset["n_clusters_pred_mean"],
            [len(np.unique(np.load(SCENARIOS_DIR / f"{scenario}.npz", allow_pickle=True)["y"][0]))
             for _ in range(len(subset))],
            label=scenario, s=40, alpha=0.7,
            color=SCENARIO_COLORS.get(scenario, "gray")
        )

    max_val = max(df["n_clusters_pred_mean"].max(), 5) + 2
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.3, label="Perfect (y=x)")

    ax.set_xlabel("Predicted clusters (HDBSCAN)")
    ax.set_ylabel("True emitters")
    ax.set_title("Cluster count: predicted vs true")
    ax.legend()
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)

    plt.tight_layout()
    save_path = PLOTS_DIR / "cluster_counts.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {save_path.name}")


# ---------------------------------------------------------------------------
# STEP 8: Plot 7 — Summary table
# ---------------------------------------------------------------------------

def plot_summary_table(df):
    """
    Create a formatted table image of the best metrics per scenario.
    """
    best_idx = df.groupby("scenario", observed=True)["v_measure_mean"].idxmax()
    best_df = df.loc[best_idx].copy()

    # Format columns
    display_df = best_df[["scenario", "param_label", "v_measure_mean", "ari_mean",
                          "ami_mean", "noise_ratio_mean", "n_windows"]].copy()
    display_df.columns = ["Scenario", "Best Params", "V-measure", "ARI", "AMI",
                          "Noise Ratio", "Windows"]
    display_df["V-measure"] = display_df["V-measure"].round(3)
    display_df["ARI"] = display_df["ARI"].round(3)
    display_df["AMI"] = display_df["AMI"].round(3)
    display_df["Noise Ratio"] = display_df["Noise Ratio"].round(2)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Style header
    for j in range(len(display_df.columns)):
        table[0, j].set_facecolor(sns.color_palette("Set2")[0])
        table[0, j].set_text_props(weight="bold", color="white")

    # Alternate row colors
    for i in range(1, len(display_df) + 1):
        for j in range(len(display_df.columns)):
            if i % 2 == 0:
                table[i, j].set_facecolor("#f5f5f5")

    ax.set_title("Best HDBSCAN Performance per Scenario", fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    save_path = PLOTS_DIR / "summary_table.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {save_path.name}")


# ---------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Generating visualizations")
    print("=" * 60)

    df = load_metrics()
    print(f"  Loaded {len(df)} scenario-param combinations")

    plot_scenario_overview()
    plot_pdw_scatter()
    plot_metrics_comparison(df)
    plot_param_sensitivity(df)
    plot_noise_analysis(df)
    plot_cluster_counts(df)
    plot_summary_table(df)

    print(f"\n{'=' * 60}")
    print(f"All plots saved to: {PLOTS_DIR}")
    print(f"{'=' * 60}")
    print(f"Files:")
    for p in sorted(PLOTS_DIR.glob("*.png")):
        print(f"  {p.name}  ({p.stat().st_size // 1024} KB)")
    print(f"\nDone!")
