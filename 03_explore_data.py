"""
03_explore_data.py — Explore the downloaded TSRD data

What this script does:
1. Loads 3 sample pulse trains from stare and scan modes
2. Prints detailed statistics (n_pulses, n_emitters, feature ranges)
3. Estimates memory requirements for windowing
4. Shows PDW features: Time of Arrival, Frequency, Pulse Width, AoA, Amplitude
5. Generates a quick scatter plot of ToA vs Frequency (first 5000 pulses)

Run: python 03_explore_data.py
"""

import os
import random
import sys
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Use non-interactive backend to avoid GUI errors
matplotlib.use("Agg")

# Load .env
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = Path(os.getenv("TSRD_DATA_DIR", BASE_DIR / "data"))
PLOTS_DIR = Path(os.getenv("TSRD_PLOTS_DIR", BASE_DIR / "plots"))
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

from turing_deinterleaving_challenge import PulseTrain


# ---------------------------------------------------------------------------
# STEP 1: Find downloaded files
# ---------------------------------------------------------------------------

def find_files():
    """Locate downloaded .h5 files for stare and scan modes"""
    stare_dir = DATA_DIR / "validation" / "stare"
    scan_dir = DATA_DIR / "validation" / "scan"

    stare_files = sorted(stare_dir.glob("*.h5")) if stare_dir.exists() else []
    scan_files = sorted(scan_dir.glob("*.h5")) if scan_dir.exists() else []

    print("=" * 60)
    print("Data inventory")
    print("=" * 60)
    print(f"  Stare files: {len(stare_files)}")
    print(f"  Scan files:  {len(scan_files)}")

    if len(stare_files) == 0 and len(scan_files) == 0:
        print("\n[ERROR] No .h5 files found. Run 02_download_data.py first.")
        sys.exit(1)

    return stare_files, scan_files


# ---------------------------------------------------------------------------
# STEP 2: Load and inspect sample pulse trains
# ---------------------------------------------------------------------------

def inspect_pulse_trains(stare_files, scan_files):
    """
    Load 1-2 files from each mode and print detailed stats.
    
    A PulseTrain object has:
    - data:   numpy array of shape (n_pulses, 5) = PDW stream
    - labels: numpy array of shape (n_pulses,) = ground truth emitter IDs
    - metadata: dict with simulation parameters
    
    PDW columns: [Time of Arrival, Frequency, Pulse Width, AoA, Amplitude]
    """
    print("\n" + "=" * 60)
    print("Sample pulse train inspection")
    print("=" * 60)

    samples = []
    # Pick 2 stare and 1 scan (or fewer if not available)
    if len(stare_files) >= 2:
        samples.extend(random.sample(stare_files, 2))
    elif len(stare_files) == 1:
        samples.append(stare_files[0])
    if len(scan_files) >= 1:
        samples.append(random.sample(scan_files, 1)[0])

    for f in samples:
        pt = PulseTrain.load(f)
        n_pulses = pt.data.shape[0]
        n_emitters = len(np.unique(pt.labels))
        mode = "stare" if "stare" in str(f) else "scan"

        print(f"\n  File: {f.name}  ({mode})")
        print(f"  Pulses:       {n_pulses:,}")
        print(f"  Emitters:     {n_emitters}")
        print(f"  Data shape:   {pt.data.shape}")
        print(f"  Data dtype:   {pt.data.dtype}")
        print(f"  Label unique: {np.unique(pt.labels).tolist()[:10]}...")

        # Feature stats
        feature_names = ["ToA (us)", "Freq (MHz)", "PW (us)", "AoA (deg)", "Ampl (dB)"]
        print(f"  Feature ranges:")
        for i, name in enumerate(feature_names):
            col = pt.data[:, i]
            print(f"    {name:15s}  min={col.min():12.4f}  max={col.max():12.4f}  "
                  f"mean={col.mean():12.4f}")

        # Emitter label distribution (top 5)
        unique, counts = np.unique(pt.labels, return_counts=True)
        top5 = sorted(zip(counts, unique), reverse=True)[:5]
        print(f"  Top emitters (by pulse count):")
        for count, label in top5:
            pct = 100 * count / n_pulses
            print(f"    Emitter {label:4d}: {count:>8,} pulses ({pct:5.1f}%)")

        print()

    return samples


# ---------------------------------------------------------------------------
# STEP 3: Memory estimation
# ---------------------------------------------------------------------------

def estimate_memory():
    """
    Estimate memory needed for windowed processing.
    
    We process 1024-pulse windows at a time.
    One window of 5 PDW features = 1024 × 5 × 8 bytes = ~41 KB (float64)
    
    With n_jobs=4 we might have 4 windows in memory simultaneously.
    Total should stay under ~500 MB including overhead.
    """
    print("=" * 60)
    print("Memory estimate for windowed processing")
    print("=" * 60)

    window_size = 1024
    n_features = 5
    bytes_per_float = 8  # float64

    window_mb = (window_size * n_features * bytes_per_float) / (1024 * 1024)
    parallel_mb = window_mb * 4  # n_jobs=4
    batch_mb = window_mb * 10   # typical batch

    print(f"  Window size:                 {window_size} pulses")
    print(f"  One window memory:           {window_mb:.2f} MB")
    print(f"  4 parallel windows:          {parallel_mb:.2f} MB")
    print(f"  Typical batch (10 windows):  {batch_mb:.2f} MB")
    print(f"  Estimated peak:              <200 MB (very safe)")
    print()


# ---------------------------------------------------------------------------
# STEP 4: Quick scatter plot (ToA vs Frequency)
# ---------------------------------------------------------------------------

def plot_sample_scatter(samples):
    """
    Create a scatter plot of ToA vs Frequency for the first sample.
    
    This shows the raw interleaved data before clustering.
    Different colors = different emitters (ground truth).
    """
    pt = PulseTrain.load(samples[0])
    mode = "stare" if "stare" in str(samples[0]) else "scan"

    # Only plot first 5000 pulses for readability
    max_plot = min(5000, pt.data.shape[0])
    data_subset = pt.data[:max_plot]
    labels_subset = np.asarray(pt.labels[:max_plot]).ravel()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Without labels (what the clustering algorithm sees)
    axes[0].scatter(data_subset[:, 0], data_subset[:, 1], s=1, alpha=0.5, c="steelblue")
    axes[0].set_xlabel("Time of Arrival (us)")
    axes[0].set_ylabel("Frequency (MHz)")
    axes[0].set_title(f"Interleaved pulses ({mode}) — no labels")
    axes[0].grid(True, alpha=0.3)

    # Plot 2: With ground truth labels (what we want to recover)
    unique_labels = np.unique(labels_subset)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    for i, label in enumerate(unique_labels):
        mask = labels_subset == label
        axes[1].scatter(
            data_subset[mask, 0], data_subset[mask, 1],
            s=1, alpha=0.6, color=colors[i], label=f"Emitter {label}"
        )
    axes[1].set_xlabel("Time of Arrival (us)")
    axes[1].set_ylabel("Frequency (MHz)")
    axes[1].set_title(f"Pulses colored by emitter ({len(unique_labels)} emitters)")
    axes[1].legend(markerscale=5, fontsize=7, ncol=2)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = PLOTS_DIR / "explore_sample_scatter.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Sample scatter saved: {save_path}")


# ---------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    stare_files, scan_files = find_files()
    samples = inspect_pulse_trains(stare_files, scan_files)
    estimate_memory()
    plot_sample_scatter(samples)
    print("Done. Explore complete.")
