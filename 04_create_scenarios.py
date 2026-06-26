"""
04_create_scenarios.py — Create 5 windowed subsets/scenarios from TSRD

What this script does:
1. Uses DeinterleavingChallengeDataset to load .h5 files with windowing
2. Creates 5 scenarios with different emitter density / receiver mode
3. Each window = 1024 consecutive pulses (matching the official benchmark)
4. Saves each scenario as a compressed .npz file ready for clustering
5. Processes one window at a time (memory-safe for 8GB RAM)

The 5 scenarios:
  - stare_low:   Stare mode, few emitters (2-5)   — easiest
  - stare_high:  Stare mode, many emitters (15-30) — challenging overlap
  - scan_low:    Scan mode, few emitters (2-5)     — realistic receiver
  - scan_high:   Scan mode, many emitters (15-30)  — hardest
  - mixed:       Both modes (stare + scan), 6-20 emitters — cross-mode generalization

Run: python 04_create_scenarios.py
"""

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
from tqdm import tqdm

# Load .env
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

HF_TOKEN = os.getenv("HF_TOKEN", "")
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = Path(os.getenv("TSRD_DATA_DIR", BASE_DIR / "data"))
SCENARIOS_DIR = Path(os.getenv("TSRD_SCENARIOS_DIR", BASE_DIR / "scenarios"))
SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# STEP 1: Define scenarios
# ---------------------------------------------------------------------------

# Each scenario specifies:
#   local_path:  which receiver mode folder to read
#   min_emitters / max_emitters:  filter by number of emitters in train
#   n_windows:   how many 1024-pulse windows to collect
SCENARIO_CONFIGS = {
    "stare_low": {
        "mode": "stare",
        "min_emitters": 2,
        "max_emitters": 5,
        "n_windows": 100,
        "description": "Stare mode, 2-5 emitters (easy)"
    },
    "stare_high": {
        "mode": "stare",
        "min_emitters": 15,
        "max_emitters": 30,
        "n_windows": 100,
        "description": "Stare mode, 15-30 emitters (dense overlap)"
    },
    "scan_low": {
        "mode": "scan",
        "min_emitters": 2,
        "max_emitters": 5,
        "n_windows": 100,
        "description": "Scan mode, 2-5 emitters (realistic)"
    },
    "scan_high": {
        "mode": "scan",
        "min_emitters": 15,
        "max_emitters": 30,
        "n_windows": 100,
        "description": "Scan mode, 15-30 emitters (hardest)"
    },
    "mixed": {
        "mode": "both",
        "min_emitters": 6,
        "max_emitters": 20,
        "n_windows": 100,
        "description": "Mixed modes (stare + scan), 6-20 emitters (generalization)"
    },
}

WINDOW_LENGTH = 1024  # pulses per window (matches official TDC benchmark)


# ---------------------------------------------------------------------------
# STEP 2: Load data using DeinterleavingChallengeDataset
# ---------------------------------------------------------------------------

def load_scenario(scenario_name, config):
    """
    Use TDC's DeinterleavingChallengeDataset to load and window the data.
    
    How it works:
    - The dataset scans the given local_path for .h5 files
    - It loads pulse trains and slices them into windows of WINDOW_LENGTH pulses
    - Each window may overlap (depending on internal logic)
    - The dataset returns (window_data, window_labels) when indexed
    
    Memory safety:
    - We collect n_windows and stop — we never hold the full dataset in memory
    """
    from turing_deinterleaving_challenge import DeinterleavingChallengeDataset

    print(f"\n  Building: {scenario_name}")
    print(f"    {config['description']}")
    print(f"    Window size: {WINDOW_LENGTH} pulses, collecting {config['n_windows']} windows")

    # Handle "both" mode: combine windows from stare and scan
    if config["mode"] == "stare":
        paths = [str(DATA_DIR / "validation" / "stare")]
    elif config["mode"] == "scan":
        paths = [str(DATA_DIR / "validation" / "scan")]
    else:  # "both" — load from both subdirectories and combine
        paths = [
            str(DATA_DIR / "validation" / "stare"),
            str(DATA_DIR / "validation" / "scan"),
        ]
        print(f"    Loading from: stare/ + scan/")

    # Collect windows from each path
    all_X_list = []
    all_y_list = []
    windows_per_path = config["n_windows"] // len(paths)

    for path in paths:
        print(f"    Scanning {Path(path).name}...")
        dataset = DeinterleavingChallengeDataset(
            local_path=path,
            window_length=WINDOW_LENGTH,
            min_emitters=config["min_emitters"],
            max_emitters=config["max_emitters"],
        )

        total_windows = len(dataset)
        mode_name = Path(path).name
        print(f"      Available windows in {mode_name}: {total_windows}")

        if total_windows == 0:
            print(f"      [WARNING] No windows match in {mode_name}. Skipping.")
            continue

        n_to_collect = min(windows_per_path, total_windows)
        print(f"      Collecting {n_to_collect} windows from {mode_name}...")

        # Pre-allocate for speed
        X_part = np.zeros((n_to_collect, WINDOW_LENGTH, 5), dtype=np.float64)
        y_part = np.zeros((n_to_collect, WINDOW_LENGTH), dtype=np.int64)

        for i in tqdm(range(n_to_collect), desc=f"  {scenario_name} ({mode_name})", unit="window"):
            X_i, y_i = dataset[i]
            X_part[i] = X_i.numpy() if hasattr(X_i, "numpy") else X_i
            y_part[i] = y_i.numpy() if hasattr(y_i, "numpy") else y_i

        all_X_list.append(X_part)
        all_y_list.append(y_part)

    if not all_X_list:
        print(f"    [WARNING] No data for scenario '{scenario_name}'. Skipping.")
        return None, None

    # Combine parts
    all_X = np.concatenate(all_X_list, axis=0)
    all_y = np.concatenate(all_y_list, axis=0)

    print(f"    Collected: {all_X.shape[0]} windows x {WINDOW_LENGTH} pulses x 5 features")
    print(f"    Emitter labels range: {all_y.min()} to {all_y.max()}")

    return all_X, all_y


# ---------------------------------------------------------------------------
# STEP 3: Save scenario to disk
# ---------------------------------------------------------------------------

def save_scenario(scenario_name, X, y):
    """Save as compressed NumPy archive (.npz)"""
    save_path = SCENARIOS_DIR / f"{scenario_name}.npz"
    np.savez_compressed(save_path, X=X, y=y)
    file_size_mb = save_path.stat().st_size / (1024 * 1024)
    print(f"    Saved: {save_path.name} ({file_size_mb:.1f} MB)")
    return save_path


# ---------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Creating 5 scenarios from TSRD validation data")
    print("=" * 60)
    print(f"  Window length: {WINDOW_LENGTH} pulses")
    print(f"  Target:        {SCENARIOS_DIR}")
    print()

    start_total = time.time()
    results = {}

    for name, config in SCENARIO_CONFIGS.items():
        scenario_path = SCENARIOS_DIR / f"{name}.npz"

        # Skip if already exists (resume support)
        if scenario_path.exists():
            print(f"\n  Skipping {name} — already exists at {scenario_path}")
            data = np.load(scenario_path)
            results[name] = (data["X"], data["y"])
            data.close()
            continue

        X, y = load_scenario(name, config)

        if X is None:
            print(f"    [WARNING] No data for scenario '{name}'. Skipping.")
            continue

        save_scenario(name, X, y)
        results[name] = (X, y)

    elapsed = time.time() - start_total
    mins, secs = divmod(elapsed, 60)

    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    for name, (X, y) in results.items():
        n_emitters = len(np.unique(y))
        print(f"  {name:15s}: {X.shape[0]:4d} windows, {n_emitters:3d} unique emitters")
    print(f"\nTotal time: {int(mins)}m {int(secs)}s")
    print(f"Next: python 05_run_hdbscan.py")
