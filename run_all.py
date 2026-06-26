"""
run_all.py — Execute the complete TSRD + HDBSCAN tutorial pipeline

Run this one script to execute steps 1-7 in order.
It checks for completion at each step and skips already-completed steps.

Usage:
    python run_all.py

What it runs:
    1. 01_setup.py         — Verify env + HF token
    2. 02_download_data.py — Download validation subset
    3. 03_explore_data.py  — Inspect data
    4. 04_create_scenarios.py — Build 5 scenarios with windowing
    5. 05_run_hdbscan.py   — HDBSCAN parameter sweep (longest step)
    6. 06_evaluate.py      — Compute metrics
    7. 07_visualize.py     — Generate plots

Checkpoints (script will SKIP steps where output already exists):
    - Step 2: checks data/validation/ for .h5 files
    - Step 4: checks scenarios/*.npz files
    - Step 5: checks results/*.json cache files
    - Step 6: checks results/summary_metrics.csv
"""

import subprocess
import sys
import time
from pathlib import Path


def run_step(script_name, checkpoint_condition=None, skip_message=""):
    """
    Run a Python script if its output doesn't already exist.
    
    Parameters:
        script_name: str, like "01_setup.py"
        checkpoint_condition: function returning True if step is already done
        skip_message: str, explanation for skipping
    """
    header = f"\n{'=' * 70}"
    header += f"\n>>> Running: {script_name}"
    header += f"\n{'=' * 70}"
    print(header)

    if checkpoint_condition and checkpoint_condition():
        print(f"  [SKIP] {skip_message}")
        print(f"  Output already exists. Delete it to re-run.")
        return True

    start = time.time()
    result = subprocess.run(
        [sys.executable, script_name],
        cwd=Path(__file__).parent,
    )
    elapsed = time.time() - start
    mins, secs = divmod(elapsed, 60)

    if result.returncode == 0:
        print(f"\n  [DONE] {script_name} finished in {int(mins)}m {int(secs)}s")
        return True
    else:
        print(f"\n  [ERROR] {script_name} failed with exit code {result.returncode}")
        print(f"  Fix the issue and re-run. The script will resume from this step.")
        return False


def main():
    base = Path(__file__).parent
    data_dir = base / "data" / "validation"
    scenarios_dir = base / "scenarios"
    results_dir = base / "results"

    print("=" * 70)
    print("TSRD + HDBSCAN Tutorial — Pipeline Runner")
    print("=" * 70)
    print(f"  Project: {base}")
    print(f"  Python:  {sys.executable}")
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Step 1: Setup
    ok = run_step("01_setup.py")
    if not ok:
        sys.exit(1)

    # Step 2: Download (check for any .h5 files)
    ok = run_step(
        "02_download_data.py",
        checkpoint_condition=lambda: len(list(data_dir.rglob("*.h5"))) > 10,
        skip_message="Download already complete (10+ .h5 files found).",
    )
    if not ok:
        sys.exit(1)

    # Step 3: Explore
    ok = run_step("03_explore_data.py")
    if not ok:
        sys.exit(1)

    # Step 4: Create scenarios (require all 5)
    SCENARIO_NAMES = ["stare_low", "stare_high", "scan_low", "scan_high", "mixed"]
    all_scenarios_exist = lambda: all(
        (scenarios_dir / f"{name}.npz").exists() for name in SCENARIO_NAMES
    )
    ok = run_step(
        "04_create_scenarios.py",
        checkpoint_condition=all_scenarios_exist,
        skip_message="All 5 scenario .npz files already exist.",
    )
    if not ok:
        sys.exit(1)

    # Step 5: HDBSCAN (longest step — check for any result .json files)
    ok = run_step(
        "05_run_hdbscan.py",
        checkpoint_condition=lambda: len(list(results_dir.glob("*.json"))) > 10,
        skip_message="HDBSCAN results already cached (10+ .json files found).",
    )
    if not ok:
        sys.exit(1)

    # Step 6: Evaluate
    ok = run_step(
        "06_evaluate.py",
        checkpoint_condition=lambda: (results_dir / "summary_metrics.csv").exists(),
        skip_message="summary_metrics.csv already exists.",
    )
    if not ok:
        sys.exit(1)

    # Step 7: Visualize
    ok = run_step("07_visualize.py")
    if not ok:
        sys.exit(1)

    # Summary
    print(f"\n{'=' * 70}")
    print("PIPELINE COMPLETE!")
    print(f"{'=' * 70}")
    print(f"  All 7 steps finished successfully.")
    print(f"  Plots:   {base / 'plots'}")
    print(f"  Summary: {results_dir / 'summary_metrics.csv'}")
    print(f"  Best params: {results_dir / 'best_params.json'}")
    print(f"\n  Open the .png files in plots/ to see results.")


if __name__ == "__main__":
    main()
