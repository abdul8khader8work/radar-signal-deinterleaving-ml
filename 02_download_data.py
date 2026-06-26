"""
02_download_data.py — Download TSRD validation subset only

What this script does:
1. Lists validation files from TSRD using HF Hub API
2. Downloads only val_stare and val_scan files (~253 each, ~1-2 GB total)
3. Saves to D:\sem-6\...\data\validation\{stare,scan} as .h5 files
4. Resumes automatically if interrupted (skips existing files)

Why validation only:
- Training set is 2,503 files per mode (too large for 8GB RAM laptop)
- Validation is 253 files per mode, manageable with windowing

IMPORTANT: The TSRD dataset structure was updated recently.
  Old: validation/stare/*.h5, validation/scan/*.h5
  New: stare/val_stare/*.h5, scan/val_scan/*.h5
  This script uses the NEW structure.

Run: python 02_download_data.py
"""

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
import numpy as np

# Load .env
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

HF_TOKEN = os.getenv("HF_TOKEN", "")
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = Path(os.getenv("TSRD_DATA_DIR", BASE_DIR / "data"))


# ---------------------------------------------------------------------------
# STEP 1: List validation files from HF dataset
# ---------------------------------------------------------------------------

def get_validation_files():
    """
    Use Hugging Face Hub API to list the validation files.
    
    The TSRD dataset has this structure (as of June 2026):
      stare/val_stare/validation_*.h5  (253 files)
      scan/val_scan/validation_*.h5    (253 files)
    
    Returns:
      stare_files: list of file paths in HF repo (str)
      scan_files:  list of file paths in HF repo (str)
    """
    from huggingface_hub import HfApi

    print("=" * 60)
    print("Listing TSRD validation files on Hugging Face")
    print("=" * 60)

    api = HfApi(token=HF_TOKEN)
    repo_id = "alan-turing-institute/turing-synthetic-radar-dataset"
    
    # List all files in the repo
    all_files = api.list_repo_files(repo_id, repo_type="dataset")

    # Filter for validation files by mode (only .h5 files)
    stare_files = sorted([f for f in all_files if f.startswith("stare/val_stare/") and f.endswith(".h5")])
    scan_files = sorted([f for f in all_files if f.startswith("scan/val_scan/") and f.endswith(".h5")])

    print(f"  Stare validation files: {len(stare_files)}")
    print(f"  Scan validation files:  {len(scan_files)}")

    if len(stare_files) == 0 and len(scan_files) == 0:
        print("\n[ERROR] No validation files found in the dataset.")
        print("        The dataset structure may have changed again.")
        print("        Showing available directories:")
        dirs = set()
        for f in all_files:
            parts = f.split("/")
            if len(parts) >= 2:
                dirs.add("/".join(parts[:2]))
        for d in sorted(dirs):
            print(f"          {d}/")
        sys.exit(1)

    return stare_files, scan_files


# ---------------------------------------------------------------------------
# STEP 2: Download files
# ---------------------------------------------------------------------------

def download_files(hf_files, local_dir, label):
    """
    Download files from Hugging Face using hf_hub_download.
    
    hf_hub_download downloads a single file from a HF dataset/model repo.
    It caches downloads, so re-running is fast (just copies from cache).
    
    Parameters:
      hf_files: list of file paths in the HF repo (e.g., "stare/val_stare/validation_0.h5")
      local_dir: Path to save files locally
      label: "Stare" or "Scan" for progress display
    """
    from huggingface_hub import hf_hub_download

    repo_id = "alan-turing-institute/turing-synthetic-radar-dataset"
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Downloading {len(hf_files)} {label} files to {local_dir}")

    downloaded = 0
    skipped = 0
    start = time.time()

    for hf_path in tqdm(hf_files, desc=f"  {label}", unit="file"):
        # Extract the filename from the HF path
        filename = Path(hf_path).name  # e.g., "validation_0.h5"
        local_path = local_dir / filename

        # Skip if already exists (resume support)
        if local_path.exists():
            skipped += 1
            continue

        try:
            # Download from HF Hub (uses local cache)
            cached_path = hf_hub_download(
                repo_id=repo_id,
                filename=hf_path,
                repo_type="dataset",
                token=HF_TOKEN,
            )

            # Copy from cache to our local directory
            import shutil
            shutil.copy2(cached_path, local_path)
            downloaded += 1

        except Exception as e:
            print(f"\n  [WARNING] Failed to download {filename}: {e}")
            print("            Will retry on next run.")

    elapsed = time.time() - start
    mins, secs = divmod(elapsed, 60)

    print(f"  Downloaded: {downloaded}  |  Skipped (already exist): {skipped}")
    if downloaded > 0:
        print(f"  Time: {int(mins)}m {int(secs)}s")
    print()


# ---------------------------------------------------------------------------
# STEP 3: Quick verification
# ---------------------------------------------------------------------------

def verify_download():
    """
    Check that downloaded files are valid .h5 files by loading a few.
    """
    from turing_deinterleaving_challenge import PulseTrain

    print("=" * 60)
    print("Verifying downloaded files")
    print("=" * 60)

    stare_dir = DATA_DIR / "validation" / "stare"
    scan_dir = DATA_DIR / "validation" / "scan"

    h5_files = list(stare_dir.glob("*.h5")) + list(scan_dir.glob("*.h5"))

    if len(h5_files) == 0:
        print("  [WARNING] No .h5 files found!")
        return False

    # Test-load 3 random files
    import random
    test_files = random.sample(h5_files, min(3, len(h5_files)))

    for f in test_files:
        try:
            pt = PulseTrain.load(f)
            n_pulses = pt.data.shape[0]
            n_emitters = len(np.unique(pt.labels))
            mode = "stare" if "stare" in str(f) else "scan"
            print(f"  [OK] {f.name} ({mode}): {n_pulses:,} pulses, {n_emitters} emitters")
        except Exception as e:
            print(f"  [WARNING] Could not load {f.name}: {e}")

    print(f"\n  Total .h5 files on disk: {len(h5_files)}")
    return True


# ---------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Step 1: List files
    stare_hf_files, scan_hf_files = get_validation_files()

    # Step 2: Download stare validation files
    download_files(
        stare_hf_files,
        DATA_DIR / "validation" / "stare",
        "Stare",
    )

    # Step 3: Download scan validation files
    download_files(
        scan_hf_files,
        DATA_DIR / "validation" / "scan",
        "Scan",
    )

    # Step 4: Verify
    verify_download()
