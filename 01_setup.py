"""
01_setup.py — Environment setup + HF token verification

What this script does:
1. Checks Python version (need >=3.10)
2. Verifies all required packages are installed
3. Tests your Hugging Face token against TSRD
4. Creates project directory structure
5. Reports system info: RAM free, disk free, CPU cores

Run: python 01_setup.py
"""

import os
import sys
import platform
import ctypes
from pathlib import Path
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# STEP 0: Load paths from .env
# ---------------------------------------------------------------------------

# Load .env from the same folder as this script
env_path = Path(__file__).parent / ".env"
if not env_path.exists():
    print("[ERROR] .env file not found at:", env_path)
    print("       Create one with your HF_TOKEN and paths.")
    sys.exit(1)

load_dotenv(dotenv_path=env_path)

HF_TOKEN = os.getenv("HF_TOKEN", "")
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = Path(os.getenv("TSRD_DATA_DIR", BASE_DIR / "data"))
SCENARIOS_DIR = Path(os.getenv("TSRD_SCENARIOS_DIR", BASE_DIR / "scenarios"))
RESULTS_DIR = Path(os.getenv("TSRD_RESULTS_DIR", BASE_DIR / "results"))
PLOTS_DIR = Path(os.getenv("TSRD_PLOTS_DIR", BASE_DIR / "plots"))
LOGS_DIR = Path(os.getenv("TSRD_LOGS_DIR", BASE_DIR / "logs"))

ALL_DIRS = [DATA_DIR, SCENARIOS_DIR, RESULTS_DIR, PLOTS_DIR, LOGS_DIR]


# ---------------------------------------------------------------------------
# STEP 1: Check Python version
# ---------------------------------------------------------------------------

def check_python():
    print("=" * 60)
    print("STEP 1: Checking Python version")
    print("=" * 60)
    version = sys.version_info
    print(f"  Python: {sys.version}")
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("  [ERROR] Python >=3.10 required. Please upgrade.")
        print("         Download: https://www.python.org/downloads/")
        sys.exit(1)
    print("  [OK] Python version meets requirements.\n")


# ---------------------------------------------------------------------------
# STEP 2: Check required packages
# ---------------------------------------------------------------------------

def check_packages():
    print("=" * 60)
    print("STEP 2: Checking required packages")
    print("=" * 60)
    required = [
        "numpy", "pandas", "sklearn", "hdbscan",
        "matplotlib", "seaborn", "tqdm", "joblib",
        "dotenv", "huggingface_hub", "h5py", "distinctipy",
    ]
    missing = []
    for pkg in required:
        try:
            if pkg == "dotenv":
                import dotenv
                print(f"  [OK] python-dotenv")
            else:
                mod = __import__(pkg)
                version = getattr(mod, "__version__", "unknown")
                print(f"  [OK] {pkg} ({version})")
        except ImportError:
            missing.append(pkg)
            print(f"  [MISSING] {pkg}")

    if missing:
        print("\n  [ERROR] Some packages are missing. Install with:")
        print(f"         pip install -r {BASE_DIR / 'requirements.txt'}")
        sys.exit(1)

    # Special check for turing_deinterleaving_challenge
    try:
        import turing_deinterleaving_challenge as tdc
        print(f"  [OK] turing-deinterleaving-challenge (v{tdc.__version__})")
    except ImportError:
        print("\n  [ERROR] turing-deinterleaving-challenge not found.")
        print("         Install from GitHub:")
        print("         pip install git+https://github.com/alan-turing-institute/turing-deinterleaving-challenge.git")
        sys.exit(1)

    print()


# ---------------------------------------------------------------------------
# STEP 3: Test HF token against TSRD
# ---------------------------------------------------------------------------

def test_hf_token():
    print("=" * 60)
    print("STEP 3: Testing Hugging Face token against TSRD")
    print("=" * 60)

    if not HF_TOKEN or HF_TOKEN.strip() == "":
        print("  [ERROR] No HF_TOKEN found in .env file.")
        print("         Add your token like: HF_TOKEN=hf_xxxxxxxxxxxx")
        sys.exit(1)

    # Show first 8 chars of token for verification (safe)
    print(f"  Token starts with: {HF_TOKEN[:8]}...")

    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError

    api = HfApi(token=HF_TOKEN)
    repo_id = "alan-turing-institute/turing-synthetic-radar-dataset"

    try:
        # Try to list the dataset files - this requires valid token + access
        files = api.list_repo_files(repo_id, repo_type="dataset")
        print(f"  [OK] Token accepted! Found {len(files)} files in the repository.")

        # Check for validation subsets
        validation_files = [f for f in files if f.startswith("validation")]
        print(f"       Validation files found: {len(validation_files)}")

        if len(validation_files) == 0:
            print("  [WARNING] No validation files found. The dataset structure")
            print("            may have changed. Continuing anyway...")
        else:
            print(f"  [OK] Dataset structure looks correct.")

    except HfHubHTTPError as e:
        status = e.response.status_code if hasattr(e, "response") else 0
        if status == 404:
            print("  [ERROR] Dataset 'alan-turing-institute/turing-synthetic-radar-dataset'")
            print("          not found. You may not have accepted the terms of use.")
            print("  Visit: https://huggingface.co/datasets/alan-turing-institute/")
            print("         turing-synthetic-radar-dataset")
            print("  Log in and accept the gated dataset terms.")
        elif status == 401:
            print("  [ERROR] Token rejected (401 Unauthorized).")
            print("          1. Go to https://huggingface.co/settings/tokens")
            print("          2. Create a new read token")
            print("          3. Update .env with the new token")
        else:
            print(f"  [ERROR] HF Hub error ({status}): {e}")
            print("  Check your token and internet connection.")
        sys.exit(1)
    except Exception as e:
        error_msg = str(e).lower()
        if "401" in error_msg or "unauthorized" in error_msg or "access" in error_msg:
            print("  [ERROR] Token rejected. The dataset requires:")
            print("          1. A Hugging Face account")
            print("          2. Accepting the dataset terms at:")
            print("             https://huggingface.co/datasets/alan-turing-institute/")
            print("             turing-synthetic-radar-dataset")
            print("          3. A valid token from: https://huggingface.co/settings/tokens")
            sys.exit(1)
        else:
            print(f"  [ERROR] Unexpected error: {e}")
            print("  Check your internet connection and try again.")
            sys.exit(1)

    print()


# ---------------------------------------------------------------------------
# STEP 4: Create project directories
# ---------------------------------------------------------------------------

def create_dirs():
    print("=" * 60)
    print("STEP 4: Creating project directories")
    print("=" * 60)
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  [OK] {d}")
    print()


# ---------------------------------------------------------------------------
# STEP 5: System info
# ---------------------------------------------------------------------------

def get_free_space_mb(path):
    """Get free disk space in MB using Windows GetDiskFreeSpaceEx"""
    try:
        drive = path.anchor  # e.g., "D:\\"
        free_bytes = ctypes.c_ulonglong(0)
        ctypes.windll.kernel32.GetDiskFreeSpaceExW(
            ctypes.c_wchar_p(drive), None, None, ctypes.pointer(free_bytes)
        )
        return free_bytes.value / (1024 * 1024)
    except Exception:
        return 0


def system_info():
    print("=" * 60)
    print("STEP 5: System information")
    print("=" * 60)
    print(f"  Platform: {platform.system()} {platform.release()}")
    print(f"  Machine:  {platform.machine()}")
    print(f"  CPU:      {os.cpu_count()} logical cores")

    # RAM (rough estimate via ctypes)
    try:
        kernel32 = ctypes.windll.kernel32
        mem_info = ctypes.c_ulonglong()
        kernel32.GetPhysicallyInstalledSystemMemory(ctypes.pointer(mem_info))
        ram_mb = mem_info.value / 1024
        print(f"  RAM:      {ram_mb:.0f} MB ({ram_mb / 1024:.1f} GB)")
    except Exception:
        print("  RAM:      (unable to detect)")

    free_mb = get_free_space_mb(BASE_DIR)
    print(f"  Disk free on {BASE_DIR.drive}: {free_mb:.0f} MB ({free_mb / 1024:.1f} GB)")

    if free_mb < 3000:
        print("  [WARNING] Less than 3 GB free. Need ~2 GB for this tutorial.")
        print("            Free up space or use a different drive.")

    print()


# ---------------------------------------------------------------------------
# STEP 6: Final summary
# ---------------------------------------------------------------------------

def summary():
    print("=" * 60)
    print("SUMMARY — All checks passed!")
    print("=" * 60)
    print(f"  Project:  {BASE_DIR}")
    print(f"  Python:   {sys.version_info.major}.{sys.version_info.minor}")
    print(f"  Token:    {HF_TOKEN[:8]}... (verified)")
    print(f"  n_jobs:   4 (we'll use 4 CPU cores)")
    print(f"")
    print("  Next: python 02_download_data.py")
    print("=" * 60)


# ---------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    check_python()
    check_packages()
    test_hf_token()
    create_dirs()
    system_info()
    summary()
