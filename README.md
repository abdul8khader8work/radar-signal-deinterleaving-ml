# TSRD + HDBSCAN Clustering Tutorial

A beginner-friendly tutorial for downloading the **Turing Synthetic Radar Dataset (TSRD)**,
creating 5 subset scenarios, applying **HDBSCAN clustering**, and comparing results.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run everything (steps 1-7 sequentially)
python run_all.py
```

## Pipeline Steps

| Step | Script | What it does | Est. time |
|------|--------|-------------|-----------|
| 1 | `01_setup.py` | Check Python, packages, HF token, disk space | 30 sec |
| 2 | `02_download_data.py` | Download TSRD validation subset | 15-40 min |
| 3 | `03_explore_data.py` | Inspect sample pulse trains | 1 min |
| 4 | `04_create_scenarios.py` | Build 5 scenario files (windowing) | 5 min |
| 5 | `05_run_hdbscan.py` | HDBSCAN parameter sweep | 45-75 min |
| 6 | `06_evaluate.py` | Compute V-measure, ARI, AMI, etc. | 2 min |
| 7 | `07_visualize.py` | Generate 7 comparison plots | 2 min |

## The 5 Scenarios

| ID | Mode | Emitters | Description |
|----|------|----------|-------------|
| `stare_low` | Stare | 2-5 | Easy: few emitters, full spectrum |
| `stare_high` | Stare | 15-30 | Challenging: dense emitter overlap |
| `scan_low` | Scan | 2-5 | Realistic receiver, few emitters |
| `scan_high` | Scan | 15-30 | Hardest: realistic + dense |
| `mixed` | Both | 5-20 | Generalization test across modes |

## HDBSCAN Parameters Tested

- **min_cluster_size**: 10, 20, 50
- **min_samples**: None (auto), same as min_cluster_size
- **cluster_selection_epsilon**: 0.0, 0.1
- **cluster_selection_method**: 'eom' (Excess of Mass)
- **metric**: 'euclidean'

Total: 12 combinations per scenario.

## Outputs

| Path | Contents |
|------|----------|
| `data/validation/` | Downloaded .h5 files (~1.5 GB) |
| `scenarios/*.npz` | Windowed data for each scenario |
| `results/*.json` | Per-window, per-param clustering results |
| `results/summary_metrics.csv` | Aggregated metrics table |
| `results/best_params.json` | Optimal parameters per scenario |
| `plots/*.png` | 7 visualization files |

## Resuming After Interruption

All scripts are **resumable**. If the pipeline is interrupted:
- `05_run_hdbscan.py` saves per-window results — re-run skips completed ones
- `02_download_data.py` checks existing .h5 files
- Simply re-run `python run_all.py` and it picks up where it left off

## System Requirements

- **Python**: 3.10+
- **RAM**: 8 GB (tested)
- **Disk**: 3 GB free (for data + results)
- **GPU**: Not required (CPU-only, n_jobs=4)
- **Internet**: Required for download step
- **OS**: Windows (tested), Linux/macOS (should work)

## Dependencies

```
pip install -r requirements.txt
```

Key packages: `turing-deinterleaving-challenge`, `hdbscan`, `scikit-learn`,
`numpy`, `pandas`, `matplotlib`, `seaborn`, `joblib`, `tqdm`.
