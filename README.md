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
| 8 | `06_run_umap_hdbscan.py` | **Exp 2A:** UMAP 2D/3D + HDBSCAN comparison | 5-10 min |
| 9 | `07_run_gmm.py` | **Exp 2B:** GMM baseline with BIC selection | 4-5 min |
| 10 | `08_final_comparison.py` | Generate 4-way comparison table | 10 sec |

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
| `results/run_comparison.csv` | 3-run comparison (A vs B vs C) |
| `results_experiment2/summary_umap_hdbscan.csv` | UMAP 2D/3D + HDBSCAN results |
| `results_experiment2/summary_gmm.csv` | GMM baseline results |
| `results_experiment2/final_comparison_4way.csv` | 4-way comparison table |
| `results_runB_backup/` | Run B optimal config (reference) |
| `plots/*.png` | 7 visualization files |

## Experiment 2: Dimensionality Reduction & Algorithm Comparison

### Exp 2A: UMAP + HDBSCAN
UMAP reduces the 5 normalized PDW features to 2D and 3D before HDBSCAN clustering.

| Scenario | Run B (5D) | UMAP 2D | UMAP 3D |
|----------|:---------:|:-------:|:-------:|
| stare_low | **0.499** | 0.430 | 0.441 |
| stare_high | **0.902** | 0.578 | 0.597 |
| scan_low | **0.648** | 0.296 | 0.302 |
| scan_high | **0.871** | 0.730 | 0.754 |
| mixed | **0.810** | 0.436 | 0.472 |

**Finding:** UMAP consistently degrades performance (−14% to −54%). The 5D PDW space is already low-dimensional; UMAP discards signal, not noise.

### Exp 2B: GMM Baseline
Gaussian Mixture Models with BIC-based K selection (K=2..20) on the same 5 normalized features.

| Scenario | Run B (5D) | GMM (BIC) | Delta |
|----------|:---------:|:---------:|:-----:|
| stare_high | **0.902** | 0.866 | −4.0% |
| scan_high | **0.871** | 0.868 | −0.4% |
| scan_low | **0.648** | 0.585 | −9.7% |
| mixed | **0.810** | 0.678 | −16.2% |
| stare_low | **0.499** | 0.367 | −26.5% |

**Finding:** GMM is competitive on high-density scenarios (within 0.4-4%) but struggles with sparse emitters where the Gaussian assumption is violated.

## Experiment 1: Feature Engineering (Run A → B → C)

| Scenario | Run A (raw) | Run B (5 norm) | Run C (13 feat) |
|----------|:---------:|:-------------:|:--------------:|
| stare_low | 0.270 | **0.499** | 0.491 |
| stare_high | 0.290 | **0.902** | 0.443 |
| scan_low | 0.450 | **0.648** | 0.507 |
| scan_high | 0.590 | **0.871** | 0.648 |
| mixed | 0.350 | **0.810** | 0.615 |

**Finding:** Normalization provided massive gains (+44% to +211%). Adding 8 PRI-derived features (lag/lead/delta) caused "Feature Dilution" — cross-emitter interleaving makes PRI features noisy, degrading HDBSCAN's density neighborhoods.

## Resuming After Interruption

All scripts are **resumable**. If the pipeline is interrupted:
- `05_run_hdbscan.py` saves per-window results — re-run skips completed ones
- `02_download_data.py` checks existing .h5 files
- Simply re-run `python run_all.py` and it picks up where it left off

## System Requirements

- **Python**: 3.11+
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
