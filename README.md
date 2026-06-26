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
| 10 | `08_advanced_features.py` | **Exp 3:** PRI stats, FFT, UMAP 13D→3D | 40-50 min |
| 11 | `08_final_comparison.py` | Generate 4-way comparison table | 10 sec |
| 12 | `09_deep_error_analysis.py` | **Exp 4:** Deep failure mode analysis of Run B | 2-3 min |

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
| `results/run_comparison.csv` | 6-run comparison (A vs B vs C vs D vs E vs F) |
| `results/best_params_run_d_pristat.json` | Exp 3 — Best params for PRI statistics (Run D) |
| `results/best_params_run_e_fft.json` | Exp 3 — Best params for FFT features (Run E) |
| `results/best_params_run_f_umap13d.json` | Exp 3 — Best params for UMAP 13D→3D (Run F) |
| `results/summary_run_d_pristat.csv` | Exp 3 — Full evaluation, PRI statistics approach |
| `results/summary_run_e_fft.csv` | Exp 3 — Full evaluation, FFT approach |
| `results/summary_run_f_umap13d.csv` | Exp 3 — Full evaluation, UMAP 13D→3D approach |
| `results_experiment2/summary_umap_hdbscan.csv` | UMAP 2D/3D + HDBSCAN results |
| `results_experiment2/summary_gmm.csv` | GMM baseline results |
| `results_experiment2/final_comparison_4way.csv` | 4-way comparison table |
| `results_runB_backup/` | Run B optimal config (reference) |
| `results_experiment4/deep_error_analysis.md` | Exp 4 — Deep failure mode analysis report |
| `results_experiment4/error_analysis_summary.csv` | Exp 4 — Summary table of over/under-segmentation |
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

## Experiment 3: Advanced Feature Engineering (Runs D, E, F)

Three alternative approaches tested against the Run B (5 normalized PDW) baseline:

### Run D — Statistical PRI Aggregation (Window-Level)
Adds 3 window-level statistics (median PRI, IQR of PRI, PRI entropy) to the 5 PDW features. These features are **constant for all 1024 pulses in a window**, so after StandardScaler normalization they become all-zeros — inert dimensions.

| Scenario | Run B (5D) | Run D (8D) | Delta |
|----------|:---------:|:---------:|:-----:|
| stare_low | 0.499 | 0.499 | 0.0% |
| stare_high | 0.902 | 0.902 | 0.0% |
| scan_low | 0.648 | 0.648 | 0.0% |
| scan_high | 0.871 | 0.871 | 0.0% |
| mixed | 0.810 | 0.810 | 0.0% |

### Run E — Frequency Domain (FFT on ToA Sequence)
Extracts 3 dominant FFT frequencies from the detrended ToA sequence per window. Same issue — constant-per-window features normalize to zeros, producing identical results to Run B.

| Scenario | Run B (5D) | Run E (8D) | Delta |
|----------|:---------:|:---------:|:-----:|
| stare_low | 0.499 | 0.499 | 0.0% |
| stare_high | 0.902 | 0.902 | 0.0% |
| scan_low | 0.648 | 0.648 | 0.0% |
| scan_high | 0.871 | 0.871 | 0.0% |
| mixed | 0.810 | 0.810 | 0.0% |

### Run F — UMAP Manifold Reduction on 13D Space
Takes the diluted 13D feature space (Run C) and reduces to 3D via UMAP before HDBSCAN. Partially recovers from feature dilution but still underperforms the 5D baseline.

| Scenario | Run B (5D) | Run F (UMAP 3D) | Delta vs B | Run C (13D) | Delta vs C |
|----------|:---------:|:---------------:|:----------:|:----------:|:----------:|
| stare_low | **0.499** | 0.464 | −6.9% | 0.491 | −5.5% |
| stare_high | **0.902** | 0.579 | −35.8% | 0.443 | **+30.7%** |
| scan_low | **0.648** | 0.431 | −33.4% | 0.507 | −14.9% |
| scan_high | **0.871** | 0.656 | −24.7% | 0.648 | +1.2% |
| mixed | **0.810** | 0.565 | −30.3% | 0.615 | −8.2% |

### Experiment 3 Conclusions
1. **Window-level aggregate features are inert** — constant values across all pulses normalize to zeros, adding no discriminatory power. Pulse-level features (per-pulse PDW measurements) are essential for HDBSCAN's density-based clustering.
2. **UMAP partially rescues diluted features** — on stare_high, UMAP 13D→3D beats raw 13D by +30.7%, confirming manifold structure exists in the noisy 13D space. However, it still can't match the clean 5D baseline.
3. **The 5D normalized PDW space remains the sweet spot** — none of the 3 advanced approaches beat Run B in any scenario.

## Experiment 4: Deep Error Analysis of Run B

A detailed failure-mode analysis of the optimal HDBSCAN model (Run B, 5D normalized PDW) across all 5 scenarios. Computes confusion matrices, per-emitter purity/completeness, indistinguishable emitter checks, and noise point characterization.

### Global Failure Mode Summary

| Scenario | Over-seg. | Under-seg. | Noise% | Primary Failure |
|---------|----------|----------|------|----------------|
| stare_low | 5 | 4 | 24.6% | Over-segmentation |
| stare_high | 18 | 11 | 3.4% | Over-segmentation |
| scan_low | 5 | 8 | 1.6% | Under-segmentation |
| scan_high | 25 | 21 | 4.3% | Over-segmentation |
| mixed | 15 | 15 | 2.7% | Balanced |

### Key Findings

1. **24.6% noise in stare_low** — The optimal params (cs50_ms50_eps0.0) are too conservative for this scenario; many sparse emitters (E3/E5/E6/E13/E15/E16) have >95% noise rate, indicating low-density regions in the 5D PDW space.

2. **scenario_high scenarios are dominated by over-segmentation** — With 15-30 emitters, HDBSCAN splits individual emitters into sub-clusters due to intra-emitter PRI variation creating local density variations.

3. **212 unique merge patterns in scan_high** — The most complex scenario shows massive under-segmentation boundary overlap. Merged emitters have distinguishable mean parameters (Freq diff > 100 MHz, PW diff > 1 µs) but their distributions overlap at cluster boundaries.

4. **Physically indistinguishable emitters** — Some merges (e.g., scan_low E30+E71: ΔFreq=0.13 MHz, ΔPW=0.49 µs) show nearly identical PDW parameters, representing a data limitation rather than a clustering failure.

5. **Noise points correspond to boundary ambiguity** — Noise consistently has distinct mean Freq/PW from clustered points, confirming they occupy low-density regions between emitter clusters.

Full report: `results_experiment4/deep_error_analysis.md`

## Scalability Analysis: Full 70 GB TSRD Dataset

All experiments above used only the **validation subset** of TSRD (~5 GB, 500 .h5 files), which is **~7% of the complete 70 GB dataset**. Here is a practical assessment of running the same pipeline on the full dataset with this laptop's hardware.

### Hardware Constraints (This Laptop)

| Component | Specification |
|-----------|--------------|
| Model | HP Pavilion 15-eh2xxx |
| CPU | AMD Ryzen 5 5625U (6 cores / 12 threads, 2.3 GHz) |
| RAM | **8 GB (7.9 GB usable)** |
| Disk | ~158 GB SSD total, **~22 GB free** |
| GPU | Integrated Radeon Graphics (none) |

### Resource Requirements for Full Dataset

| Resource | Validation Subset (current) | Full 70 GB Dataset | Verdict |
|----------|:--------------------------:|:------------------:|:-------:|
| **Raw data on disk** | 5 GB | 70 GB | ❌ **22 GB free — can't fit** |
| **Download time** | ~30 min (500 files) | ~5-8 hours (5,512 files) | ⚠️ Doable overnight |
| **Scenario windowing** | ~30 sec per scenario | ~5 min per scenario | ✅ Fine (streaming) |
| **Parameter sweep (12 combos)** | 45-75 min per run | ~9-13 hours per run | ⚠️ Possible overnight |
| **Multiple runs (A–F)** | ~6 hours total | **~65 hours total** | ❌ Nearly 3 days nonstop |
| **Per-window JSON results** | ~80 MB per run | ~1-2 GB per run | ❌ < 22 GB free |
| **Evaluation (contingency matrices)** | 100 windows × 5-30 emitters | 10,000+ windows × 100+ emitters | ❌ **8 GB RAM → OOM** |

### Bottleneck Analysis

#### 1. Disk Space — Immediate Blocker
The full 70 GB raw dataset cannot be stored on this laptop (only 22 GB free). Even with an external drive:
- USB 3.0 sequential read ~100 MB/s → ~12 min to load the full dataset (once), but
- Random-access I/O for the pipeline (scanning 5,512 files for windowing) would be far slower
- Additionally, per-window JSON results for 6 runs would consume ~6-12 GB on top

#### 2. Memory — Evaluation Step Will OOM
The per-window HDBSCAN clustering (1024 points) fits easily in 8 GB RAM. However, the evaluation step (`06_evaluate.py`) builds global contingency matrices across all windows simultaneously. With 10,000+ windows and 100+ unique emitters:
- Contingency matrix: 100 × 100 × int64 = ~80 KB per window
- Aggregated: 10,000 × 80 KB = **800 MB** for a single matrix
- Multiple metrics (V-measure, ARI, AMI, NMI) each require separate passes
- Plus the existing overhead of loading all predictions into memory → **exceeds 8 GB**

#### 3. Compute Time — 9+ Days Total
The parameter sweep (`05_run_hdbscan.py`) is the bottleneck:
- Each window × parameter combination takes ~5 seconds
- Full dataset: ~10,000 windows (estimate) × 12 param combos = 120,000 HDBSCAN fits
- At 5 seconds each: **~167 hours per run**
- With 6 runs (A through F): **~1,000 hours = 42 days** on a single thread
- With `n_jobs=4`: **~10 days** of continuous computation
- This excludes the additional GMM (Exp 2B) and UMAP (Exp 2A, Exp 3) computations

#### 4. Results Storage — I/O Bottleneck
The current pipeline saves one JSON file per window per parameter set:
- Full dataset: 120,000+ JSON files per run = 720,000+ files for 6 runs
- Even at 10 KB each, that's **~7 GB of tiny files**
- File system overhead for 720K+ files on an SSD: degraded performance, slow `os.listdir()`, Windows file indexing crawl

### Would a Server Help? (Minimum Specs)

To run the full 70 GB TSRD pipeline efficiently:

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 32 GB | 64 GB |
| **Disk** | 200 GB free SSD | 500 GB NVMe |
| **CPU** | 8 cores | 16+ cores |
| **GPU** | Not required | Not required |

Even on a server, the total pipeline runtime would be **2-5 days** (with parallelism).

### Why We Used the Validation Subset

The decision to use only the validation subset was deliberate:

1. **It's the official benchmark split** — TSRD's paper uses validation for evaluation
2. **Representative diversity** — 500 files × ~30,000 pulses each captures the same emitter types, receiver modes, and SNR ranges as the full set
3. **8 GB RAM feasibility** — The validation subset fits in this laptop's memory at every pipeline step
4. **Rapid iteration** — 6-run experiment in hours vs weeks allows faster learning
5. **Same conclusions, less cost** — The key findings (normalization beats raw, feature dilution hurts, UMAP degrades) are structural properties of the feature space, not sampling artifacts

### Bottom Line

The 8 GB laptop can handle the validation subset end-to-end. Scaling to the full 70 GB TSRD is **not feasible** on this hardware — disk space, RAM (during evaluation), and compute time (10+ days) are all blockers. A server with 32+ GB RAM and 200+ GB free disk would be the minimum viable upgrade.

## Resuming After Interruption

All scripts are **resumable**. If the pipeline is interrupted:
- `05_run_hdbscan.py` saves per-window results — re-run skips completed ones
- `02_download_data.py` checks existing .h5 files
- Simply re-run `python run_all.py` and it picks up where it left off

## System Requirements

This pipeline is designed and tested for the **TSRD validation subset** (~5 GB, 500 .h5 files). See the Scalability Analysis section above for requirements to run on the full 70 GB dataset.

- **Python**: 3.11+
- **RAM**: 8 GB (tested for validation subset)
- **Disk**: 5 GB free minimum (3 GB data + 2 GB results) — 22+ GB recommended
- **GPU**: Not required (CPU-only, n_jobs=4)
- **Internet**: Required for download step
- **OS**: Windows (tested), Linux/macOS (should work)

## Dependencies

```
pip install -r requirements.txt
```

Key packages: `turing-deinterleaving-challenge`, `hdbscan`, `scikit-learn`,
`numpy`, `pandas`, `matplotlib`, `seaborn`, `joblib`, `tqdm`.
