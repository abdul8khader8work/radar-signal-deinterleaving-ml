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
| 13 | `09_advanced_architectures.py` | **Exp 5:** Graph+HMM, CDIF, 1D-CNN embedding vs Run B | 2-3 min |
| 14 | `10_final_breakthrough.py` | **Exp 6:** Multi-scale PRI, Ensemble, CDIF standalone, Bi-GRU | 8-15 min |

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
| `results_experiment5/experiment5_final_verdict.md` | Exp 5 — Final verdict, analysis, executive summary |
| `results_experiment5/final_comparison_table.csv` | Exp 5 — Runs G/H/I vs Run B comparison table |
| `results_experiment5/Run_G_GraphHMM/aggregate_metrics.csv` | Exp 5 — Graph+HMM per-scenario metrics |
| `results_experiment5/Run_H_CDIF/aggregate_metrics.csv` | Exp 5 — CDIF+HDBSCAN per-scenario metrics |
| `results_experiment5/Run_I_CNN/aggregate_metrics.csv` | Exp 5 — 1D-CNN+HDBSCAN per-scenario metrics |
| `results_experiment6/final_comparison_table.md` | Exp 6 — Final comparison table (Runs J/K/L/M vs B) |
| `results_experiment6/winner_analysis.md` | Exp 6 — Winner analysis, 5 DRDO questions answered |
| `results_experiment6/production_recommendation.md` | Exp 6 — Production deployment guide + executive summary |
| `results_experiment6/experiment6_metrics.csv` | Exp 6 — Per-scenario metrics for Runs J/K/L/M |
| `plots/*.png` | 7 visualization files |
| `results/PLOT_INDEX.md` | Complete listing of all 160 generated plots |
| `results/run_b/plots/` | 31 per-run plots for Run B (5D HDBSCAN baseline) |
| `results/run_j/plots/` | 31 per-run plots for Run J (Multi-scale PRI) |
| `results/run_k/plots/` | 31 per-run plots for Run K (Ensemble Voting) |
| `results/run_l/plots/` | 31 per-run plots for Run L (CDIF Standalone) |
| `results/run_m/plots/` | 31 per-run plots for Run M (Bi-GRU Post-Processor) |
| `results/comparison/plots/` | 5 cross-run comparison plots (silhouette, V-measure, heatmap, timing, dashboard) |

## Visualizations

All 160 publication-ready plots are in `results/`. Each run directory contains 6 plots per scenario (cluster scatter, GT vs predicted, silhouette analysis, cluster sizes, feature profiles, PRI histogram) plus a 5-subplot summary dashboard.

### Per-Run Structure
```
results/run_X/plots/
  stare_low_cluster_scatter.png    # UMAP 2D cluster visualization
  stare_low_gt_vs_pred.png         # Ground truth vs predicted side-by-side
  stare_low_silhouette.png         # Silhouette coefficient per cluster
  stare_low_cluster_sizes.png      # Cluster size distribution bar chart
  stare_low_feature_profiles.png   # Feature profile per cluster (5D PDW)
  stare_low_pri_histogram.png      # PRI histogram colored by cluster
  ... (same 6 for each of 5 scenarios)
  summary_metrics.png              # 5-subplot dashboard for this run
```

### Comparison Plots
| File | Description |
|------|-------------|
| `all_runs_silhouette_comparison.png` | Grouped bar chart across all 5 runs |
| `all_runs_vmeasure_comparison.png` | V-Measure with Run B baseline line |
| `winner_heatmap.png` | 5x5 heatmap — best approach per scenario |
| `timing_vs_performance.png` | Accuracy vs compute time scatter |
| `final_dashboard.png` | 2x2 dashboard with winner, radar, improvements, noise |

See `results/PLOT_INDEX.md` for the complete listing of all 160 plots.

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

## Experiment 5: Targeted Breakthroughs (Runs G, H, I)

Three CPU-efficient approaches directly attacking the failure modes found in Experiment 4.
The 5D PDW ceiling was broken by a lightweight 1D-CNN that learns temporal PRI patterns.

### Final Results

| Scenario | Run B (5D PDW) | Run G (Graph+HMM) | Run H (CDIF) | Run I (CNN Emb) |
|----------|:--------------:|:-----------------:|:------------:|:---------------:|
| stare_low | 0.499 | 0.148 (−70%) | **0.599 (+20%)** | **0.786 (+58%)** |
| stare_high | 0.902 | 0.080 (−91%) | 0.754 (−16%) | **0.933 (+3%)** |
| scan_low | **0.648** | 0.126 (−81%) | 0.617 (−5%) | 0.583 (−10%) |
| scan_high | **0.871** | 0.058 (−93%) | 0.813 (−7%) | 0.862 (−1%) |
| mixed | 0.810 | 0.121 (−85%) | 0.802 (−1%) | **0.852 (+5%)** |

### Winner: Run I — 1D-CNN Embedding + HDBSCAN (+58% on stare_low)

A tiny 3-layer 1D-CNN (50K parameters) trained on 50 windows (~50K pulses, 3 seconds
CPU training) learns a discriminative 16D embedding from the ToA sequence. HDBSCAN on
this embedding beats Run B on 3/5 scenarios and achieves the single highest V-measure
across all 9 runs (stare_high: 0.933).

**Why it works:** The 1D convolutions extract PRI transition patterns directly from the
Time of Arrival sequence — information that no static per-pulse feature space (5D PDW, 13D
features, FFT, CDIF) can capture. This addresses the fundamental limitation of density-based
clustering on frame-level features.

### Honorable Mention: Run H — CDIF/PDIF Histogram Features (+20% on stare_low)

CDIF correctly identifies dominant PRIs per window (avg 4.5–5.0). When concatenated as
per-pulse PRI-match features with 5D PDW and passed to HDBSCAN, it beats Run B on the
sparse stare scenario. However, noise increases to 15–26% and dense scenarios (15-30
emitters) overwhelm the CDIF's 5-level histogram.

### Failed: Run G — Hybrid Graph+Louvain+HMM (V-measure 0.06–0.15)

The k-NN graph in PDW+PRI space does not partition into emitter communities — Louvain
detects ~9 clusters regardless of true emitter count. HMM merging with heuristic PRI
comparison was too aggressive. Not viable for production.

### Production Recommendation

**Two-stage hybrid pipeline:** Run B (HDBSCAN on 5D PDW) as primary classifier for speed,
with windows flagged as high-noise (>10% noise) routed to the 1D-CNN embedding model for
re-clustering. This gives Run B's speed on easy windows + CNN's precision on hard windows,
all within CPU-only constraints.

Full report: `results_experiment5/experiment5_final_verdict.md`

## Experiment 6: Four Final Approaches (Runs J, K, L, M)

Four additional CPU-efficient approaches tested against Run B. Only Ensemble Voting (Run K)
showed meaningful results; the others failed to surpass the baseline.

### Final Results

| Run | Approach | stare_low | stare_high | scan_low | scan_high | mixed | AVG | Noise | Time/Sc |
|-----|----------|:--------:|:---------:|:-------:|:---------:|:-----:|:---:|:-----:|:------:|
| **Run B** | 5D HDBSCAN (baseline) | 0.499 | 0.902 | 0.648 | 0.871 | 0.810 | 0.746 | 7.3% | 10s |
| **Run K** | Ensemble Voting (4 algos) | **0.771** | 0.890 | 0.629 | 0.838 | **0.819** | **0.789** | 0.7% | 59s |
| Run_J | Multi-scale PRI Histogram | 0.044 | 0.085 | 0.056 | 0.098 | 0.076 | 0.072 | 58.4% | 5s |
| Run_L | CDIF Standalone Peaks | 0.076 | 0.256 | 0.072 | 0.112 | 0.091 | 0.121 | 6.8% | 10s |
| Run_M | Bi-GRU Post-Processor | 0.000 | 0.004 | 0.004 | 0.011 | 0.048 | 0.014 | 7.3% | 81s |
| **Run I** | 1D-CNN Embedding (Exp 5) | **0.786** | **0.933** | 0.583 | 0.862 | **0.852** | **0.803** | 6.2% | 5s |

### Run-By-Run Analysis

**Run J — Multi-scale PRI Histogram (FAILED, V=0.07 avg)**
The multi-scale peak detection identifies too many spurious peaks from interleaved pulse
combinations. The tolerance-based pulse assignment (15% of interval) is too coarse for dense
scenarios. Average 58% noise rate means most pulses are unlabeled.

**Run K — Ensemble Voting (BEATS Run B on stare_low, V=0.771 vs 0.499)**
Majority voting across HDBSCAN + GMM + KMeans + Spectral clustering achieves near-zero
noise (0.1-1.9%) across all scenarios. Beats Run B on stare_low (the hardest scenario for
Run B) and nearly matches on stare_high (0.890 vs 0.902) and mixed (0.819 vs 0.810).
However, it loses on scan_low and scan_high where individual algorithms all perform poorly
and voting can't rescue them. The 59s/scenario cost is higher than Run B's 10s.

**Run L — CDIF Standalone Peaks (FAILED, V=0.12 avg)**
Using CDIF-extracted PRI peaks as the only feature source (no PDW context) is insufficient.
The per-pulse PRI-match features lose all information about Frequency, PW, AoA, and Amplitude —
the primary discriminants when multiple emitters share the same PRI. CDIF works as an
augmentation (Run H: +20% on stare_low) but not as a standalone feature.

**Run M — Bi-GRU Post-Processor (FAILED, V=0.01 avg)**
The GRU successfully learns per-cluster PRI rhythms (validated: low training error) but the
post-processing logic (flagging deviant pulses as noise, merging clusters with similar PRI
means) destroys the original Run B labels. The merge threshold (1.5× max std) is too
aggressive, collapsing all clusters into 1. Future work: use GRU prediction error as a
soft reweighting signal rather than a hard threshold for noise reassignment.

### Overall Ranking (Excluding Run B)

| Rank | Approach | Avg V | Time | Real-time? |
|:----:|----------|:-----:|:----:|:----------:|
| 1 | **Run I (1D-CNN)** | **0.803** | 5s | ✅ YES |
| 2 | Run K (Ensemble) | 0.789 | 59s | ⚠️ Batch |
| 3 | Run L (CDIF) | 0.121 | 10s | ✅ YES |
| 4 | Run J (Multi-scale PRI) | 0.072 | 5s | ✅ YES |
| 5 | Run M (Bi-GRU) | 0.014 | 81s | ❌ NO |

Full reports: `results_experiment6/winner_analysis.md`, `results_experiment6/production_recommendation.md`

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
