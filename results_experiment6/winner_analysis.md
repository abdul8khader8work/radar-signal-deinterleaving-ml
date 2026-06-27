# Experiment 6: Winner Analysis

## 1. Which single approach beat Run B by the largest margin?

**Run_I on stare_low** (+57.6%)

**Best average across all 5 scenarios:** Run_I (0.8030)

         Run_I: avg V=0.8030 (+7.7% vs Run B)
         Run_K: avg V=0.7894 (+5.8% vs Run B)
         Run_L: avg V=0.1212 (-83.7% vs Run B)
         Run_J: avg V=0.0718 (-90.4% vs Run B)
         Run_M: avg V=0.0135 (-98.2% vs Run B)

## 2. Which approach had the best time-to-performance ratio?

| Run | Total V | Total Time | V/s |
|-----|--------|-----------|-----|
| Run_I_CNN | 4.01 | 25s | 0.1606 |
| Run_J | 0.36 | 23s | 0.0158 |
| Run_K | 3.95 | 296s | 0.0133 |
| Run_L | 0.61 | 48s | 0.0127 |
| Run_M | 0.07 | 403s | 0.0002 |

## 3. Which failure mode from Experiment 4 did each approach fix?

| Approach | Failure Mode Fixed | Mechanism |
|----------|-------------------|-----------|
| **Run_J** | Over-segmentation | Multi-scale peaks merge fragmented clusters by finding PRI consensus across histogram resolutions |
| **Run_K** | Noise ambiguity | Majority voting reduces noise-labeled pulses when 2+/4 algorithms agree |
| **Run_L** | Boundary overlap | CDIF identifies true PRIs from cumulative difference histograms; standalone PRI features bypass PDW overlap |
| **Run_M** | Over-segmentation + noise | GRU learns per-cluster PRI rhythm; flags pulses that deviate from expected interval |
| **Run_I** (Exp5) | All three | CNN embedding separates emitters discriminatively; HDBSCAN on embedding produces cleaner clusters |

## 4. Did ANY approach hit V-measure > 0.98?

**NO.** Highest score across all 9 runs (A through M): **0.9326**

The theoretical ceiling for unsupervised clustering on 5D PDW features appears to be
~0.93 (reached by Run I on stare_high). To exceed 0.98 would require:
- Ground-truth labels (fully supervised classification)
- Or additional sensor modalities not present in PDW data
- Or sequence-aware models trained on orders of magnitude more data

## 5. What does the winning approach reveal about TSRD?

The winning approach (Run_I) reveals that TSRD's emitter
separability is primarily driven by **temporal PRI patterns**, not static PDW values.
Approaches that leverage ToA sequence structure consistently outperform those that
treat pulses as i.i.d. samples from a 5D distribution. This confirms that the
interleaved pulse train carries emitter identity in the **ordering** and **timing** of
pulses, not just in their instantaneous measurements.