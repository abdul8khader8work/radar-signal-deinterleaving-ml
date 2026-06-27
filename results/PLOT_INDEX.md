# Plot Index - All Visualizations

## Per-Run Plots

| Run | Scenarios | Plots per Scenario | Summary Plot |
|-----|-----------|-------------------|--------------|
| Run_B | 5/5 | 6 per scenario | Yes |
| Run_J | 5/5 | 6 per scenario | Yes |
| Run_K | 5/5 | 6 per scenario | Yes |
| Run_L | 5/5 | 6 per scenario | Yes |
| Run_M | 5/5 | 6 per scenario | Yes |

### Run B - `results/run_b/plots/`

- **stare_low**: cluster_scatter, gt_vs_pred, silhouette, cluster_sizes, feature_profiles, pri_histogram
- **stare_high**: cluster_scatter, gt_vs_pred, silhouette, cluster_sizes, feature_profiles, pri_histogram
- **scan_low**: cluster_scatter, gt_vs_pred, silhouette, cluster_sizes, feature_profiles, pri_histogram
- **scan_high**: cluster_scatter, gt_vs_pred, silhouette, cluster_sizes, feature_profiles, pri_histogram
- **mixed**: cluster_scatter, gt_vs_pred, silhouette, cluster_sizes, feature_profiles, pri_histogram
- **summary_metrics.png** - 5-subplot dashboard across scenarios

### Run_J - `results/run_j/plots/`

- **stare_low**: cluster_scatter, gt_vs_pred, silhouette, cluster_sizes, feature_profiles, pri_histogram
- **stare_high**: cluster_scatter, gt_vs_pred, silhouette, cluster_sizes, feature_profiles, pri_histogram
- **scan_low**: cluster_scatter, gt_vs_pred, silhouette, cluster_sizes, feature_profiles, pri_histogram
- **scan_high**: cluster_scatter, gt_vs_pred, silhouette, cluster_sizes, feature_profiles, pri_histogram
- **mixed**: cluster_scatter, gt_vs_pred, silhouette, cluster_sizes, feature_profiles, pri_histogram
### Run_K - `results/run_k/plots/`

- **stare_low**: cluster_scatter, gt_vs_pred, silhouette, cluster_sizes, feature_profiles, pri_histogram
- **stare_high**: cluster_scatter, gt_vs_pred, silhouette, cluster_sizes, feature_profiles, pri_histogram
- **scan_low**: cluster_scatter, gt_vs_pred, silhouette, cluster_sizes, feature_profiles, pri_histogram
- **scan_high**: cluster_scatter, gt_vs_pred, silhouette, cluster_sizes, feature_profiles, pri_histogram
- **mixed**: cluster_scatter, gt_vs_pred, silhouette, cluster_sizes, feature_profiles, pri_histogram
### Run_L - `results/run_l/plots/`

- **stare_low**: cluster_scatter, gt_vs_pred, silhouette, cluster_sizes, feature_profiles, pri_histogram
- **stare_high**: cluster_scatter, gt_vs_pred, silhouette, cluster_sizes, feature_profiles, pri_histogram
- **scan_low**: cluster_scatter, gt_vs_pred, silhouette, cluster_sizes, feature_profiles, pri_histogram
- **scan_high**: cluster_scatter, gt_vs_pred, silhouette, cluster_sizes, feature_profiles, pri_histogram
- **mixed**: cluster_scatter, gt_vs_pred, silhouette, cluster_sizes, feature_profiles, pri_histogram
### Run_M - `results/run_m/plots/`

- **stare_low**: cluster_scatter, gt_vs_pred, silhouette, cluster_sizes, feature_profiles, pri_histogram
- **stare_high**: cluster_scatter, gt_vs_pred, silhouette, cluster_sizes, feature_profiles, pri_histogram
- **scan_low**: cluster_scatter, gt_vs_pred, silhouette, cluster_sizes, feature_profiles, pri_histogram
- **scan_high**: cluster_scatter, gt_vs_pred, silhouette, cluster_sizes, feature_profiles, pri_histogram
- **mixed**: cluster_scatter, gt_vs_pred, silhouette, cluster_sizes, feature_profiles, pri_histogram
- **summary_metrics.png** - 5-subplot dashboard across scenarios

## Comparison Plots - `results/comparison/plots/`

| File | Description |
|------|-------------|
| all_runs_silhouette_comparison.png | Grouped bar chart of silhouette scores across all runs |
| all_runs_vmeasure_comparison.png | V-Measure comparison with Run B baseline line |
| winner_heatmap.png | 5x5 heatmap showing best approach per scenario |
| timing_vs_performance.png | Scatter plot of accuracy vs compute time |
| final_dashboard.png | 2x2 dashboard with winner, radar, improvements, noise |

## Summary

- **Total plots:** 160
- **Runs visualized:** Run_B, Run_J, Run_K, Run_L, Run_M
- **Scenarios per run:** 5
- **Comparison plots:** 5

### Top 3 Most Important Plates for DRDO Report

1. **`results/comparison/plots/final_dashboard.png`** - Complete overview of all runs
2. **`results/comparison/plots/winner_heatmap.png`** - Best approach per scenario at a glance
3. **`results/comparison/plots/all_runs_vmeasure_comparison.png`** - V-Measure vs Run B baseline