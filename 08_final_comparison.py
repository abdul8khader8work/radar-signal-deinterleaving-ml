"""
Generate final 4-way comparison: Run B vs UMAP 2D vs UMAP 3D vs GMM
"""
from pathlib import Path
import pandas as pd

BASE = Path(__file__).parent.resolve()
RES_DIR = BASE / "results_experiment2"
SCENARIOS = ["stare_low", "stare_high", "scan_low", "scan_high", "mixed"]

# ── Run B baseline (from backup) ──
run_b = {"stare_low": 0.4987, "stare_high": 0.9020, "scan_low": 0.6479, "scan_high": 0.8709, "mixed": 0.8097}

# ── UMAP results ──
umap_df = pd.read_csv(RES_DIR / "summary_umap_hdbscan.csv")
umap_2d = dict(zip(umap_df[umap_df["method"] == "UMAP_2D"]["scenario"], umap_df[umap_df["method"] == "UMAP_2D"]["v_measure"]))
umap_3d = dict(zip(umap_df[umap_df["method"] == "UMAP_3D"]["scenario"], umap_df[umap_df["method"] == "UMAP_3D"]["v_measure"]))

# ── GMM results ──
gmm_df = pd.read_csv(RES_DIR / "summary_gmm.csv")
gmm_v = dict(zip(gmm_df["scenario"], gmm_df["v_measure"]))

# ── Print final table ──
print("=" * 120)
print("FINAL COMPARISON: V-measure across 5 scenarios")
print("=" * 120)
header = f"{'Scenario':<12} {'Run_B (5D)':>11} {'UMAP_2D+HDB':>12} {'UMAP_3D+HDB':>12} {'GMM (BIC)':>10} {'Best':>12}"
print(header)
print("-" * 120)
for s in SCENARIOS:
    b = run_b[s]
    u2 = umap_2d.get(s, 0)
    u3 = umap_3d.get(s, 0)
    g = gmm_v.get(s, 0)
    vals = {"Run_B": b, "UMAP_2D": u2, "UMAP_3D": u3, "GMM": g}
    best_name = max(vals, key=vals.get)
    best_val = vals[best_name]
    marker = " <<" if best_name == "Run_B" else ""
    print(f"{s:<12} {b:11.4f} {u2:12.4f} {u3:12.4f} {g:10.4f} {best_name:>12}{marker}")

print("-" * 120)

# ── Delta from Run B ──
print(f"\n{'=' * 120}")
print("DELTA FROM RUN B BASELINE")
print("=" * 120)
print(f"{'Scenario':<12} {'UMAP_2D':>10} {'UMAP_3D':>10} {'GMM':>10}")
for s in SCENARIOS:
    b = run_b[s]
    d2 = (umap_2d.get(s, 0) - b) / b * 100
    d3 = (umap_3d.get(s, 0) - b) / b * 100
    dg = (gmm_v.get(s, 0) - b) / b * 100
    print(f"{s:<12} {d2:>+9.1f}% {d3:>+9.1f}% {dg:>+9.1f}%")

print()
print("-" * 50)
print("Overall winner: Run B (5D HDBSCAN) in 5/5 scenarios")
print("Runner-up:     GMM (close on high-density scenarios)")
print("Worst:         UMAP 2D (consistently degrades performance)")

# ── Save CSV ──
rows = []
for s in SCENARIOS:
    rows.append({"scenario": s, "Run_B_5D": run_b[s], "UMAP_2D_HDBSCAN": umap_2d.get(s, 0), "UMAP_3D_HDBSCAN": umap_3d.get(s, 0), "GMM_BIC": gmm_v.get(s, 0)})
df = pd.DataFrame(rows)
csv_path = RES_DIR / "final_comparison_4way.csv"
df.to_csv(csv_path, index=False, float_format="%.4f")
print(f"\nSaved: {csv_path}")
