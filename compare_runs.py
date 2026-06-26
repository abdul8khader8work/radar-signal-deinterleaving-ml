"""
compare_runs.py — Compare Run A (baseline), Run B (5 normalized), and Run C (13 features)

Produces:
  - Comparison table (stdout + CSV)
  - Grouped bar chart (V-measure, ARI, noise ratio across 5 scenarios)
"""

import json
import csv
from pathlib import Path

BASE = Path(__file__).parent.resolve()

# ── Run A: Baseline (no normalization) — from session history ──
RUN_A_BEST = {
    "stare_low":      {"v_measure": 0.27, "ari": 0.20, "noise_ratio": 0.35},
    "stare_high":     {"v_measure": 0.29, "ari": 0.22, "noise_ratio": 0.32},
    "scan_low":       {"v_measure": 0.45, "ari": 0.38, "noise_ratio": 0.12},
    "scan_high":      {"v_measure": 0.59, "ari": 0.50, "noise_ratio": 0.08},
    "mixed":          {"v_measure": 0.35, "ari": 0.28, "noise_ratio": 0.20},
}

# ── Run B: 5 features, normalized per-scenario ──
RUN_B_PATH = BASE / "results_runB_backup" / "best_params.json"
with open(RUN_B_PATH) as f:
    RUN_B_BEST = json.load(f)

# ── Run C: 13 features (5 PDW + 8 PRI-derived), normalized ──
RUN_C_PATH = BASE / "results" / "best_params.json"
with open(RUN_C_PATH) as f:
    RUN_C_BEST = json.load(f)

SCENARIOS = ["stare_low", "stare_high", "scan_low", "scan_high", "mixed"]

# ── Build comparison data ──
rows = []
for s in SCENARIOS:
    a = RUN_A_BEST[s]
    b = RUN_B_BEST[s]
    c = RUN_C_BEST[s]
    rows.append({
        "scenario": s,
        "A_v": a["v_measure"],
        "B_v": b["v_measure"],
        "C_v": c["v_measure"],
        "A_ari": a["ari"],
        "B_ari": b["ari"],
        "C_ari": c["ari"],
        "A_noise": a["noise_ratio"],
        "B_noise": b["noise_ratio"],
        "C_noise": c["noise_ratio"],
        "B_param": b["param_label"],
        "C_param": c["param_label"],
    })

# ── Print table ──
print("=" * 130)
print(f"{'Scenario':<12} {'V-measure':>30} {'ARI':>30} {'Noise%':>30}")
print(f"{'':<12} {'A':>8} {'B':>8} {'C':>8} {'A':>8} {'B':>8} {'C':>8} {'A':>8} {'B':>8} {'C':>8}")
print("-" * 130)
for r in rows:
    print(f"{r['scenario']:<12} "
          f"{r['A_v']:8.3f} {r['B_v']:8.3f} {r['C_v']:8.3f}  "
          f"{r['A_ari']:8.3f} {r['B_ari']:8.3f} {r['C_ari']:8.3f}  "
          f"{r['A_noise']*100:8.1f} {r['B_noise']*100:8.1f} {r['C_noise']*100:8.1f}")
print("=" * 130)
print(f"{'':<12}  {'B best param':<22} {'C best param':<22}")
for r in rows:
    print(f"  {r['scenario']:<12} {r['B_param']:<22} {r['C_param']:<22}")

# ── Save CSV ──
out_path = BASE / "results" / "run_comparison.csv"
with open(out_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"\nComparison saved to: {out_path}")

# ── Summary ──
print("\n-- SUMMARY --")
print(f"  Run A: Baseline (no normalization), 5 PDW features")
print(f"  Run B: Per-scenario StandardScaler,  5 PDW features")
print(f"  Run C: Per-scenario StandardScaler, 13 features (5 PDW + 8 PRI-derived)")
print()
improvements = 0
degradations = 0
for r in rows:
    gain_b = (r["B_v"] - r["A_v"]) / r["A_v"] * 100
    gain_c = (r["C_v"] - r["B_v"]) / r["B_v"] * 100
    if gain_c > 0:
        improvements += 1
    else:
        degradations += 1
    print(f"  {r['scenario']:<12} A->B: {gain_b:+.1f}%  B->C: {gain_c:+.1f}%")
print(f"\n  B improved over A in {sum(1 for r in rows if r['B_v'] > r['A_v'])}/5 scenarios")
print(f"  C {'improved' if improvements > degradations else 'degraded'} over B in {improvements}/5 scenarios")
