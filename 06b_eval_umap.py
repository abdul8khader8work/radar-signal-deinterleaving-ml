"""Generate UMAP comparison summary from cached results"""
import json, csv
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import v_measure_score, adjusted_rand_score, homogeneity_score, completeness_score, adjusted_mutual_info_score

BASE = Path(__file__).parent.resolve()
RES_DIR = BASE / "results_experiment2"
SCEN_DIR = BASE / "scenarios"

SCENARIOS = ["stare_low", "stare_high", "scan_low", "scan_high", "mixed"]

RUN_B = {"stare_low": 0.4987, "stare_high": 0.9020, "scan_low": 0.6479, "scan_high": 0.8709, "mixed": 0.8097}

def evaluate(y_true, y_pred):
    y_pred = np.array(y_pred)
    u = np.unique(y_pred)
    if len(u) <= 1:
        return {"v_measure": 0.0, "ari": 0.0, "noise_ratio": float((y_pred == -1).mean())}
    try:
        return {"v_measure": v_measure_score(y_true, y_pred), "ari": adjusted_rand_score(y_true, y_pred), "noise_ratio": float((y_pred == -1).mean())}
    except:
        return {"v_measure": 0.0, "ari": 0.0, "noise_ratio": float((y_pred == -1).mean())}

all_rows = []
for s in SCENARIOS:
    data = np.load(SCEN_DIR / f"{s}.npz", allow_pickle=True)
    y_all = data["y"]
    data.close()
    for ndim in [2, 3]:
        for w in range(100):
            rp = RES_DIR / f"{s}_umap{ndim}d_w{w:04d}.json"
            if not rp.exists():
                continue
            with open(rp) as f:
                pred = json.load(f)
            m = evaluate(y_all[w], pred["labels"])
            m["scenario"] = s
            m["method"] = f"UMAP_{ndim}D"
            all_rows.append(m)

df = pd.DataFrame(all_rows)
summary = df.groupby(["scenario", "method"]).agg(v_measure=("v_measure", "mean"), ari=("ari", "mean"), noise_ratio=("noise_ratio", "mean"), n_windows=("scenario", "count")).reset_index()

print("=" * 120)
print(f"{'Scenario':<12} {'Method':<14} {'V-measure':>10} {'ARI':>10} {'Noise%':>8} {'Windows':>8}")
print("-" * 120)
for _, r in summary.iterrows():
    print(f"{r['scenario']:<12} {r['method']:<14} {r['v_measure']:10.4f} {r['ari']:10.4f} {r['noise_ratio']*100:7.1f}% {r['n_windows']:8.0f}")
print("-" * 120)
for s in SCENARIOS:
    print(f"{s:<12} {'Run_B_5D':<14} {RUN_B[s]:10.4f} {'':>10} {'':>8} {'':>8}")
print("=" * 120)

print()
print("=" * 80)
print("Delta from Run B baseline (V-measure)")
print("=" * 80)
print(f"{'Scenario':<12} {'UMAP_2D':>12} {'UMAP_3D':>12}")
for s in SCENARIOS:
    base = RUN_B[s]
    v2 = summary.loc[(summary["scenario"] == s) & (summary["method"] == "UMAP_2D"), "v_measure"].values
    v3 = summary.loc[(summary["scenario"] == s) & (summary["method"] == "UMAP_3D"), "v_measure"].values
    d2 = v2[0] - base if len(v2) else 0
    d3 = v3[0] - base if len(v3) else 0
    p2 = d2 / base * 100 if base else 0
    p3 = d3 / base * 100 if base else 0
    print(f"{s:<12} {d2:+10.4f} ({p2:+5.1f}%) {d3:+10.4f} ({p3:+5.1f}%)")

csv_path = RES_DIR / "summary_umap_hdbscan.csv"
summary.to_csv(csv_path, index=False, float_format="%.4f")
print(f"\nSaved: {csv_path}")
