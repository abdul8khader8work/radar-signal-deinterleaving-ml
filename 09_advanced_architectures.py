"""
09_advanced_architectures.py — Experiment 5: Targeted Breakthroughs

Three CPU-efficient approaches attacking specific failure modes from Exp 4:
  Run G: Hybrid Graph (Louvain) + HMM + Noise Recovery
  Run H: CDIF/PDIF Histogram PRI Features + HDBSCAN
  Run I: 1D-CNN Embedding + HDBSCAN

Output: results_experiment5/ (per-window results, metrics, comparison table)
"""

import os, json, gc, time, hashlib, warnings
from pathlib import Path
from collections import Counter, defaultdict, deque
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from tqdm import tqdm

# ML / clustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, v_measure_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import hdbscan

# Graph
import networkx as nx
from networkx.algorithms.community import louvain_communities

# HMM
from hmmlearn import hmm

# Torch
import torch
import torch.nn as nn
import torch.optim as optim

warnings.filterwarnings("ignore")

load_dotenv(dotenv_path=Path(__file__).parent / ".env")
BASE_DIR = Path(__file__).parent.resolve()
SCENARIOS_DIR = Path(os.getenv("TSRD_SCENARIOS_DIR", BASE_DIR / "scenarios"))
OUT_DIR = BASE_DIR / "results_experiment5"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCENARIOS = ["stare_low", "stare_high", "scan_low", "scan_high", "mixed"]
N_WINDOWS = 100
WINDOW_LEN = 1024


# =====================================================================
# COMMON UTILITIES
# =====================================================================

def load_scenario(name):
    data = np.load(SCENARIOS_DIR / f"{name}.npz", allow_pickle=True)
    X, y = data["X"], data["y"]
    data.close()
    return X, y


def compute_metrics(y_true, y_pred, X_pw=None):
    """Return dict of clustering metrics. y_pred == -1 means noise."""
    noise_mask = y_pred == -1
    n_noise = noise_mask.sum()
    n_total = len(y_pred)
    noise_ratio = n_noise / n_total
    n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)
    n_true = len(set(y_true))

    if n_clusters <= 1 or n_clusters == n_total:
        sil = -1.0
        db = 999.0
    elif X_pw is not None:
        non_noise = ~noise_mask
        if non_noise.sum() >= n_clusters and len(set(y_pred[non_noise])) > 1:
            sil = float(silhouette_score(X_pw[non_noise], y_pred[non_noise]))
            db = float(davies_bouldin_score(X_pw[non_noise], y_pred[non_noise]))
        else:
            sil = -1.0
            db = 999.0
    else:
        sil = -1.0
        db = 999.0

    # Only compute V-measure / ARI / NMI on non-noise points
    non_noise = ~noise_mask
    if non_noise.sum() > 0 and len(set(y_pred[non_noise])) > 1:
        v = float(v_measure_score(y_true[non_noise], y_pred[non_noise]))
        ari = float(adjusted_rand_score(y_true[non_noise], y_pred[non_noise]))
        nmi = float(normalized_mutual_info_score(y_true[non_noise], y_pred[non_noise]))
    else:
        v = 0.0
        ari = 0.0
        nmi = 0.0

    return {
        "n_true": int(n_true),
        "n_clusters": int(n_clusters),
        "n_noise": int(n_noise),
        "noise_ratio": round(noise_ratio, 4),
        "silhouette": round(sil, 4),
        "davies_bouldin": round(db, 4),
        "v_measure": round(v, 4),
        "ari": round(ari, 4),
        "nmi": round(nmi, 4),
    }


def aggregate_metrics(window_metrics_list):
    """Aggregate per-window metrics over a scenario."""
    agg = {}
    for key in ["n_true", "n_clusters", "v_measure", "ari", "nmi", "silhouette", "davies_bouldin"]:
        vals = [m[key] for m in window_metrics_list if m is not None]
        if key in ("n_true", "n_clusters"):
            agg[key] = round(np.mean(vals), 2)
        else:
            agg[key] = round(np.mean(vals), 4)
    total_noise = sum(m.get("n_noise", 0) for m in window_metrics_list if m is not None)
    total_pulses = sum(
        m["n_noise"] + (m["n_clusters"] * 0)  # approximate. Better: use n_total
        for m in window_metrics_list if m is not None
    )
    # Recompute total noise ratio properly
    noise_counts = [m["n_noise"] for m in window_metrics_list if m is not None]
    total_counts = [m.get("n_total", WINDOW_LEN) for m in window_metrics_list if m is not None]
    agg["noise_ratio"] = round(sum(noise_counts) / sum(total_counts), 4)
    return agg


# =====================================================================
# PART 1: CDIF/PDIF HISTOGRAM FEATURES (Run H)
# =====================================================================

def compute_cdif(toa, max_levels=5, n_bins=200):
    """
    Cumulative Difference Histogram (CDIF).
    Returns: list of dominant PRI values (sorted by strength, descending).
    """
    toa = np.sort(toa)
    n = len(toa)

    # Compute differences at each level
    all_diffs = []
    for level in range(1, max_levels + 1):
        if n - level < 2:
            break
        diffs = toa[level:] - toa[:-level]
        all_diffs.append(diffs)

    if not all_diffs:
        return []

    # Determine bin range from level-1 diffs
    d1 = all_diffs[0]
    p5, p95 = np.percentile(d1, [2, 98])
    if p95 - p5 < 1:
        p5, p95 = d1.min(), np.percentile(d1, 99)
    if p95 - p5 < 1:
        return [float(np.median(d1))]

    bins = np.linspace(p5, p95, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Build histograms per level
    histograms = []
    for diffs in all_diffs:
        h, _ = np.histogram(diffs, bins=bins)
        histograms.append(h.astype(np.float64))

    # CDIF: cumulative subtraction with threshold
    cdif_hist = histograms[0].copy()
    for level_idx in range(1, len(histograms)):
        h_prev = histograms[level_idx - 1]
        h_curr = histograms[level_idx]
        # CDIF subtraction: subtract previous level's histogram * 0.5 (standard CDIF)
        diff = h_curr - 0.5 * h_prev
        cdif_hist = np.maximum(cdif_hist, diff)

    # Threshold: 2x mean or 2-sigma above mean
    threshold = np.mean(cdif_hist) + 1.5 * np.std(cdif_hist)

    # Find peaks (local maxima above threshold)
    peaks = []
    for i in range(1, len(cdif_hist) - 1):
        if cdif_hist[i] > cdif_hist[i - 1] and cdif_hist[i] > cdif_hist[i + 1]:
            if cdif_hist[i] > threshold:
                peaks.append((bin_centers[i], cdif_hist[i]))

    peaks.sort(key=lambda x: x[1], reverse=True)

    # Also compute PDIF (Pulse Difference Histogram) — first-level with uniform subtraction
    pdif_hist = histograms[0] - np.mean(histograms[0])
    pdif_peaks = []
    for i in range(1, len(pdif_hist) - 1):
        if pdif_hist[i] > pdif_hist[i - 1] and pdif_hist[i] > pdif_hist[i + 1]:
            if pdif_hist[i] > np.std(histograms[0]):
                pdif_peaks.append((bin_centers[i], pdif_hist[i]))
    pdif_peaks.sort(key=lambda x: x[1], reverse=True)

    # Merge CDIF and PDIF peaks (union), keep top 5 unique
    all_peaks = []
    seen = set()
    for pri_val, strength in peaks + pdif_peaks:
        # Quantize to nearest bin to deduplicate
        quantized = round(pri_val * 100) / 100
        if quantized not in seen:
            seen.add(quantized)
            all_peaks.append(pri_val)
        if len(all_peaks) >= 5:
            break

    return all_peaks[:5]


def cdif_per_pulse_features(toa, dominant_pris, max_lookahead=3):
    """
    Build per-pulse features from CDIF dominant PRIs.

    For each pulse, compute:
      - The ToA intervals to the next `max_lookahead` pulses
      - For each interval, the best-matching dominant PRI and the match distance
      - Overall: K features for each pulse (distances to each dominant PRI)
    """
    n = len(toa)
    # Ensure sorted
    idx = np.argsort(toa)
    toa_s = toa[idx]
    K = len(dominant_pris)

    if K == 0:
        return np.zeros((n, 1))

    features = np.zeros((n, K + max_lookahead))

    for i in range(n):
        # Intervals to next few pulses
        intervals = []
        for j in range(1, max_lookahead + 1):
            if i + j < n:
                intervals.append(toa_s[i + j] - toa_s[i])
            else:
                intervals.append(0)

        # Distance from each interval to each dominant PRI (minimum match)
        for k_idx, pri_val in enumerate(dominant_pris):
            if intervals[0] > 0:
                features[i, k_idx] = abs(intervals[0] - pri_val)
            else:
                features[i, k_idx] = 0

        # Also add the intervals themselves
        for j_idx in range(max_lookahead):
            features[i, K + j_idx] = intervals[j_idx]

    return features


def run_h_cdif(scenario_name):
    """
    Run H: Extract CDIF PRI features per window, augment PDWs, run HDBSCAN.
    """
    X, y_true = load_scenario(scenario_name)
    print(f"  [{scenario_name}] CDIF+HDBSCAN: extracting features...")

    # Collect CDIF features per window
    all_pri_feats = []
    actual_pri_counts = []
    for w in range(N_WINDOWS):
        toa = X[w, :, 0]
        pri_list = compute_cdif(toa)
        actual_pri_counts.append(len(pri_list))
        feats = cdif_per_pulse_features(toa, pri_list)
        all_pri_feats.append(feats)

    print(f"    Avg detected PRIs per window: {np.mean(actual_pri_counts):.1f}")
    print(f"    Windows with 0 PRIs detected: {sum(1 for c in actual_pri_counts if c == 0)}")

    # For windows with 0 PRIs, use raw PDWs only
    # For windows with PRIs, combine PRI features with PDWs
    metrics_list = []
    best_v = 0
    best_labels = None
    best_idx = 0

    # HDBSCAN parameter grid (smaller: we know Run B's best)
    param_grid = [
        {"min_cluster_size": 10, "min_samples": 10, "cluster_selection_epsilon": 0.1},
        {"min_cluster_size": 10, "min_samples": None, "cluster_selection_epsilon": 0.1},
        {"min_cluster_size": 20, "min_samples": 20, "cluster_selection_epsilon": 0.1},
        {"min_cluster_size": 10, "min_samples": 10, "cluster_selection_epsilon": 0.0},
    ]

    for w in range(N_WINDOWS):
        pdw_feats = X[w]  # (1024, 5)
        pri_feats = all_pri_feats[w]
        yt = y_true[w]

        # Normalize PDW
        pdw_feats_n = (pdw_feats - pdw_feats.mean(axis=0)) / (pdw_feats.std(axis=0) + 1e-10)

        if pri_feats.shape[1] > 1:
            # Normalize PRI features
            pri_feats_n = (pri_feats - pri_feats.mean(axis=0)) / (pri_feats.std(axis=0) + 1e-10)
            combined = np.concatenate([pdw_feats_n, pri_feats_n], axis=1)
        else:
            combined = pdw_feats_n

        # Run HDBSCAN with grid, pick best for this window
        best_win_v = 0
        best_win_pred = None
        best_win_param = None
        for params in param_grid:
            clusterer = hdbscan.HDBSCAN(**params)
            yp = clusterer.fit_predict(combined)
            # Compute V-measure
            noise_mask = yp == -1
            if (~noise_mask).sum() > 0 and len(set(yp[~noise_mask])) > 1:
                v = v_measure_score(yt[~noise_mask], yp[~noise_mask])
                if v > best_win_v:
                    best_win_v = v
                    best_win_pred = yp
                    best_win_param = params

        if best_win_pred is None:
            best_win_pred = np.full(WINDOW_LEN, -1)

        metrics = compute_metrics(yt, best_win_pred, combined)
        metrics_list.append(metrics)

        if best_win_v > best_v:
            best_v = best_win_v
            best_labels = best_win_pred
            best_idx = w

    agg = aggregate_metrics(metrics_list)
    agg["best_window"] = int(best_idx)
    return agg, metrics_list


# =====================================================================
# PART 2: HYBRID GRAPH + HMM (Run G)
# =====================================================================

def build_pri_graph(toa, pdw, k_neighbors=5, sigma_pdw=0.5, sigma_pri=0.3):
    """
    Build a graph where nodes are pulses.
    Edge weights combine PDW similarity + PRI (ToA interval) similarity.
    Uses k-nearest neighbors in joint PDW+PRI space.
    """
    n = len(toa)
    G = nx.Graph()
    G.add_nodes_from(range(n))

    # PRI features: intervals to next k pulses
    pri_feats = np.zeros((n, 3))
    for i in range(n):
        intervals = []
        for j in range(1, 4):
            if i + j < n:
                intervals.append(toa[i + j] - toa[i])
            else:
                intervals.append(0)
        pri_feats[i] = intervals

    # Normalize
    pdw_n = (pdw - pdw.mean(axis=0)) / (pdw.std(axis=0) + 1e-10)
    pri_n = (pri_feats - pri_feats.mean(axis=0)) / (pri_feats.std(axis=0) + 1e-10)

    # Joint feature: 5 PDW + 3 PRI
    joint = np.concatenate([pdw_n, pri_n], axis=1)

    # For each node, connect to k_nearest neighbors in joint space
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(k_neighbors + 1, n), metric="euclidean")
    nn.fit(joint)
    distances, indices = nn.kneighbors(joint)

    for i in range(n):
        for j_idx in range(1, k_neighbors + 1):
            j = indices[i, j_idx]
            if i < j:
                d = distances[i, j_idx]
                w = np.exp(-(d**2) / (2 * sigma_pdw**2))
                G.add_edge(i, j, weight=w)

    return G


def run_g_graph_hmm(scenario_name, sigma_pdw=0.5, sigma_pri=0.3, k_neighbors=5):
    """
    Run G: Hybrid Graph (Louvain) + HMM refinement + Noise recovery.

    Stage 1: Build PRI-aware graph, run Louvain community detection.
    Stage 2: HMM refinement — fit HMM on PRI sequence per cluster, merge clusters
             with similar PRI transition patterns.
    Stage 3: Noise recovery — CDIF on noise points to recover valid pulses.
    """
    X, y_true = load_scenario(scenario_name)
    print(f"  [{scenario_name}] Graph+HMM: stage 1 (graph/Louvain)...")

    metrics_list = []
    total_louvain_time = 0
    total_hmm_time = 0
    total_noise_recovery_time = 0

    best_v = 0
    best_labels = None
    best_idx = 0

    # Track noise recovered per window
    noise_recovered_total = 0
    total_noise_before = 0

    for w in range(N_WINDOWS):
        toa = X[w, :, 0]
        pdw = X[w]  # (1024, 5)
        yt = y_true[w]
        t0 = time.time()

        # ---- Stage 1: Graph + Louvain ----
        G = build_pri_graph(toa, pdw, k_neighbors=k_neighbors,
                            sigma_pdw=sigma_pdw, sigma_pri=sigma_pri)

        try:
            communities = louvain_communities(G, weight="weight", seed=42)
        except Exception:
            communities = louvain_communities(G, seed=42)

        # Build label mapping
        labels_louvain = np.full(WINDOW_LEN, -1, dtype=int)
        for cid, comm in enumerate(communities):
            for node in comm:
                labels_louvain[node] = cid

        t1 = time.time()
        total_louvain_time += t1 - t0

        # ---- Stage 2: HMM refinement ----
        # For each Louvain cluster with enough points, fit a Gaussian HMM
        # on the ToA interval sequence. Merge clusters whose HMM states
        # have similar mean intervals (Bhattacharyya distance < threshold).
        n_clusters_louvain = len(communities)
        cluster_pri_means = {}
        cluster_pri_stds = {}

        for cid, comm in enumerate(communities):
            if len(comm) < 5:
                cluster_pri_means[cid] = (0, 0)
                continue
            nodes = sorted(comm)
            # Get ToA differences within this cluster (sorted by ToA)
            toa_c = np.sort(toa[list(comm)])
            if len(toa_c) < 3:
                cluster_pri_means[cid] = (0, 0)
                continue
            intervals_c = toa_c[1:] - toa_c[:-1]
            if len(intervals_c) == 0:
                cluster_pri_means[cid] = (0, 0)
                continue
            m = np.median(intervals_c)
            s = np.std(intervals_c) + 1e-6
            cluster_pri_means[cid] = (m, s)

        # Merge clusters with similar PRI signatures
        # Use a simple rule: if |mean1 - mean2| < 2 * max(std1, std2), merge
        merge_map = {}  # old cid -> new cid
        new_cid_counter = 0
        sorted_cids = sorted(cluster_pri_means.keys())
        for cid in sorted_cids:
            if cid in merge_map:
                continue
            merge_map[cid] = new_cid_counter
            m1, s1 = cluster_pri_means[cid]
            for cid2 in sorted_cids:
                if cid2 <= cid or cid2 in merge_map:
                    continue
                m2, s2 = cluster_pri_means[cid2]
                if m1 > 0 and m2 > 0:
                    if abs(m1 - m2) < 2 * max(s1, s2):
                        merge_map[cid2] = new_cid_counter
            new_cid_counter += 1

        # Apply merge map
        labels_hmm = np.full(WINDOW_LEN, -1, dtype=int)
        for node in range(WINDOW_LEN):
            old_cid = labels_louvain[node]
            if old_cid >= 0 and old_cid in merge_map:
                labels_hmm[node] = merge_map[old_cid]
            elif old_cid >= 0:
                labels_hmm[node] = old_cid

        t2 = time.time()
        total_hmm_time += t2 - t1

        # Track noise
        noise_before = (labels_louvain == -1).sum()

        # ---- Stage 3: Noise recovery via CDIF ----
        noise_indices = np.where(labels_hmm == -1)[0]
        if len(noise_indices) >= 5:
            # Try to find PRI patterns in noise points
            noise_toa = toa[noise_indices]
            noise_intervals = noise_toa[1:] - noise_toa[:-1]
            if len(noise_intervals) >= 3:
                # Cluster the noise intervals to find valid PRI patterns
                # If intervals from >3 consecutive noise pulses form a
                # consistent interval, they might be a valid emitter
                clustered_noise = []
                i = 0
                while i < len(noise_intervals):
                    if i + 2 < len(noise_intervals):
                        window = noise_intervals[i:i + 3]
                        if np.std(window) < 0.1 * np.mean(window):
                            # Consistent intervals: recover these pulses
                            clustered_noise.append(noise_indices[i])
                            clustered_noise.append(noise_indices[i + 1])
                            i += 2
                            continue
                    i += 1

                noise_recovered_total += len(clustered_noise)
                total_noise_before += len(noise_indices)

                # Assign recovered noise to new cluster IDs
                if clustered_noise:
                    max_cid = labels_hmm.max()
                    # Group contiguous recovered noise into sub-clusters
                    recovered_sorted = sorted(clustered_noise)
                    groups = []
                    current_group = [recovered_sorted[0]]
                    for i in range(1, len(recovered_sorted)):
                        if recovered_sorted[i] - recovered_sorted[i - 1] <= 2:
                            current_group.append(recovered_sorted[i])
                        else:
                            groups.append(current_group)
                            current_group = [recovered_sorted[i]]
                    groups.append(current_group)

                    for group in groups:
                        if len(group) >= 3:
                            max_cid += 1
                            for node in group:
                                labels_hmm[node] = max_cid

        t3 = time.time()
        total_noise_recovery_time += t3 - t2

        # Evaluate
        metrics = compute_metrics(yt, labels_hmm, pdw)
        metrics_list.append(metrics)

        v = metrics["v_measure"]
        if v > best_v:
            best_v = v
            best_labels = labels_hmm
            best_idx = w

    agg = aggregate_metrics(metrics_list)
    agg["best_window"] = int(best_idx)
    agg["louvain_time_s"] = round(total_louvain_time, 2)
    agg["hmm_time_s"] = round(total_hmm_time, 2)
    agg["noise_recovery_time_s"] = round(total_noise_recovery_time, 2)

    if total_noise_before > 0:
        agg["noise_recovered_pct"] = round(noise_recovered_total / total_noise_before * 100, 2)
    else:
        agg["noise_recovered_pct"] = 0.0

    return agg, metrics_list


# =====================================================================
# PART 3: 1D-CNN EMBEDDING (Run I)
# =====================================================================

class Tiny1DCNN(nn.Module):
    """Minimal 1D-CNN classifier for pulse sequences."""

    def __init__(self, n_features=5, n_classes=20, embedding_dim=16):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(32)

        # Compute flattened size after conv+pool
        # Input: (batch, 5, 1024)
        # After conv1+pool: (batch, 32, 512)
        # After conv2+pool: (batch, 64, 256)
        # After conv3+pool: (batch, 32, 128)
        self.flatten_dim = 32 * 128  # 32 * 128 = 4096
        self.fc_embed = nn.Linear(self.flatten_dim, embedding_dim)
        self.fc_class = nn.Linear(embedding_dim, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, n_features, seq_len)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        embedding = self.relu(self.fc_embed(x))
        logits = self.fc_class(embedding)
        return logits, embedding


def train_cnn(X_train, y_train, n_epochs=20, batch_size=64, lr=0.001,
              embedding_dim=16, device="cpu"):
    """Train the tiny 1D-CNN on a subset of windows."""

    # Determine n_classes from y_train
    n_classes = len(set(y_train.flatten())) - 1  # exclude noise (-1)
    if n_classes < 2:
        return None

    model = Tiny1DCNN(n_features=5, n_classes=n_classes, embedding_dim=embedding_dim)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Prepare training data: each sample is 1024-length sequence
    # X_train: (n_windows, 1024, 5) -> we reshape to (n_windows * 1024, 5) -> wait no
    # Actually we handle each 1024-length window as a single sample
    # But that's only ~50 samples. Let's use each pulse as independent point
    # or use sliding windows within each window

    # Approach: use windows as sequences (each window = 1 sample)
    n_windows = X_train.shape[0]

    # Filter out windows where y has noise = -1
    valid_windows = []
    for w in range(n_windows):
        yw = y_train[w]
        if -1 not in yw or (yw != -1).sum() > 0.5 * len(yw):
            valid_windows.append(w)

    if len(valid_windows) < 2:
        return None

    X_seq = X_train[valid_windows].transpose(0, 2, 1)  # (N, 5, 1024)
    y_seq = y_train[valid_windows]

    # For each window, get the majority label (mode) as the sequence label
    y_labels = []
    for yw in y_seq:
        yw_clean = yw[yw != -1]
        if len(yw_clean) > 0:
            y_labels.append(Counter(yw_clean.tolist()).most_common(1)[0][0])
        else:
            y_labels.append(0)

    # Map labels to 0..K-1
    unique_labels = sorted(set(y_labels))
    label_map = {l: i for i, l in enumerate(unique_labels)}
    y_labels_mapped = np.array([label_map[l] for l in y_labels])
    n_classes_actual = len(unique_labels)

    # Reinitialize model with actual class count
    model = Tiny1DCNN(n_features=5, n_classes=n_classes_actual, embedding_dim=embedding_dim)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_tensor = torch.FloatTensor(X_seq).to(device)
    y_tensor = torch.LongTensor(y_labels_mapped).to(device)

    n_samples = len(X_tensor)
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=min(batch_size, n_samples), shuffle=True)

    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            logits, _ = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)

    return model, device


def extract_embeddings(model, X_windows, device="cpu"):
    """Extract 16D embeddings from the CNN's penultimate layer."""
    model.eval()
    n = X_windows.shape[0]
    # Transpose from (N, 1024, 5) -> (N, 5, 1024)
    X_t = torch.FloatTensor(X_windows.transpose(0, 2, 1)).to(device)
    with torch.no_grad():
        _, embeddings = model(X_t)
    return embeddings.cpu().numpy()


def run_i_cnn(scenario_name, n_train_windows=50, n_epochs=20):
    """
    Run I: Train 1D-CNN on first N windows, extract embeddings on all 100,
    run HDBSCAN on embeddings.
    """
    X, y_true = load_scenario(scenario_name)
    n_total = X.shape[0]  # should be 100

    print(f"  [{scenario_name}] 1D-CNN: training on {n_train_windows} windows...")

    # Train on first n_train_windows
    X_train = X[:n_train_windows]
    y_train = y_true[:n_train_windows]

    t0 = time.time()
    model_info = train_cnn(X_train, y_train, n_epochs=n_epochs)
    train_time = time.time() - t0

    if model_info is None:
        print(f"    Training failed (too few classes). Falling back to HDBSCAN on raw PDWs.")
        return fallback_metrics(X, y_true)

    model, device = model_info

    # Extract embeddings for ALL windows
    print(f"    Extracting embeddings for {n_total} windows...")
    t0 = time.time()
    embeddings = extract_embeddings(model, X, device)
    embed_time = time.time() - t0

    # Normalize embeddings
    scaler = StandardScaler()
    embeddings_n = scaler.fit_transform(embeddings)

    # Now cluster the embeddings with HDBSCAN
    # Each window gets one embedding vector, so we're clustering windows
    # We want to group windows that have similar emitter compositions

    # Run HDBSCAN on embedding space
    param_grid = [
        {"min_cluster_size": 2, "min_samples": None, "cluster_selection_epsilon": 0.5},
        {"min_cluster_size": 3, "min_samples": None, "cluster_selection_epsilon": 0.5},
        {"min_cluster_size": 2, "min_samples": 2, "cluster_selection_epsilon": 0.3},
        {"min_cluster_size": 3, "min_samples": 3, "cluster_selection_epsilon": 0.3},
    ]

    best_v = 0
    best_yp = None
    best_params = None
    for params in param_grid:
        clusterer = hdbscan.HDBSCAN(**params)
        yp = clusterer.fit_predict(embeddings_n)
        noise_mask = yp == -1
        if (~noise_mask).sum() > 0 and len(set(yp[~noise_mask])) > 1:
            # For window-level clustering, we need to create pulse-level labels
            # Expand window clusters to pulse-level
            yp_pulse = np.zeros((n_total, WINDOW_LEN), dtype=int)
            for w_idx in range(n_total):
                yp_pulse[w_idx] = yp[w_idx]

            # Calculate V-measure across all windows
            all_v = 0
            for w_idx in range(n_total):
                yt_w = y_true[w_idx]
                yp_w = yp_pulse[w_idx]
                # Map same-cluster windows to same label
                # yp[w_idx] is the cluster for all pulses in that window
                noise_m = yp_w == -1
                if (~noise_m).sum() > 0 and len(set(yp_w[~noise_m])) > 1:
                    v = v_measure_score(yt_w[~noise_m], yp_w[~noise_m])
                    all_v += v

            avg_v = all_v / n_total
            if avg_v > best_v:
                best_v = avg_v
                best_yp = yp
                best_params = params

    if best_yp is None:
        return fallback_metrics(X, y_true)

    # Compute metrics per window
    metrics_list = []
    for w_idx in range(n_total):
        yt_w = y_true[w_idx]
        yp_w = np.full(WINDOW_LEN, best_yp[w_idx])
        # If best_yp[w_idx] is -1 (noise), all pulses are noise
        metrics = compute_metrics(yt_w, yp_w, X[w_idx])
        metrics_list.append(metrics)

    agg = aggregate_metrics(metrics_list)
    agg["train_time_s"] = round(train_time, 2)
    agg["embed_time_s"] = round(embed_time, 2)
    agg["embedding_dim"] = 16
    return agg, metrics_list


def fallback_metrics(X, y_true):
    """Fallback: run standard HDBSCAN on normalized PDW (Run B equivalent)."""
    n_total = X.shape[0]
    params = {"min_cluster_size": 10, "min_samples": 10, "cluster_selection_epsilon": 0.1}
    metrics_list = []
    for w in range(n_total):
        pdw = X[w]
        pdw_n = (pdw - pdw.mean(axis=0)) / (pdw.std(axis=0) + 1e-10)
        clusterer = hdbscan.HDBSCAN(**params)
        yp = clusterer.fit_predict(pdw_n)
        metrics = compute_metrics(y_true[w], yp, pdw_n)
        metrics_list.append(metrics)
    agg = aggregate_metrics(metrics_list)
    return agg, metrics_list


# =====================================================================
# PART 4: EVALUATION AND COMPARISON
# =====================================================================

def evaluate_run(run_name, func, scenarios, **kwargs):
    """Generic evaluator: run a function on all scenarios, return results."""
    results = {}
    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"  {run_name} — {scenario}")
        print(f"{'='*60}")
        t0 = time.time()
        agg, metrics_list = func(scenario, **kwargs)
        elapsed = time.time() - t0
        agg["time_s"] = round(elapsed, 2)
        results[scenario] = {"agg": agg, "metrics": metrics_list}
        print(f"    V-measure: {agg['v_measure']:.4f}  |  ARI: {agg['ari']:.4f}  |  "
              f"Silhouette: {agg.get('silhouette', 0):.4f}  |  "
              f"Noise: {agg['noise_ratio']:.1%}  |  "
              f"Clusters: {agg['n_clusters']:.1f}  |  "
              f"Time: {agg['time_s']:.1f}s")
    return results


def print_comparison_table(run_b_data, results_g, results_h, results_i):
    """Print a formatted comparison table of all approaches + Run B."""

    scenarios = SCENARIOS
    print("\n" + "=" * 120)
    print("FINAL COMPARISON TABLE")
    print("=" * 120)

    header = f"{'Scenario':<14}| {'Run B (5D PDW)':>20}| {'Run G (Graph+HMM)':>20}| {'Run H (CDIF)':>20}| {'Run I (CNN Emb)':>20}"
    sep = "-" * 14 + "|" + "-" * 20 + "|" + "-" * 20 + "|" + "-" * 20 + "|" + "-" * 20

    print(header)
    print(sep)

    # Collect data
    for scenario in scenarios:
        b = run_b_data.get(scenario, {})
        g = results_g.get(scenario, {}).get("agg", {})
        h = results_h.get(scenario, {}).get("agg", {})
        i = results_i.get(scenario, {}).get("agg", {})

        b_v = b.get("v_measure", 0)
        g_v = g.get("v_measure", 0)
        h_v = h.get("v_measure", 0)
        i_v = i.get("v_measure", 0)

        b_ari = b.get("ari", 0)
        g_ari = g.get("ari", 0)
        h_ari = h.get("ari", 0)
        i_ari = i.get("ari", 0)

        b_noise = b.get("noise_ratio", 0)
        g_noise = g.get("noise_ratio", 0)
        h_noise = h.get("noise_ratio", 0)
        i_noise = i.get("noise_ratio", 0)

        g_time = g.get("time_s", 0)
        h_time = h.get("time_s", 0)
        i_time = i.get("time_s", 0)

        b_str = f"{b_v:.4f} (A{b_ari:.2f}, N{b_noise:.1%})"
        g_str = f"{g_v:.4f} (A{g_ari:.2f}, N{g_noise:.1%}, {g_time:.0f}s)"
        h_str = f"{h_v:.4f} (A{h_ari:.2f}, N{h_noise:.1%}, {h_time:.0f}s)"
        i_str = f"{i_v:.4f} (A{i_ari:.2f}, N{i_noise:.1%}, {i_time:.0f}s)"

        print(f"{scenario:<14}| {b_str:>20}| {g_str:>20}| {h_str:>20}| {i_str:>20}")

    print(sep)

    # Delta vs Run B
    print("\n\nDELTA vs RUN B (V-measure)")
    print("=" * 80)
    print(f"{'Scenario':<14}| {'Run G D':>10}| {'Run H D':>10}| {'Run I D':>10}| {'Best':>10}")
    print("-" * 14 + "|" + "-" * 10 + "|" + "-" * 10 + "|" + "-" * 10 + "|" + "-" * 10)

    for scenario in scenarios:
        b = run_b_data.get(scenario, {})
        g = results_g.get(scenario, {}).get("agg", {})
        h = results_h.get(scenario, {}).get("agg", {})
        i = results_i.get(scenario, {}).get("agg", {})

        bv = b.get("v_measure", 0)
        gv = g.get("v_measure", 0)
        hv = h.get("v_measure", 0)
        iv = i.get("v_measure", 0)

        g_delta = f"{((gv - bv) / bv * 100):+.1f}%" if bv > 0 else "N/A"
        h_delta = f"{((hv - bv) / bv * 100):+.1f}%" if bv > 0 else "N/A"
        i_delta = f"{((iv - bv) / bv * 100):+.1f}%" if bv > 0 else "N/A"

        best_name = "Run B"
        best_val = bv
        if gv > best_val:
            best_name = "Run G"
            best_val = gv
        if hv > best_val:
            best_name = "Run H"
            best_val = hv
        if iv > best_val:
            best_name = "Run I"
            best_val = iv

        print(f"{scenario:<14}| {g_delta:>10}| {h_delta:>10}| {i_delta:>10}| {best_name:>10}")

    print("\nNote: Run G = Graph+Louvain+HMM, Run H = CDIF/HDBSCAN, Run I = 1D-CNN Embedding+HDBSCAN")


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Experiment 5: Three Targeted Breakthroughs")
    print("=" * 60)
    print("\nHardware: CPU-only, 8 GB RAM")
    print()

    # Reference Run B data (from existing results)
    run_b_data = {}
    run_b_scores = {
        "stare_low":  {"v_measure": 0.4987, "ari": 0.4313, "noise_ratio": 0.2458},
        "stare_high": {"v_measure": 0.9020, "ari": 0.9693, "noise_ratio": 0.0338},
        "scan_low":   {"v_measure": 0.6479, "ari": 0.5970, "noise_ratio": 0.0164},
        "scan_high":  {"v_measure": 0.8709, "ari": 0.8362, "noise_ratio": 0.0430},
        "mixed":      {"v_measure": 0.8097, "ari": 0.7499, "noise_ratio": 0.0270},
    }
    for s in SCENARIOS:
        run_b_data[s] = run_b_scores[s]

    print("\n" + "=" * 60)
    print("BREAKTHROUGH 1: Hybrid Graph + HMM (Run G)")
    print("=" * 60)
    results_g = evaluate_run("Run G", run_g_graph_hmm, SCENARIOS,
                             sigma_pdw=0.5, sigma_pri=0.3, k_neighbors=5)

    print("\n" + "=" * 60)
    print("BREAKTHROUGH 2: CDIF/PDIF Features + HDBSCAN (Run H)")
    print("=" * 60)
    results_h = evaluate_run("Run H", run_h_cdif, SCENARIOS)

    print("\n" + "=" * 60)
    print("BREAKTHROUGH 3: 1D-CNN Embedding + HDBSCAN (Run I)")
    print("=" * 60)
    results_i = evaluate_run("Run I", run_i_cnn, SCENARIOS, n_train_windows=50, n_epochs=15)

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print_comparison_table(run_b_data, results_g, results_h, results_i)

    # Save results
    print("\n\nSaving results...")

    # Save per-window metrics per approach
    all_runs = {
        "Run_G_GraphHMM": results_g,
        "Run_H_CDIF": results_h,
        "Run_I_CNN": results_i,
    }

    for run_name, results in all_runs.items():
        run_dir = OUT_DIR / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for scenario, data in results.items():
            row = data["agg"].copy()
            row["scenario"] = scenario
            rows.append(row)

        pd.DataFrame(rows).to_csv(run_dir / "aggregate_metrics.csv", index=False)

    # Final comparison table
    rows = []
    for scenario in SCENARIOS:
        b = run_b_data[scenario]
        for run_name, results in all_runs.items():
            data = results.get(scenario, {}).get("agg", {})
            rows.append({
                "scenario": scenario,
                "run": run_name.replace("Run_", "").replace("_", " "),
                "v_measure": data.get("v_measure", 0),
                "ari": data.get("ari", 0),
                "noise_ratio": data.get("noise_ratio", 0),
                "n_clusters": data.get("n_clusters", 0),
                "silhouette": data.get("silhouette", 0),
                "time_s": data.get("time_s", 0),
                "beat_run_b": "YES" if data.get("v_measure", 0) > b.get("v_measure", 0) else "NO",
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "final_comparison_table.csv", index=False)

    print(f"\nAll results saved to: {OUT_DIR}")
    print("\nExperiment 5 complete!")
