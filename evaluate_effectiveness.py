import argparse
import csv
import json
from collections import deque
from datetime import datetime
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import davies_bouldin_score, silhouette_score

from src.clustering import cluster_objects
from src.config import (
    CLUSTER_EPS,
    CLUSTER_MIN_POINTS,
    ENABLE_INTENSITY_COMP,
    EWMA_ALPHA,
    FUSION_WINDOW,
    GROUND_THRESHOLD,
    MAX_RANGE,
    RANGE_ATTENUATION_ALPHA,
    USE_EWMA_FUSION,
    VOXEL_SIZE,
)
from src.lidar_loader import load_las
from src.object_features import extract_object_features
from src.pipeline import iter_lidar_frames
from src.preprocessing import compensate_intensity, filter_range, remove_ground
from src.reflectivity_map import build_reflectivity
from src.temporal_fusion import fuse_maps
from src.voxelization import voxelize


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure LiDAR pipeline effectiveness with unsupervised proxy metrics "
            "(no ground-truth labels required)."
        )
    )
    parser.add_argument(
        "--pattern",
        default="data/velodyne_points/las/*.las",
        help="Glob pattern for LAS frames.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of frames to evaluate (0 = all).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Initial frames to exclude from final aggregate metrics.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=4000,
        help="Max points sampled for silhouette metric per frame.",
    )
    parser.add_argument(
        "--out-dir",
        default="effectiveness_results",
        help="Directory to store results and plots.",
    )
    return parser.parse_args()


def _safe_mean(values):
    return float(mean(values)) if values else 0.0


def _cluster_compactness(clusters):
    if not clusters:
        return 0.0
    per_cluster = []
    for cluster in clusters:
        if cluster.shape[0] == 0:
            continue
        centroid = cluster.mean(axis=0, keepdims=True)
        distances = np.linalg.norm(cluster - centroid, axis=1)
        per_cluster.append(float(distances.mean()))
    return _safe_mean(per_cluster)


def _temporal_jaccard(current_map, previous_map):
    if previous_map is None:
        return np.nan
    curr_keys = set(current_map.keys())
    prev_keys = set(previous_map.keys())
    union = curr_keys | prev_keys
    if not union:
        return np.nan
    inter = curr_keys & prev_keys
    return float(len(inter) / len(union))


def _temporal_reflectivity_corr(current_map, previous_map):
    if previous_map is None:
        return np.nan
    shared_keys = set(current_map.keys()) & set(previous_map.keys())
    if len(shared_keys) < 2:
        return np.nan
    curr = np.array([current_map[k] for k in shared_keys], dtype=np.float32)
    prev = np.array([previous_map[k] for k in shared_keys], dtype=np.float32)
    if np.std(curr) < 1e-6 or np.std(prev) < 1e-6:
        return np.nan
    return float(np.corrcoef(curr, prev)[0, 1])


def _centroid_shift(curr_centroids, prev_centroids):
    if curr_centroids.size == 0 or prev_centroids.size == 0:
        return np.nan
    curr_xy = curr_centroids[:, :2]
    prev_xy = prev_centroids[:, :2]
    dists = np.linalg.norm(curr_xy[:, None, :] - prev_xy[None, :, :], axis=2)
    return float(dists.min(axis=1).mean())


def _compute_clustering_metrics(points, eps, min_samples, sample_size):
    if points.shape[0] < min_samples:
        return {
            "clusters_count": 0,
            "clustered_points": 0,
            "noise_ratio": 1.0 if points.shape[0] > 0 else 0.0,
            "cluster_coverage": 0.0,
            "silhouette": np.nan,
            "davies_bouldin": np.nan,
            "labels": np.full(points.shape[0], -1, dtype=np.int32),
        }

    features = np.column_stack((points[:, :2], points[:, 2] * 0.25))
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(features)
    valid = labels != -1
    clustered_points = int(valid.sum())
    n_points = points.shape[0]
    cluster_coverage = float(clustered_points / n_points) if n_points else 0.0
    noise_ratio = 1.0 - cluster_coverage

    unique_clusters = sorted([label for label in set(labels.tolist()) if label != -1])
    clusters_count = len(unique_clusters)

    silhouette = np.nan
    davies_bouldin = np.nan
    if clusters_count >= 2 and clustered_points >= max(min_samples, 50):
        cluster_feats = features[valid]
        cluster_labels = labels[valid]

        if cluster_feats.shape[0] > sample_size:
            rng = np.random.default_rng(seed=42)
            sample_idx = rng.choice(cluster_feats.shape[0], size=sample_size, replace=False)
            sample_feats = cluster_feats[sample_idx]
            sample_labels = cluster_labels[sample_idx]
        else:
            sample_feats = cluster_feats
            sample_labels = cluster_labels

        if len(set(sample_labels.tolist())) >= 2:
            try:
                silhouette = float(silhouette_score(sample_feats, sample_labels))
            except Exception:
                silhouette = np.nan

        try:
            davies_bouldin = float(davies_bouldin_score(cluster_feats, cluster_labels))
        except Exception:
            davies_bouldin = np.nan

    return {
        "clusters_count": clusters_count,
        "clustered_points": clustered_points,
        "noise_ratio": noise_ratio,
        "cluster_coverage": cluster_coverage,
        "silhouette": silhouette,
        "davies_bouldin": davies_bouldin,
        "labels": labels,
    }


def _heuristic_effectiveness(row):
    # Heuristic score [0, 100] for trend comparison when labels are unavailable.
    retention = np.clip(row["retention_ratio"], 0.0, 1.0)
    coverage = np.clip(row["cluster_coverage"], 0.0, 1.0)

    sil = row["silhouette"]
    sil_score = 0.5 if np.isnan(sil) else float(np.clip((sil + 1.0) / 2.0, 0.0, 1.0))

    jac = row["temporal_voxel_jaccard"]
    jac_score = 0.5 if np.isnan(jac) else float(np.clip(jac, 0.0, 1.0))

    dbi = row["davies_bouldin"]
    dbi_score = 0.5 if np.isnan(dbi) else float(np.clip(np.exp(-dbi / 3.0), 0.0, 1.0))

    centroid_shift = row["centroid_shift_xy_m"]
    shift_score = (
        0.5
        if np.isnan(centroid_shift)
        else float(np.clip(np.exp(-centroid_shift / 3.0), 0.0, 1.0))
    )

    weighted = (
        0.20 * retention
        + 0.20 * coverage
        + 0.20 * sil_score
        + 0.15 * jac_score
        + 0.15 * dbi_score
        + 0.10 * shift_score
    )
    return float(weighted * 100.0)


def evaluate_frame(file_path, map_history, previous_map, previous_centroids, sample_size):
    row = {"frame_file": file_path}

    points_raw, intensity = load_las(file_path)
    row["points_raw"] = int(points_raw.shape[0])

    if ENABLE_INTENSITY_COMP:
        intensity = compensate_intensity(points_raw, intensity, RANGE_ATTENUATION_ALPHA)

    points, intensity = filter_range(points_raw, intensity, MAX_RANGE)
    row["points_after_range"] = int(points.shape[0])

    points, intensity = remove_ground(points, intensity, GROUND_THRESHOLD)
    row["points_after_ground"] = int(points.shape[0])
    row["retention_ratio"] = (
        float(points.shape[0] / points_raw.shape[0]) if points_raw.shape[0] else 0.0
    )

    voxels, voxel_intensity, _ = voxelize(points, intensity, VOXEL_SIZE)
    reflectivity_map = build_reflectivity(voxels, voxel_intensity)
    map_history.append(reflectivity_map)
    fused_map, stability_map = fuse_maps(
        list(map_history),
        use_ewma=USE_EWMA_FUSION,
        ewma_alpha=EWMA_ALPHA,
    )

    clustering = _compute_clustering_metrics(
        points=points,
        eps=CLUSTER_EPS,
        min_samples=CLUSTER_MIN_POINTS,
        sample_size=sample_size,
    )
    row.update(
        {
            "clusters_count": clustering["clusters_count"],
            "clustered_points": clustering["clustered_points"],
            "cluster_coverage": clustering["cluster_coverage"],
            "noise_ratio": clustering["noise_ratio"],
            "silhouette": clustering["silhouette"],
            "davies_bouldin": clustering["davies_bouldin"],
        }
    )

    clusters = cluster_objects(points, CLUSTER_EPS, CLUSTER_MIN_POINTS)
    row["cluster_compactness_m"] = _cluster_compactness(clusters)

    object_features = extract_object_features(clusters, fused_map, stability_map, VOXEL_SIZE)
    row["objects_count"] = int(len(object_features))
    row["mean_object_height_m"] = (
        float(np.mean([feature["height"] for feature in object_features]))
        if object_features
        else 0.0
    )
    row["mean_object_footprint_m2"] = (
        float(np.mean([feature["footprint_area"] for feature in object_features]))
        if object_features
        else 0.0
    )
    row["mean_stability"] = _safe_mean(list(stability_map.values()))

    row["temporal_voxel_jaccard"] = _temporal_jaccard(reflectivity_map, previous_map)
    row["temporal_reflectivity_corr"] = _temporal_reflectivity_corr(
        reflectivity_map, previous_map
    )

    curr_centroids = (
        np.array([cluster.mean(axis=0) for cluster in clusters], dtype=np.float32)
        if clusters
        else np.empty((0, 3), dtype=np.float32)
    )
    row["centroid_shift_xy_m"] = _centroid_shift(curr_centroids, previous_centroids)
    row["effectiveness_score"] = _heuristic_effectiveness(row)

    return row, reflectivity_map, curr_centroids


def summarize(rows):
    if not rows:
        return {}

    def finite(values):
        return [value for value in values if not np.isnan(value)]

    summary = {
        "num_frames": len(rows),
        "mean_points_raw": _safe_mean([row["points_raw"] for row in rows]),
        "mean_points_after_ground": _safe_mean([row["points_after_ground"] for row in rows]),
        "mean_retention_ratio": _safe_mean([row["retention_ratio"] for row in rows]),
        "mean_cluster_coverage": _safe_mean([row["cluster_coverage"] for row in rows]),
        "mean_noise_ratio": _safe_mean([row["noise_ratio"] for row in rows]),
        "mean_clusters": _safe_mean([row["clusters_count"] for row in rows]),
        "mean_objects": _safe_mean([row["objects_count"] for row in rows]),
        "mean_cluster_compactness_m": _safe_mean(
            [row["cluster_compactness_m"] for row in rows]
        ),
        "mean_silhouette": _safe_mean(finite([row["silhouette"] for row in rows])),
        "mean_davies_bouldin": _safe_mean(finite([row["davies_bouldin"] for row in rows])),
        "mean_temporal_voxel_jaccard": _safe_mean(
            finite([row["temporal_voxel_jaccard"] for row in rows])
        ),
        "mean_temporal_reflectivity_corr": _safe_mean(
            finite([row["temporal_reflectivity_corr"] for row in rows])
        ),
        "mean_centroid_shift_xy_m": _safe_mean(
            finite([row["centroid_shift_xy_m"] for row in rows])
        ),
        "mean_effectiveness_score": _safe_mean(
            [row["effectiveness_score"] for row in rows]
        ),
    }
    return summary


def write_csv(rows, path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(payload, path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def plot(rows, out_dir):
    if not rows:
        return

    idx = np.arange(len(rows))
    retention = np.array([row["retention_ratio"] for row in rows], dtype=np.float32)
    coverage = np.array([row["cluster_coverage"] for row in rows], dtype=np.float32)
    score = np.array([row["effectiveness_score"] for row in rows], dtype=np.float32)
    silhouette = np.array([row["silhouette"] for row in rows], dtype=np.float32)
    dbi = np.array([row["davies_bouldin"] for row in rows], dtype=np.float32)
    jaccard = np.array([row["temporal_voxel_jaccard"] for row in rows], dtype=np.float32)
    shift = np.array([row["centroid_shift_xy_m"] for row in rows], dtype=np.float32)

    plt.style.use("seaborn-v0_8-whitegrid")

    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(idx, retention, label="retention_ratio", linewidth=2)
    ax1.plot(idx, coverage, label="cluster_coverage", linewidth=2)
    ax1.set_ylim(0.0, 1.0)
    ax1.set_title("Signal Retention and Cluster Coverage")
    ax1.set_xlabel("Frame index")
    ax1.set_ylabel("Ratio")
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(out_dir / "retention_coverage.png", dpi=150)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(idx, silhouette, label="silhouette (higher better)", linewidth=2)
    ax2.plot(idx, dbi, label="davies_bouldin (lower better)", linewidth=2)
    ax2.set_title("Clustering Quality")
    ax2.set_xlabel("Frame index")
    ax2.set_ylabel("Metric")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(out_dir / "clustering_quality.png", dpi=150)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(12, 5))
    ax3.plot(idx, jaccard, label="voxel_jaccard", linewidth=2)
    ax3.plot(idx, shift, label="centroid_shift_xy_m", linewidth=2)
    ax3.set_title("Temporal Stability")
    ax3.set_xlabel("Frame index")
    ax3.set_ylabel("Metric")
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(out_dir / "temporal_stability.png", dpi=150)
    plt.close(fig3)

    fig4, ax4 = plt.subplots(figsize=(12, 5))
    ax4.plot(idx, score, color="#0b5d7a", linewidth=2)
    ax4.set_title("Heuristic Effectiveness Score (0-100)")
    ax4.set_xlabel("Frame index")
    ax4.set_ylabel("Score")
    fig4.tight_layout()
    fig4.savefig(out_dir / "effectiveness_score.png", dpi=150)
    plt.close(fig4)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = iter_lidar_frames(args.pattern)
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        raise FileNotFoundError(f"No files found with pattern: {args.pattern}")

    print(f"Found {len(files)} frame(s) for evaluation.")
    print(f"Warmup excluded from summary: {args.warmup}")

    map_history = deque(maxlen=max(FUSION_WINDOW, 1))
    previous_map = None
    previous_centroids = np.empty((0, 3), dtype=np.float32)
    all_rows = []

    for idx, file_path in enumerate(files):
        row, previous_map, previous_centroids = evaluate_frame(
            file_path=file_path,
            map_history=map_history,
            previous_map=previous_map,
            previous_centroids=previous_centroids,
            sample_size=args.sample_size,
        )
        if idx >= args.warmup:
            row["frame_index"] = idx - args.warmup
            all_rows.append(row)
            print(
                f"[eval] frame={row['frame_index']:03d} "
                f"score={row['effectiveness_score']:.2f} "
                f"coverage={row['cluster_coverage']:.3f} "
                f"silhouette={row['silhouette']:.3f}"
            )
        else:
            print(f"[warmup] frame={idx:03d}")

    summary = summarize(all_rows)
    summary["generated_at"] = datetime.now().isoformat(timespec="seconds")
    summary["pattern"] = args.pattern
    summary["warmup"] = args.warmup
    summary["limit"] = args.limit
    summary["sample_size"] = args.sample_size
    summary["note"] = (
        "No ground-truth labels were used. Metrics and score are unsupervised proxies."
    )

    csv_path = out_dir / "effectiveness_metrics.csv"
    json_path = out_dir / "effectiveness_summary.json"
    write_csv(all_rows, csv_path)
    write_json(summary, json_path)
    plot(all_rows, out_dir)

    print()
    print(f"Saved per-frame metrics: {csv_path}")
    print(f"Saved summary: {json_path}")
    print(
        "Saved plots: retention_coverage.png, clustering_quality.png, "
        "temporal_stability.png, effectiveness_score.png"
    )
    if summary:
        print(
            "Summary: "
            f"mean_score={summary['mean_effectiveness_score']:.2f}, "
            f"mean_silhouette={summary['mean_silhouette']:.3f}, "
            f"mean_jaccard={summary['mean_temporal_voxel_jaccard']:.3f}"
        )


if __name__ == "__main__":
    main()
