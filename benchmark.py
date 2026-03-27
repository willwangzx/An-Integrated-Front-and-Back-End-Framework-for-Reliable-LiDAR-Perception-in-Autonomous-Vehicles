import argparse
import csv
import json
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from statistics import mean
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from src.bev_map import generate_bev
from src.clustering import cluster_objects
from src.config import (
    BEV_RESOLUTION,
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
from src.reflectivity_map import build_reflectivity, interpolate_reflectivity
from src.temporal_fusion import fuse_maps
from src.voxelization import voxelize

STAGE_KEYS = [
    "load_s",
    "intensity_comp_s",
    "range_filter_s",
    "ground_remove_s",
    "voxelize_s",
    "reflectivity_build_s",
    "cluster_s",
    "fusion_s",
    "interpolate_s",
    "feature_extract_s",
    "bev_s",
    "total_s",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run evaluation + performance benchmarks for the LiDAR pipeline."
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
        help="Max number of frames to benchmark (0 = all).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of initial frames to run but exclude from metrics.",
    )
    parser.add_argument(
        "--out-dir",
        default="benchmark_results",
        help="Directory to store benchmark CSV, JSON, and plots.",
    )
    return parser.parse_args()


def _safe_mean(values):
    return float(mean(values)) if values else 0.0


def run_profiled_frame(file_path, map_history):
    row = {
        "frame_file": file_path,
    }

    total_start = perf_counter()

    start = perf_counter()
    points, intensity = load_las(file_path)
    row["load_s"] = perf_counter() - start
    row["points_raw"] = int(points.shape[0])

    if ENABLE_INTENSITY_COMP:
        start = perf_counter()
        intensity = compensate_intensity(points, intensity, RANGE_ATTENUATION_ALPHA)
        row["intensity_comp_s"] = perf_counter() - start
    else:
        row["intensity_comp_s"] = 0.0

    start = perf_counter()
    points, intensity = filter_range(points, intensity, MAX_RANGE)
    row["range_filter_s"] = perf_counter() - start
    row["points_after_range"] = int(points.shape[0])

    start = perf_counter()
    points, intensity = remove_ground(points, intensity, GROUND_THRESHOLD)
    row["ground_remove_s"] = perf_counter() - start
    row["points_after_ground"] = int(points.shape[0])

    start = perf_counter()
    voxels, voxel_intensity, voxel_counts = voxelize(points, intensity, VOXEL_SIZE)
    row["voxelize_s"] = perf_counter() - start
    row["voxels_count"] = int(voxels.shape[0])
    row["mean_voxel_density"] = float(voxel_counts.mean()) if voxel_counts.size else 0.0

    start = perf_counter()
    reflectivity_map = build_reflectivity(voxels, voxel_intensity)
    row["reflectivity_build_s"] = perf_counter() - start

    map_history.append(reflectivity_map)

    start = perf_counter()
    clusters = cluster_objects(points, CLUSTER_EPS, CLUSTER_MIN_POINTS)
    row["cluster_s"] = perf_counter() - start
    row["clusters_count"] = int(len(clusters))
    row["mean_cluster_size"] = (
        float(np.mean([cluster.shape[0] for cluster in clusters])) if clusters else 0.0
    )

    start = perf_counter()
    fused_map, stability_map = fuse_maps(
        list(map_history),
        use_ewma=USE_EWMA_FUSION,
        ewma_alpha=EWMA_ALPHA,
    )
    row["fusion_s"] = perf_counter() - start
    row["fused_voxels_count"] = int(len(fused_map))
    row["mean_stability"] = float(_safe_mean(list(stability_map.values())))

    start = perf_counter()
    fused_map = interpolate_reflectivity(fused_map, stability_map)
    row["interpolate_s"] = perf_counter() - start

    start = perf_counter()
    object_features = extract_object_features(
        clusters,
        fused_map,
        stability_map,
        VOXEL_SIZE,
    )
    row["feature_extract_s"] = perf_counter() - start
    row["objects_count"] = int(len(object_features))

    if object_features:
        row["mean_object_height"] = float(
            np.mean([feature["height"] for feature in object_features])
        )
        row["mean_object_footprint"] = float(
            np.mean([feature["footprint_area"] for feature in object_features])
        )
    else:
        row["mean_object_height"] = 0.0
        row["mean_object_footprint"] = 0.0

    start = perf_counter()
    bev = generate_bev(points, BEV_RESOLUTION)
    row["bev_s"] = perf_counter() - start
    row["bev_nonzero"] = int(np.count_nonzero(bev)) if bev.size else 0
    row["bev_cells"] = int(bev.size)

    row["total_s"] = perf_counter() - total_start
    row["total_stage_s"] = sum(row[key] for key in STAGE_KEYS if key != "total_s")

    return row


def summarize_rows(rows):
    total_times = [row["total_s"] for row in rows]
    if not total_times:
        return {}

    fps_values = [1.0 / value for value in total_times if value > 0]

    stage_means = {stage: _safe_mean([row[stage] for row in rows]) for stage in STAGE_KEYS}

    cluster_counts = [row["clusters_count"] for row in rows]
    object_counts = [row["objects_count"] for row in rows]
    points_after_ground = [row["points_after_ground"] for row in rows]

    return {
        "num_frames_benchmarked": len(rows),
        "mean_total_s": _safe_mean(total_times),
        "median_total_s": float(np.median(total_times)),
        "p95_total_s": float(np.percentile(total_times, 95)),
        "mean_fps": _safe_mean(fps_values),
        "median_fps": float(np.median(fps_values)) if fps_values else 0.0,
        "mean_clusters": _safe_mean(cluster_counts),
        "mean_objects": _safe_mean(object_counts),
        "mean_points_after_ground": _safe_mean(points_after_ground),
        "stage_mean_s": stage_means,
        "cluster_count_distribution": dict(Counter(cluster_counts)),
    }


def write_csv(rows, out_csv):
    if not rows:
        return
    fields = list(rows[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(payload, out_json):
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def plot_results(rows, summary, out_dir):
    if not rows:
        return

    frame_idx = np.arange(len(rows))
    total_s = np.array([row["total_s"] for row in rows], dtype=np.float32)
    points = np.array([row["points_after_ground"] for row in rows], dtype=np.int32)
    clusters = np.array([row["clusters_count"] for row in rows], dtype=np.int32)
    objects = np.array([row["objects_count"] for row in rows], dtype=np.int32)
    voxels = np.array([row["voxels_count"] for row in rows], dtype=np.int32)

    plt.style.use("seaborn-v0_8-whitegrid")

    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(frame_idx, total_s * 1000, color="#0b5d7a", linewidth=2)
    ax1.set_title("Per-frame End-to-end Runtime")
    ax1.set_xlabel("Frame index")
    ax1.set_ylabel("Runtime (ms)")
    fig1.tight_layout()
    fig1.savefig(out_dir / "runtime_per_frame.png", dpi=150)
    plt.close(fig1)

    stage_names = [stage for stage in STAGE_KEYS if stage != "total_s"]
    stage_means_ms = [summary["stage_mean_s"][stage] * 1000 for stage in stage_names]

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    x = np.arange(len(stage_names))
    ax2.bar(x, stage_means_ms, color="#2c9f45")
    ax2.set_title("Average Stage Runtime")
    ax2.set_ylabel("Runtime (ms)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(stage_names, rotation=35, ha="right")
    fig2.tight_layout()
    fig2.savefig(out_dir / "stage_runtime_breakdown.png", dpi=150)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(12, 5))
    ax3.scatter(points, total_s * 1000, c=clusters, cmap="viridis", s=40)
    ax3.set_title("Runtime vs Point Count (color = cluster count)")
    ax3.set_xlabel("Points after preprocessing")
    ax3.set_ylabel("Runtime (ms)")
    fig3.tight_layout()
    fig3.savefig(out_dir / "runtime_vs_points.png", dpi=150)
    plt.close(fig3)

    fig4, ax4 = plt.subplots(figsize=(12, 5))
    ax4.plot(frame_idx, voxels, label="voxels", linewidth=2)
    ax4.plot(frame_idx, clusters, label="clusters", linewidth=2)
    ax4.plot(frame_idx, objects, label="objects", linewidth=2)
    ax4.set_title("Scene Complexity Metrics by Frame")
    ax4.set_xlabel("Frame index")
    ax4.set_ylabel("Count")
    ax4.legend()
    fig4.tight_layout()
    fig4.savefig(out_dir / "scene_complexity.png", dpi=150)
    plt.close(fig4)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_files = iter_lidar_frames(args.pattern)
    if args.limit > 0:
        frame_files = frame_files[: args.limit]

    if not frame_files:
        raise FileNotFoundError(
            f"No LAS files found with pattern: {args.pattern}"
        )

    print(f"Found {len(frame_files)} frame(s).")
    print(f"Warmup frames excluded from metrics: {args.warmup}")

    map_history = deque(maxlen=max(FUSION_WINDOW, 1))
    benchmark_rows = []

    for index, file_path in enumerate(frame_files):
        row = run_profiled_frame(file_path, map_history)
        if index >= args.warmup:
            row["frame_index"] = index - args.warmup
            benchmark_rows.append(row)
            print(
                f"[bench] frame={row['frame_index']:03d} "
                f"total={row['total_s'] * 1000:.2f}ms "
                f"points={row['points_after_ground']} "
                f"clusters={row['clusters_count']}"
            )
        else:
            print(
                f"[warmup] frame={index:03d} "
                f"total={row['total_s'] * 1000:.2f}ms"
            )

    summary = summarize_rows(benchmark_rows)
    summary["generated_at"] = datetime.now().isoformat(timespec="seconds")
    summary["pattern"] = args.pattern
    summary["warmup"] = args.warmup
    summary["limit"] = args.limit

    csv_path = out_dir / "benchmark_metrics.csv"
    json_path = out_dir / "benchmark_summary.json"
    write_csv(benchmark_rows, csv_path)
    write_json(summary, json_path)
    plot_results(benchmark_rows, summary, out_dir)

    print()
    print(f"Saved per-frame metrics: {csv_path}")
    print(f"Saved summary metrics: {json_path}")
    print(
        "Saved plots: runtime_per_frame.png, stage_runtime_breakdown.png, "
        "runtime_vs_points.png, scene_complexity.png"
    )

    if summary:
        print(
            "Benchmark summary: "
            f"mean_total={summary['mean_total_s'] * 1000:.2f}ms "
            f"p95_total={summary['p95_total_s'] * 1000:.2f}ms "
            f"mean_fps={summary['mean_fps']:.2f}"
        )


if __name__ == "__main__":
    main()
