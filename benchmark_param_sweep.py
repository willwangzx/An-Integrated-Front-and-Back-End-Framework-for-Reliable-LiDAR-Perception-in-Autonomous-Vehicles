import argparse
import csv
import json
from collections import Counter, deque
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from statistics import mean
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from src.bev_map import generate_bev
from src.clustering import cluster_objects, cluster_objects_adaptive
from src.config import (
    ADAPTIVE_CLUSTER_EPS_SCALES,
    ADAPTIVE_CLUSTER_MIN_SCALES,
    ADAPTIVE_CLUSTER_RANGE_BINS,
    BEV_RESOLUTION,
    CLUSTER_EPS,
    CLUSTER_MIN_POINTS,
    CLUSTER_Z_SCALE,
    ENABLE_INTENSITY_COMP,
    EWMA_ALPHA,
    FUSION_WINDOW,
    GROUND_THRESHOLD,
    MAX_RANGE,
    RANGE_ATTENUATION_ALPHA,
    USE_ADAPTIVE_CLUSTER,
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

INT_PARAMS = {"cluster_min_points", "fusion_window"}
FLOAT_PARAMS = {
    "voxel_size",
    "cluster_eps",
    "bev_resolution",
    "max_range",
    "ground_threshold",
    "ewma_alpha",
    "range_attenuation_alpha",
}
BOOL_PARAMS = {"use_ewma", "enable_intensity_comp"}
SUPPORTED_PARAMS = INT_PARAMS | FLOAT_PARAMS | BOOL_PARAMS


@dataclass(frozen=True)
class PipelineParams:
    voxel_size: float = VOXEL_SIZE
    cluster_eps: float = CLUSTER_EPS
    cluster_min_points: int = CLUSTER_MIN_POINTS
    fusion_window: int = FUSION_WINDOW
    bev_resolution: float = BEV_RESOLUTION
    max_range: float = MAX_RANGE
    ground_threshold: float = GROUND_THRESHOLD
    ewma_alpha: float = EWMA_ALPHA
    use_ewma: bool = USE_EWMA_FUSION
    enable_intensity_comp: bool = ENABLE_INTENSITY_COMP
    range_attenuation_alpha: float = RANGE_ATTENUATION_ALPHA
    use_adaptive_cluster: bool = USE_ADAPTIVE_CLUSTER


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sweep LiDAR pipeline parameters (voxel size, clustering, fusion, etc.) "
            "and compare runtime costs."
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
        default=10,
        help="Max number of frames per experiment (0 = all).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Initial frames excluded from metrics in each experiment.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="How many times to repeat each parameter value.",
    )
    parser.add_argument(
        "--out-dir",
        default="benchmark_param_sweep",
        help="Directory to store CSV, JSON, and plots.",
    )
    parser.add_argument(
        "--sweep",
        action="append",
        default=[],
        help=(
            "One-dimensional sweep in the format param=v1,v2,... "
            "(example: --sweep voxel_size=0.1,0.2,0.3). "
            f"Supported params: {', '.join(sorted(SUPPORTED_PARAMS))}"
        ),
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation.",
    )
    return parser.parse_args()


def _safe_mean(values):
    return float(mean(values)) if values else 0.0


def _unique_preserve_order(values):
    out = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def _parse_bool(raw):
    text = str(raw).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {raw}")


def _parse_value(param, raw):
    if param in INT_PARAMS:
        return int(raw)
    if param in FLOAT_PARAMS:
        return float(raw)
    if param in BOOL_PARAMS:
        return _parse_bool(raw)
    raise ValueError(f"Unsupported sweep parameter: {param}")


def _format_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _validate_params(params):
    if params.voxel_size <= 0:
        raise ValueError("voxel_size must be > 0")
    if params.cluster_eps <= 0:
        raise ValueError("cluster_eps must be > 0")
    if params.cluster_min_points < 1:
        raise ValueError("cluster_min_points must be >= 1")
    if params.fusion_window < 1:
        raise ValueError("fusion_window must be >= 1")
    if params.bev_resolution <= 0:
        raise ValueError("bev_resolution must be > 0")
    if params.max_range <= 0:
        raise ValueError("max_range must be > 0")
    if not 0.0 <= params.ewma_alpha <= 1.0:
        raise ValueError("ewma_alpha must be in [0, 1]")


def _default_sweeps(base):
    return {
        "voxel_size": _unique_preserve_order(
            [round(max(0.05, base.voxel_size * scale), 4) for scale in (0.5, 1.0, 1.5, 2.0)]
        ),
        "cluster_eps": _unique_preserve_order(
            [round(base.cluster_eps * scale, 4) for scale in (0.75, 1.0, 1.25, 1.5)]
        ),
        "cluster_min_points": _unique_preserve_order(
            [
                max(3, int(round(base.cluster_min_points * scale)))
                for scale in (0.5, 1.0, 1.5, 2.0)
            ]
        ),
        "fusion_window": _unique_preserve_order(
            sorted({1, max(1, base.fusion_window // 2), base.fusion_window, base.fusion_window * 2})
        ),
        "bev_resolution": _unique_preserve_order(
            [round(max(0.02, base.bev_resolution * scale), 4) for scale in (0.5, 1.0, 2.0, 3.0)]
        ),
    }


def parse_sweeps(sweep_specs, base):
    if not sweep_specs:
        defaults = _default_sweeps(base)
        return [(name, defaults[name]) for name in defaults]

    parsed = []
    for spec in sweep_specs:
        if "=" not in spec:
            raise ValueError(
                f"Invalid --sweep '{spec}'. Expected format: param=v1,v2,..."
            )
        raw_name, raw_values = spec.split("=", 1)
        param = raw_name.strip().replace("-", "_")

        if param not in SUPPORTED_PARAMS:
            raise ValueError(
                f"Unsupported parameter '{param}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_PARAMS))}"
            )

        values = [v.strip() for v in raw_values.split(",") if v.strip()]
        if not values:
            raise ValueError(f"No values provided for --sweep {param}=...")

        typed_values = [_parse_value(param, value) for value in values]
        baseline = getattr(base, param)
        parsed.append((param, _unique_preserve_order([baseline] + typed_values)))
    return parsed


def run_profiled_frame(file_path, map_history, params):
    row = {"frame_file": file_path}

    total_start = perf_counter()

    start = perf_counter()
    points, intensity = load_las(file_path)
    row["load_s"] = perf_counter() - start
    row["points_raw"] = int(points.shape[0])

    if params.enable_intensity_comp:
        start = perf_counter()
        intensity = compensate_intensity(
            points, intensity, params.range_attenuation_alpha
        )
        row["intensity_comp_s"] = perf_counter() - start
    else:
        row["intensity_comp_s"] = 0.0

    start = perf_counter()
    points, intensity = filter_range(points, intensity, params.max_range)
    row["range_filter_s"] = perf_counter() - start
    row["points_after_range"] = int(points.shape[0])

    start = perf_counter()
    points, intensity = remove_ground(points, intensity, params.ground_threshold)
    row["ground_remove_s"] = perf_counter() - start
    row["points_after_ground"] = int(points.shape[0])

    start = perf_counter()
    voxels, voxel_intensity, voxel_counts = voxelize(points, intensity, params.voxel_size)
    row["voxelize_s"] = perf_counter() - start
    row["voxels_count"] = int(voxels.shape[0])
    row["mean_voxel_density"] = float(voxel_counts.mean()) if voxel_counts.size else 0.0  

    start = perf_counter()
    reflectivity_map = build_reflectivity(voxels, voxel_intensity)
    row["reflectivity_build_s"] = perf_counter() - start
    map_history.append(reflectivity_map)

    start = perf_counter()
    if params.use_adaptive_cluster:
        clusters = cluster_objects_adaptive(
            points=points,
            base_eps=params.cluster_eps,
            base_min_samples=params.cluster_min_points,
            range_bins=ADAPTIVE_CLUSTER_RANGE_BINS,
            eps_scales=ADAPTIVE_CLUSTER_EPS_SCALES,
            min_samples_scales=ADAPTIVE_CLUSTER_MIN_SCALES,
            z_scale=CLUSTER_Z_SCALE,
        )
    else:
        clusters = cluster_objects(
            points,
            params.cluster_eps,
            params.cluster_min_points,
            z_scale=CLUSTER_Z_SCALE,
        )
    row["cluster_s"] = perf_counter() - start
    row["clusters_count"] = int(len(clusters))
    row["mean_cluster_size"] = (
        float(np.mean([cluster.shape[0] for cluster in clusters])) if clusters else 0.0
    )

    start = perf_counter()
    fused_map, stability_map = fuse_maps(
        list(map_history),
        use_ewma=params.use_ewma,
        ewma_alpha=params.ewma_alpha,
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
        params.voxel_size,
    )
    row["feature_extract_s"] = perf_counter() - start
    row["objects_count"] = int(len(object_features))

    start = perf_counter()
    bev = generate_bev(points, params.bev_resolution)
    row["bev_s"] = perf_counter() - start
    row["bev_nonzero"] = int(np.count_nonzero(bev)) if bev.size else 0
    row["bev_cells"] = int(bev.size)

    row["total_s"] = perf_counter() - total_start
    row["processing_s"] = row["total_s"] - row["load_s"]

    return row


def summarize_rows(rows):
    total_times = [row["total_s"] for row in rows]
    processing_times = [row["processing_s"] for row in rows]
    if not total_times:
        return {}

    fps_values = [1.0 / value for value in total_times if value > 0]
    stage_means = {stage: _safe_mean([row[stage] for row in rows]) for stage in STAGE_KEYS}

    return {
        "num_frames_benchmarked": len(rows),
        "mean_total_s": _safe_mean(total_times),
        "median_total_s": float(np.median(total_times)),
        "p95_total_s": float(np.percentile(total_times, 95)),
        "mean_processing_s": _safe_mean(processing_times),
        "median_processing_s": float(np.median(processing_times)),
        "p95_processing_s": float(np.percentile(processing_times, 95)),
        "mean_fps": _safe_mean(fps_values),
        "mean_points_after_ground": _safe_mean([row["points_after_ground"] for row in rows]),
        "mean_voxels": _safe_mean([row["voxels_count"] for row in rows]),
        "mean_clusters": _safe_mean([row["clusters_count"] for row in rows]),
        "mean_objects": _safe_mean([row["objects_count"] for row in rows]),
        "stage_mean_s": stage_means,
        "cluster_count_distribution": dict(Counter(row["clusters_count"] for row in rows)),
    }


def write_csv(rows, out_csv):
    if not rows:
        return
    fields = list(rows[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(payload, out_json):
    with out_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def plot_param_curves(summary_rows, out_dir):
    if not summary_rows:
        return

    grouped = {}
    for row in summary_rows:
        grouped.setdefault(row["sweep_param"], []).append(row)

    plt.style.use("seaborn-v0_8-whitegrid")
    for param, rows in grouped.items():
        rows = sorted(rows, key=lambda item: item["param_value_numeric"])
        x = np.arange(len(rows))
        x_labels = [row["param_value_str"] for row in rows]
        total_ms = np.array([row["mean_total_ms"] for row in rows], dtype=np.float32)
        proc_ms = np.array([row["mean_processing_ms"] for row in rows], dtype=np.float32)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x, total_ms, marker="o", linewidth=2, color="#0b5d7a", label="mean total")
        ax.plot(
            x,
            proc_ms,
            marker="s",
            linewidth=2,
            color="#c76800",
            label="mean processing (total - load)",
        )
        ax.set_title(f"Runtime vs {param}")
        ax.set_xlabel(param)
        ax.set_ylabel("Runtime (ms)")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=20, ha="right")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"runtime_vs_{param}.png", dpi=150)
        plt.close(fig)


def _value_for_numeric_axis(value):
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return float(value)


def build_experiments(base_params, sweep_plan):
    experiments = []
    exp_id = 0
    for param_name, values in sweep_plan:
        for value in values:
            params = replace(base_params, **{param_name: value})
            _validate_params(params)
            exp_id += 1
            experiments.append(
                {
                    "experiment_id": exp_id,
                    "sweep_param": param_name,
                    "param_value": value,
                    "param_value_str": _format_value(value),
                    "param_value_numeric": _value_for_numeric_axis(value),
                    "params": params,
                }
            )
    return experiments


def run_experiment(frame_files, params, warmup, repeats):
    rows = []
    for repeat_idx in range(repeats):
        map_history = deque(maxlen=max(int(params.fusion_window), 1))
        for frame_idx, frame_file in enumerate(frame_files):
            row = run_profiled_frame(frame_file, map_history, params)
            if frame_idx >= warmup:
                row["repeat_index"] = repeat_idx
                row["frame_index"] = frame_idx - warmup
                rows.append(row)
    return rows


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_files = iter_lidar_frames(args.pattern)
    if args.limit > 0:
        frame_files = frame_files[: args.limit]
    if not frame_files:
        raise FileNotFoundError(f"No LAS files found with pattern: {args.pattern}")
    if args.warmup >= len(frame_files):
        raise ValueError("--warmup must be smaller than number of selected frames.")
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

    base_params = PipelineParams()
    sweep_plan = parse_sweeps(args.sweep, base_params)
    experiments = build_experiments(base_params, sweep_plan)

    print(f"Selected frames: {len(frame_files)}")
    print(f"Warmup frames per experiment: {args.warmup}")
    print(f"Repeats per experiment: {args.repeats}")
    print(f"Total experiments: {len(experiments)}")
    print()

    frame_rows = []
    summary_rows = []
    result_cache = {}
    for experiment in experiments:
        params = experiment["params"]
        print(
            f"[run {experiment['experiment_id']:03d}/{len(experiments):03d}] "
            f"{experiment['sweep_param']}={experiment['param_value_str']}"
        )

        if params in result_cache:
            cached_rows, cached_summary = result_cache[params]
            rows = [row.copy() for row in cached_rows]
            summary = cached_summary.copy()
        else:
            rows = run_experiment(
                frame_files=frame_files,
                params=params,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            summary = summarize_rows(rows)
            result_cache[params] = ([row.copy() for row in rows], summary.copy())

        for row in rows:
            row.update(
                {
                    "experiment_id": experiment["experiment_id"],
                    "sweep_param": experiment["sweep_param"],
                    "param_value_str": experiment["param_value_str"],
                    "param_value_numeric": experiment["param_value_numeric"],
                }
            )
            frame_rows.append(row)

        summary_row = {
            "experiment_id": experiment["experiment_id"],
            "sweep_param": experiment["sweep_param"],
            "param_value_str": experiment["param_value_str"],
            "param_value_numeric": experiment["param_value_numeric"],
            "num_frames_benchmarked": summary.get("num_frames_benchmarked", 0),
            "mean_total_ms": summary.get("mean_total_s", 0.0) * 1000.0,
            "p95_total_ms": summary.get("p95_total_s", 0.0) * 1000.0,
            "mean_processing_ms": summary.get("mean_processing_s", 0.0) * 1000.0,
            "p95_processing_ms": summary.get("p95_processing_s", 0.0) * 1000.0,
            "mean_fps": summary.get("mean_fps", 0.0),
            "mean_points_after_ground": summary.get("mean_points_after_ground", 0.0),
            "mean_voxels": summary.get("mean_voxels", 0.0),
            "mean_clusters": summary.get("mean_clusters", 0.0),
            "mean_objects": summary.get("mean_objects", 0.0),
        }
        for stage in STAGE_KEYS:
            summary_row[f"stage_{stage}_ms"] = (
                summary.get("stage_mean_s", {}).get(stage, 0.0) * 1000.0
            )
        summary_rows.append(summary_row)

        print(
            f"  mean_total={summary_row['mean_total_ms']:.2f}ms "
            f"p95={summary_row['p95_total_ms']:.2f}ms "
            f"fps={summary_row['mean_fps']:.2f}"
        )

    frame_csv = out_dir / "sweep_frame_metrics.csv"
    summary_csv = out_dir / "sweep_summary.csv"
    summary_json = out_dir / "sweep_summary.json"

    write_csv(frame_rows, frame_csv)
    write_csv(summary_rows, summary_csv)

    fastest = min(summary_rows, key=lambda row: row["mean_total_ms"]) if summary_rows else None
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "pattern": args.pattern,
        "limit": args.limit,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "base_params": asdict(base_params),
        "sweep_plan": {param: [_format_value(v) for v in values] for param, values in sweep_plan},
        "num_experiments": len(summary_rows),
        "selected_frames": len(frame_files),
        "fastest_experiment": fastest,
        "experiments": summary_rows,
    }
    write_json(payload, summary_json)

    if not args.no_plots:
        plot_param_curves(summary_rows, out_dir)

    print()
    print(f"Saved per-frame metrics: {frame_csv}")
    print(f"Saved sweep summary: {summary_csv}")
    print(f"Saved JSON summary: {summary_json}")
    if not args.no_plots:
        print("Saved plots: runtime_vs_<param>.png")
    if fastest:
        print(
            "Fastest config: "
            f"{fastest['sweep_param']}={fastest['param_value_str']} "
            f"(mean_total={fastest['mean_total_ms']:.2f}ms)"
        )


if __name__ == "__main__":
    main()
