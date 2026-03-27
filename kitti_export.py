import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.clustering import cluster_objects, cluster_objects_adaptive
from src.config import (
    ADAPTIVE_CLUSTER_EPS_SCALES,
    ADAPTIVE_CLUSTER_MIN_SCALES,
    ADAPTIVE_CLUSTER_RANGE_BINS,
    CAMERA_FOV_X_OVER_Z_MAX,
    CAMERA_MIN_DEPTH,
    CLUSTER_EPS,
    CLUSTER_MIN_POINTS,
    CLUSTER_Z_SCALE,
    ENABLE_INTENSITY_COMP,
    GROUND_THRESHOLD,
    MAX_BOX_HWL,
    MAX_RANGE,
    MIN_BOX_HWL,
    RANGE_ATTENUATION_ALPHA,
    ROI_X_MAX,
    ROI_X_MIN,
    ROI_Y_ABS_MAX,
    ROI_Z_MAX,
    ROI_Z_MIN,
)
from src.lidar_loader import load_las
from src.pipeline import iter_lidar_frames
from src.preprocessing import compensate_intensity, filter_range, remove_ground


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export LiDAR clustering results to KITTI prediction text files "
            "(one .txt per frame)."
        )
    )
    parser.add_argument(
        "--pattern",
        default="data/velodyne_points/las/*.las",
        help="Glob pattern for LAS frames when --bin-dir is not provided.",
    )
    parser.add_argument(
        "--bin-dir",
        default="",
        help="Directory with KITTI .bin point clouds (preferred for KITTI eval).",
    )
    parser.add_argument(
        "--calib-dir",
        default="",
        help="Directory with KITTI calibration .txt files matching frame ids.",
    )
    parser.add_argument(
        "--out-dir",
        default="kitti_predictions/data",
        help="Output directory for KITTI prediction txt files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of frames to export (0 = all).",
    )
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.35,
        help="Drop predictions with score below this value.",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=CLUSTER_MIN_POINTS,
        help="Base DBSCAN min_samples before adaptive scaling.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=CLUSTER_EPS,
        help="Base DBSCAN eps before adaptive scaling.",
    )
    parser.add_argument(
        "--disable-adaptive-cluster",
        action="store_true",
        help="Disable range-adaptive DBSCAN and use one global DBSCAN config.",
    )
    parser.add_argument(
        "--disable-fov-filter",
        action="store_true",
        help="Disable camera-overlap FOV filtering (calibrated mode only).",
    )
    parser.add_argument(
        "--score-calib-json",
        default="",
        help=(
            "Optional score calibration JSON from validation (per-class scale/bias). "
            "If missing, conservative default calibration is used."
        ),
    )
    parser.add_argument(
        "--allow-uncalibrated",
        action="store_true",
        help=(
            "Allow export without camera calibration for debugging only "
            "(produces KITTI-invalid alpha/bbox fields)."
        ),
    )
    return parser.parse_args()


def load_bin(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    raw = np.fromfile(file_path, dtype=np.float32)
    if raw.size % 4 != 0:
        raise ValueError(f"Invalid KITTI .bin file: {file_path}")

    points = raw.reshape(-1, 4)
    xyz = points[:, :3].astype(np.float32, copy=False)
    intensity = points[:, 3].astype(np.float32, copy=False)

    peak = float(np.max(intensity)) if intensity.size else 0.0
    if peak > 0:
        intensity = intensity / peak
    return xyz, intensity


def discover_input_files(bin_dir: str, pattern: str, limit: int) -> List[Path]:
    if bin_dir:
        files = sorted(Path(bin_dir).glob("*.bin"))
    else:
        files = [Path(path) for path in iter_lidar_frames(pattern)]

    if limit > 0:
        files = files[:limit]
    return files


def load_calibration(calib_path: Path) -> Dict[str, np.ndarray]:
    values: Dict[str, np.ndarray] = {}
    with calib_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            key, data = line.split(":", 1)
            nums = np.array([float(x) for x in data.strip().split()], dtype=np.float32)
            values[key] = nums

    if "Tr_velo_to_cam" in values:
        tr = values["Tr_velo_to_cam"].reshape(3, 4)
    elif "Tr_velo_cam" in values:
        tr = values["Tr_velo_cam"].reshape(3, 4)
    else:
        raise KeyError(f"Missing Tr_velo_to_cam in {calib_path}")

    if "R0_rect" in values:
        r0 = values["R0_rect"].reshape(3, 3)
    elif "R_rect" in values:
        r0 = values["R_rect"].reshape(3, 3)
    else:
        r0 = np.eye(3, dtype=np.float32)

    p2 = values["P2"].reshape(3, 4) if "P2" in values else None
    if p2 is None:
        raise KeyError(f"Missing P2 projection matrix in {calib_path}")

    return {
        "Tr_velo_to_cam": tr,
        "R0_rect": r0,
        "P2": p2,
    }


def lidar_to_camera(points_lidar: np.ndarray, calib: Dict[str, np.ndarray]) -> np.ndarray:
    tr = calib["Tr_velo_to_cam"]
    r0 = calib["R0_rect"]
    cam = points_lidar @ tr[:, :3].T + tr[:, 3]
    cam = cam @ r0.T
    return cam.astype(np.float32, copy=False)


def preprocess_points(points: np.ndarray, intensity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if ENABLE_INTENSITY_COMP:
        intensity = compensate_intensity(points, intensity, RANGE_ATTENUATION_ALPHA)
    points, intensity = filter_range(points, intensity, MAX_RANGE)
    points, intensity = remove_ground(points, intensity, GROUND_THRESHOLD)
    return points, intensity


def roi_filter_lidar(points: np.ndarray, intensity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if points.size == 0:
        return points, intensity
    mask = (
        (points[:, 0] >= ROI_X_MIN)
        & (points[:, 0] <= ROI_X_MAX)
        & (np.abs(points[:, 1]) <= ROI_Y_ABS_MAX)
        & (points[:, 2] >= ROI_Z_MIN)
        & (points[:, 2] <= ROI_Z_MAX)
    )
    return points[mask], intensity[mask]


def fov_filter_with_calib(
    points: np.ndarray,
    intensity: np.ndarray,
    calib: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    if points.size == 0:
        return points, intensity
    points_cam = lidar_to_camera(points, calib)
    depth = points_cam[:, 2]
    x_over_z = np.abs(points_cam[:, 0] / np.maximum(depth, 1e-3))
    mask = (depth >= CAMERA_MIN_DEPTH) & (x_over_z <= CAMERA_FOV_X_OVER_Z_MAX)
    return points[mask], intensity[mask]


def oriented_box_xz(points_cam: np.ndarray) -> Optional[Dict[str, float]]:
    if points_cam.shape[0] < 3:
        return None

    xz = points_cam[:, [0, 2]]
    center = xz.mean(axis=0)
    centered = xz - center
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, int(np.argmax(eigvals))]
    major_norm = np.linalg.norm(major)
    if major_norm < 1e-6:
        major = np.array([1.0, 0.0], dtype=np.float32)
    else:
        major = (major / major_norm).astype(np.float32, copy=False)
    minor = np.array([-major[1], major[0]], dtype=np.float32)

    proj_major = centered @ major
    proj_minor = centered @ minor
    min_ma = float(np.min(proj_major))
    max_ma = float(np.max(proj_major))
    min_mi = float(np.min(proj_minor))
    max_mi = float(np.max(proj_minor))
    length = max(max_ma - min_ma, 1e-2)
    width = max(max_mi - min_mi, 1e-2)

    center_xz = (
        center
        + major * ((min_ma + max_ma) * 0.5)
        + minor * ((min_mi + max_mi) * 0.5)
    )

    y_min = float(np.min(points_cam[:, 1]))
    y_max = float(np.max(points_cam[:, 1]))
    height = max(y_max - y_min, 1e-2)
    ry = math.atan2(float(major[0]), float(major[1]))

    return {
        "x": float(center_xz[0]),
        "y": float(y_max),
        "z": float(center_xz[1]),
        "h": float(height),
        "w": float(width),
        "l": float(length),
        "ry": float(ry),
    }


def pseudo_box_lidar(points_lidar: np.ndarray) -> Optional[Dict[str, float]]:
    if points_lidar.shape[0] < 3:
        return None
    mins = points_lidar.min(axis=0)
    maxs = points_lidar.max(axis=0)
    extent = np.maximum(maxs - mins, 1e-2)
    center = (mins + maxs) * 0.5

    xy = points_lidar[:, :2]
    centered = xy - xy.mean(axis=0)
    if centered.shape[0] >= 3:
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        major = eigvecs[:, int(np.argmax(eigvals))]
        yaw = 0.0 if np.linalg.norm(major) < 1e-6 else float(math.atan2(major[1], major[0]))
    else:
        yaw = 0.0

    return {
        "x": float(center[0]),
        "y": float(center[1]),
        "z": float(center[2]),
        "h": float(extent[2]),
        "w": float(extent[1]),
        "l": float(extent[0]),
        "ry": float(yaw),
    }


def bbox_from_projection(points_cam: np.ndarray, p2: np.ndarray) -> Tuple[float, float, float, float]:
    valid_depth = points_cam[:, 2] > 1e-3
    if int(np.sum(valid_depth)) < 4:
        return -1.0, -1.0, -1.0, -1.0

    pts = points_cam[valid_depth]
    pts_h = np.column_stack((pts, np.ones((pts.shape[0],), dtype=np.float32)))
    proj = pts_h @ p2.T
    valid_proj = proj[:, 2] > 1e-6
    if int(np.sum(valid_proj)) < 4:
        return -1.0, -1.0, -1.0, -1.0

    uv = proj[valid_proj, :2] / proj[valid_proj, 2:3]
    return (
        float(np.min(uv[:, 0])),
        float(np.min(uv[:, 1])),
        float(np.max(uv[:, 0])),
        float(np.max(uv[:, 1])),
    )


def alpha_from_ry_xyz(ry: float, x: float, z: float) -> float:
    alpha = ry - math.atan2(x, z)
    return float((alpha + math.pi) % (2 * math.pi) - math.pi)


def valid_hwl(h: float, w: float, l: float) -> bool:
    min_h, min_w, min_l = MIN_BOX_HWL
    max_h, max_w, max_l = MAX_BOX_HWL
    return (
        (h >= min_h)
        and (w >= min_w)
        and (l >= min_l)
        and (h <= max_h)
        and (w <= max_w)
        and (l <= max_l)
    )


def class_logit(
    h: float,
    w: float,
    l: float,
    num_points: int,
    range_m: float,
    mean_hwl: Tuple[float, float, float],
    std_hwl: Tuple[float, float, float],
    expected_points: Tuple[int, int],
) -> float:
    mh, mw, ml = mean_hwl
    sh, sw, sl = std_hwl
    z_h = abs(h - mh) / max(sh, 1e-3)
    z_w = abs(w - mw) / max(sw, 1e-3)
    z_l = abs(l - ml) / max(sl, 1e-3)
    geom = -(0.45 * z_h + 0.25 * z_w + 0.30 * z_l)

    p_lo, p_hi = expected_points
    if num_points < p_lo:
        point_penalty = -0.06 * (p_lo - num_points)
    elif num_points > p_hi:
        point_penalty = -0.015 * (num_points - p_hi)
    else:
        point_penalty = 0.1

    range_penalty = -0.015 * max(range_m - 45.0, 0.0)
    return float(geom + point_penalty + range_penalty)


def classify_object(
    hwl: Tuple[float, float, float],
    num_points: int,
    range_m: float,
) -> Tuple[str, Dict[str, float]]:
    h, w, l = hwl
    specs = {
        "Car": {
            "mean": (1.55, 1.75, 3.95),
            "std": (0.45, 0.7, 1.4),
            "pts": (18, 320),
        },
        "Pedestrian": {
            "mean": (1.72, 0.65, 0.85),
            "std": (0.35, 0.25, 0.45),
            "pts": (8, 160),
        },
        "Cyclist": {
            "mean": (1.70, 0.70, 1.80),
            "std": (0.35, 0.3, 0.7),
            "pts": (8, 220),
        },
    }

    logits = {
        cls_name: class_logit(
            h=h,
            w=w,
            l=l,
            num_points=num_points,
            range_m=range_m,
            mean_hwl=spec["mean"],
            std_hwl=spec["std"],
            expected_points=spec["pts"],
        )
        for cls_name, spec in specs.items()
    }

    sorted_items = sorted(logits.items(), key=lambda kv: kv[1], reverse=True)
    best_cls, best_score = sorted_items[0]
    second_score = sorted_items[1][1]
    margin = best_score - second_score
    return best_cls, {"best_logit": best_score, "margin": margin}


def load_score_calibration(calib_json: str) -> Dict[str, Dict[str, float]]:
    if not calib_json:
        return {}
    path = Path(calib_json)
    if not path.exists():
        raise FileNotFoundError(f"Score calibration file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Score calibration JSON must be an object.")
    return payload


def calibrated_probability(
    raw_score: float,
    cls_name: str,
    calibration: Dict[str, Dict[str, float]],
) -> float:
    cls_cal = calibration.get(cls_name, {})
    global_cal = calibration.get("global", {})
    scale = float(cls_cal.get("scale", global_cal.get("scale", 4.0)))
    bias = float(cls_cal.get("bias", global_cal.get("bias", -2.0)))
    p = 1.0 / (1.0 + math.exp(-(scale * raw_score + bias)))
    return float(np.clip(p, 0.01, 0.99))


def confidence_score(
    num_points: int,
    hwl: Tuple[float, float, float],
    cls_name: str,
    cls_info: Dict[str, float],
    bbox: Tuple[float, float, float, float],
    calibration: Dict[str, Dict[str, float]],
) -> float:
    h, w, l = hwl
    volume = h * w * l
    points_term = np.clip(np.log1p(num_points) / 6.0, 0.0, 1.0)
    size_term = float(np.exp(-abs(volume - 12.0) / 16.0))
    margin_term = float(np.clip((cls_info["margin"] + 2.0) / 4.0, 0.0, 1.0))

    lft, top, rgt, bot = bbox
    has_bbox = float(lft >= 0 and top >= 0 and rgt > lft and bot > top)
    raw = float(0.35 * margin_term + 0.30 * points_term + 0.25 * size_term + 0.10 * has_bbox)
    return calibrated_probability(raw, cls_name, calibration)


def format_kitti_line(
    cls_name: str,
    alpha: float,
    bbox: Tuple[float, float, float, float],
    box: Dict[str, float],
    score: float,
) -> str:
    l, t, r, b = bbox
    return (
        f"{cls_name} "
        f"-1 -1 {alpha:.4f} "
        f"{l:.2f} {t:.2f} {r:.2f} {b:.2f} "
        f"{box['h']:.3f} {box['w']:.3f} {box['l']:.3f} "
        f"{box['x']:.3f} {box['y']:.3f} {box['z']:.3f} {box['ry']:.4f} "
        f"{score:.4f}"
    )


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    calibration = load_score_calibration(args.score_calib_json)

    input_files = discover_input_files(args.bin_dir, args.pattern, args.limit)
    if not input_files:
        raise FileNotFoundError("No input frames found. Check --bin-dir or --pattern.")

    if args.bin_dir and not args.calib_dir and not args.allow_uncalibrated:
        raise ValueError(
            "KITTI eval export requires calibration. Provide --calib-dir "
            "or set --allow-uncalibrated for debug-only output."
        )
    if (not args.bin_dir) and (not args.allow_uncalibrated):
        raise ValueError(
            "LAS-only export is debug-only. For KITTI eval use --bin-dir + --calib-dir, "
            "or pass --allow-uncalibrated."
        )

    use_calib = bool(args.bin_dir and args.calib_dir)
    summary = {
        "num_frames": len(input_files),
        "frames": [],
        "use_calib": use_calib,
        "out_dir": str(out_dir),
        "score_thresh": args.score_thresh,
        "eps": args.eps,
        "min_points": args.min_points,
        "adaptive_cluster": not args.disable_adaptive_cluster,
        "fov_filter": use_calib and (not args.disable_fov_filter),
        "score_calibration": bool(calibration),
    }

    for idx, input_path in enumerate(input_files):
        stem = input_path.stem
        if args.bin_dir:
            points, intensity = load_bin(str(input_path))
        else:
            points, intensity = load_las(str(input_path))

        points, intensity = preprocess_points(points, intensity)
        points, intensity = roi_filter_lidar(points, intensity)

        calib = None
        if use_calib:
            calib_path = Path(args.calib_dir) / f"{stem}.txt"
            if not calib_path.exists():
                raise FileNotFoundError(f"Missing calibration file for frame {stem}: {calib_path}")
            calib = load_calibration(calib_path)
            if not args.disable_fov_filter:
                points, intensity = fov_filter_with_calib(points, intensity, calib)

        if points.shape[0] < max(args.min_points, 3):
            clusters: List[np.ndarray] = []
        elif args.disable_adaptive_cluster:
            clusters = cluster_objects(
                points,
                eps=args.eps,
                min_samples=args.min_points,
                z_scale=CLUSTER_Z_SCALE,
            )
        else:
            clusters = cluster_objects_adaptive(
                points=points,
                base_eps=args.eps,
                base_min_samples=args.min_points,
                range_bins=ADAPTIVE_CLUSTER_RANGE_BINS,
                eps_scales=ADAPTIVE_CLUSTER_EPS_SCALES,
                min_samples_scales=ADAPTIVE_CLUSTER_MIN_SCALES,
                z_scale=CLUSTER_Z_SCALE,
            )

        lines: List[str] = []
        class_counter: Dict[str, int] = {"Car": 0, "Pedestrian": 0, "Cyclist": 0}
        dropped_size = 0
        dropped_bbox = 0
        dropped_score = 0

        for cluster in clusters:
            if cluster.shape[0] < max(3, args.min_points):
                continue

            if calib is not None:
                cluster_cam = lidar_to_camera(cluster, calib)
                front_mask = cluster_cam[:, 2] >= CAMERA_MIN_DEPTH
                if int(np.sum(front_mask)) < args.min_points:
                    continue
                cluster_cam = cluster_cam[front_mask]
                box = oriented_box_xz(cluster_cam)
                if box is None or box["z"] < CAMERA_MIN_DEPTH:
                    continue
                bbox = bbox_from_projection(cluster_cam, calib["P2"])
                if bbox[0] < 0:
                    dropped_bbox += 1
                    continue
                alpha = alpha_from_ry_xyz(box["ry"], box["x"], box["z"])
                range_m = float(math.hypot(box["x"], box["z"]))
            else:
                box = pseudo_box_lidar(cluster)
                if box is None:
                    continue
                bbox = (-1.0, -1.0, -1.0, -1.0)
                alpha = -10.0
                range_m = float(math.hypot(box["x"], box["y"]))

            hwl = (box["h"], box["w"], box["l"])
            if not valid_hwl(*hwl):
                dropped_size += 1
                continue

            cls_name, cls_info = classify_object(
                hwl=hwl,
                num_points=int(cluster.shape[0]),
                range_m=range_m,
            )
            score = confidence_score(
                num_points=int(cluster.shape[0]),
                hwl=hwl,
                cls_name=cls_name,
                cls_info=cls_info,
                bbox=bbox,
                calibration=calibration,
            )
            if score < args.score_thresh:
                dropped_score += 1
                continue

            lines.append(format_kitti_line(cls_name, alpha, bbox, box, score))
            class_counter[cls_name] += 1

        out_path = out_dir / f"{stem}.txt"
        with out_path.open("w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")

        frame_info = {
            "frame": stem,
            "points_after_filter": int(points.shape[0]),
            "clusters": int(len(clusters)),
            "exported": int(len(lines)),
            "dropped_size": int(dropped_size),
            "dropped_bbox": int(dropped_bbox),
            "dropped_score": int(dropped_score),
            "class_counts": class_counter,
        }
        summary["frames"].append(frame_info)

        print(
            f"[export] frame={idx:04d} stem={stem} "
            f"points={frame_info['points_after_filter']} "
            f"clusters={len(clusters)} exported={len(lines)} "
            f"drop_size={dropped_size} drop_bbox={dropped_bbox} drop_score={dropped_score}"
        )

    summary_path = out_dir.parent / "kitti_export_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"Saved KITTI prediction txt files to: {out_dir}")
    print(f"Saved export summary: {summary_path}")
    if not use_calib:
        print(
            "[warn] Export used uncalibrated fallback for debugging. "
            "Do not use this output for official KITTI AP."
        )


if __name__ == "__main__":
    main()
