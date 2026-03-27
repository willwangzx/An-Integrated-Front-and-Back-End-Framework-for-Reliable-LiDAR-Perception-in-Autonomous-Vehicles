import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.clustering import cluster_objects
from src.config import (
    CLUSTER_EPS,
    CLUSTER_MIN_POINTS,
    ENABLE_INTENSITY_COMP,
    GROUND_THRESHOLD,
    MAX_RANGE,
    RANGE_ATTENUATION_ALPHA,
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
        help="Directory with KITTI .bin point clouds (overrides --pattern).",
    )
    parser.add_argument(
        "--calib-dir",
        default="",
        help=(
            "Directory with KITTI calibration .txt files. If provided together with "
            "--bin-dir, boxes are exported in camera coordinates (KITTI-compatible)."
        ),
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
        default=0.15,
        help="Drop predictions with score below this value.",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=CLUSTER_MIN_POINTS,
        help="DBSCAN min_samples for cluster extraction.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=CLUSTER_EPS,
        help="DBSCAN eps for cluster extraction.",
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

    center_xz = center + major * ((min_ma + max_ma) * 0.5) + minor * ((min_mi + max_mi) * 0.5)

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
        if np.linalg.norm(major) < 1e-6:
            yaw = 0.0
        else:
            yaw = float(math.atan2(float(major[1]), float(major[0])))
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


def classify_object(hwl: Tuple[float, float, float], num_points: int) -> str:
    h, w, l = hwl

    if h > 1.2 and h < 2.4 and w < 1.1 and l < 1.2 and num_points < 250:
        return "Pedestrian"

    if h > 1.1 and h < 2.4 and w < 1.3 and l >= 1.2 and l < 2.8 and num_points < 400:
        return "Cyclist"

    return "Car"


def confidence_score(num_points: int, hwl: Tuple[float, float, float]) -> float:
    h, w, l = hwl
    size_term = np.clip((h * w * l) / 18.0, 0.0, 1.0)
    density_term = 1.0 - math.exp(-max(num_points, 0) / 35.0)
    score = 0.2 + 0.5 * density_term + 0.3 * size_term
    return float(np.clip(score, 0.05, 0.99))


def alpha_from_ry_xyz(ry: float, x: float, z: float) -> float:
    if abs(x) < 1e-6 and abs(z) < 1e-6:
        return -10.0
    alpha = ry - math.atan2(x, z)
    return float((alpha + math.pi) % (2 * math.pi) - math.pi)


def bbox_from_projection(points_cam: np.ndarray, p2: Optional[np.ndarray]) -> Tuple[float, float, float, float]:
    if p2 is None or points_cam.shape[0] == 0:
        return -1.0, -1.0, -1.0, -1.0

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
    left = float(np.min(uv[:, 0]))
    top = float(np.min(uv[:, 1]))
    right = float(np.max(uv[:, 0]))
    bottom = float(np.max(uv[:, 1]))
    return left, top, right, bottom


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


def discover_input_files(bin_dir: str, pattern: str, limit: int) -> List[Path]:
    if bin_dir:
        files = sorted(Path(bin_dir).glob("*.bin"))
    else:
        files = [Path(path) for path in iter_lidar_frames(pattern)]

    if limit > 0:
        files = files[:limit]

    return files


def preprocess_points(points: np.ndarray, intensity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if ENABLE_INTENSITY_COMP:
        intensity = compensate_intensity(points, intensity, RANGE_ATTENUATION_ALPHA)

    points, intensity = filter_range(points, intensity, MAX_RANGE)
    points, intensity = remove_ground(points, intensity, GROUND_THRESHOLD)
    return points, intensity


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_files = discover_input_files(args.bin_dir, args.pattern, args.limit)
    if not input_files:
        raise FileNotFoundError("No input frames found. Check --bin-dir or --pattern.")

    use_calib = bool(args.bin_dir and args.calib_dir)
    if args.calib_dir and not args.bin_dir:
        print("[warn] --calib-dir is ignored unless --bin-dir is used.")

    summary = {
        "num_frames": len(input_files),
        "frames": [],
        "use_calib": use_calib,
        "out_dir": str(out_dir),
        "score_thresh": args.score_thresh,
        "eps": args.eps,
        "min_points": args.min_points,
    }

    for idx, input_path in enumerate(input_files):
        stem = input_path.stem

        if args.bin_dir:
            points, intensity = load_bin(str(input_path))
        else:
            points, intensity = load_las(str(input_path))

        points, intensity = preprocess_points(points, intensity)
        clusters = cluster_objects(points, args.eps, args.min_points)

        calib = None
        if use_calib:
            calib_path = Path(args.calib_dir) / f"{stem}.txt"
            if calib_path.exists():
                calib = load_calibration(calib_path)
            else:
                print(f"[warn] Missing calib for {stem}: {calib_path}")

        lines: List[str] = []
        class_counter: Dict[str, int] = {"Car": 0, "Pedestrian": 0, "Cyclist": 0}

        for cluster in clusters:
            if cluster.shape[0] < args.min_points:
                continue

            if calib is not None:
                cluster_cam = lidar_to_camera(cluster, calib)
                box = oriented_box_xz(cluster_cam)
                bbox = bbox_from_projection(cluster_cam, calib.get("P2"))
            else:
                cluster_cam = None
                box = pseudo_box_lidar(cluster)
                bbox = (-1.0, -1.0, -1.0, -1.0)

            if box is None:
                continue

            hwl = (box["h"], box["w"], box["l"])
            cls_name = classify_object(hwl, int(cluster.shape[0]))
            score = confidence_score(int(cluster.shape[0]), hwl)

            if score < args.score_thresh:
                continue

            if cluster_cam is not None:
                alpha = alpha_from_ry_xyz(box["ry"], box["x"], box["z"])
            else:
                alpha = -10.0

            line = format_kitti_line(cls_name, alpha, bbox, box, score)
            lines.append(line)
            class_counter[cls_name] += 1

        out_path = out_dir / f"{stem}.txt"
        with out_path.open("w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")

        frame_info = {
            "frame": stem,
            "clusters": int(len(clusters)),
            "exported": int(len(lines)),
            "class_counts": class_counter,
        }
        summary["frames"].append(frame_info)

        print(
            f"[export] frame={idx:04d} stem={stem} "
            f"clusters={len(clusters)} exported={len(lines)} "
            f"car={class_counter['Car']} ped={class_counter['Pedestrian']} cyc={class_counter['Cyclist']}"
        )

    summary_path = out_dir.parent / "kitti_export_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"Saved KITTI prediction txt files to: {out_dir}")
    print(f"Saved export summary: {summary_path}")
    if not use_calib:
        print(
            "[note] Export used LiDAR-frame pseudo boxes (no camera calibration). "
            "For official KITTI-style camera-coordinate evaluation, use --bin-dir + --calib-dir."
        )


if __name__ == "__main__":
    main()
