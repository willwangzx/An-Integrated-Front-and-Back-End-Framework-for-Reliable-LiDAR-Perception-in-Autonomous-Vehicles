import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.frontend import (
    OpenPCDetFrontendHook,
    apply_rigid_transform,
    make_frontend_stats,
    merge_frontend_stats,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run OpenPCDet inference on KITTI velodyne .bin frames and export "
            "KITTI-format prediction txt files."
        )
    )
    parser.add_argument(
        "--openpcdet-root",
        required=True,
        help="Path to OpenPCDet repository root.",
    )
    parser.add_argument(
        "--cfg-file",
        required=True,
        help="OpenPCDet model config YAML.",
    )
    parser.add_argument(
        "--ckpt",
        required=True,
        help="OpenPCDet checkpoint file.",
    )
    parser.add_argument(
        "--bin-dir",
        required=True,
        help="KITTI velodyne directory (training/velodyne).",
    )
    parser.add_argument(
        "--calib-dir",
        required=True,
        help="KITTI calibration directory (training/calib).",
    )
    parser.add_argument(
        "--out-dir",
        default="kitti_predictions_openpcdet/data",
        help="Output directory for KITTI prediction txt files.",
    )
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.1,
        help="Drop predictions with score below this threshold.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of frames to export (0 = all).",
    )
    parser.add_argument(
        "--ext",
        default=".bin",
        help="Point cloud extension, usually .bin.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Inference device.",
    )
    parser.add_argument(
        "--frontend-profile",
        choices=[
            "none",
            "safe",
            "denoise",
            "adaptive",
            "adaptive_enhance",
            "adaptive_ground",
            "temporal_3frame",
        ],
        default="adaptive",
        help=(
            "Optional OpenPCDet input frontend hook. "
            "'safe' applies only conservative preprocessing: "
            "drop NaN/Inf rows, clip extreme values, and filter to point_cloud_range. "
            "'denoise' applies safe preprocessing + voxel-neighborhood denoising. "
            "'adaptive' applies safe preprocessing plus intensity auto-repair, and can "
            "optionally apply guarded denoising. "
            "'adaptive_enhance' applies adaptive + distance-adaptive voxel-stat "
            "foreground feature enhancement. "
            "'adaptive_ground' applies adaptive + sparse near-ground denoise with "
            "drop-ratio guard. "
            "'temporal_3frame' stacks prev/current/next frames and applies optional "
            "pose-based ego-motion compensation."
        ),
    )
    parser.add_argument(
        "--frontend-xyz-clip-abs",
        type=float,
        default=1e4,
        help="Absolute clip bound for xyz in safe frontend profile.",
    )
    parser.add_argument(
        "--frontend-intensity-clip-abs",
        type=float,
        default=1e3,
        help="Absolute clip bound for intensity in safe frontend profile.",
    )
    parser.add_argument(
        "--frontend-denoise-voxel-size",
        type=float,
        default=0.25,
        help="Voxel size (meters) for denoise profile.",
    )
    parser.add_argument(
        "--frontend-denoise-neighbor-radius-voxels",
        type=int,
        default=1,
        help="Neighbor search radius in voxels for denoise profile.",
    )
    parser.add_argument(
        "--frontend-denoise-min-neighbor-points",
        type=int,
        default=4,
        help="Minimum neighboring points required to keep points in denoise profile.",
    )
    parser.add_argument(
        "--frontend-temporal-pose-file",
        default="",
        help=(
            "Optional pose txt for temporal profile. Supports per-line 12-float KITTI "
            "pose format or 'frame_id + 12 floats'. Poses are interpreted as T_world_lidar."
        ),
    )
    parser.add_argument(
        "--frontend-temporal-allow-missing-pose",
        action="store_true",
        help=(
            "Allow temporal neighbor frames even when pose is missing "
            "(fallback to identity transform)."
        ),
    )
    parser.add_argument(
        "--frontend-adaptive-enable-denoise",
        action="store_true",
        help="Enable guarded denoise in adaptive profile.",
    )
    parser.add_argument(
        "--frontend-adaptive-enable-feature-enhance",
        action="store_true",
        help="Enable distance-adaptive voxel-stat feature enhancement in adaptive profile.",
    )
    parser.add_argument(
        "--frontend-adaptive-max-denoise-drop-ratio",
        type=float,
        default=0.02,
        help=(
            "Max dropped-point ratio allowed for denoise to be applied in adaptive profile."
        ),
    )
    parser.add_argument(
        "--frontend-adaptive-feature-grid-size-xy",
        type=float,
        default=0.8,
        help="Base XY voxel size (meters) for distance-adaptive feature enhancement.",
    )
    parser.add_argument(
        "--frontend-adaptive-feature-min-relative-height",
        type=float,
        default=0.2,
        help="Minimum relative local height (meters) to start foreground boosting.",
    )
    parser.add_argument(
        "--frontend-adaptive-feature-relative-height-scale",
        type=float,
        default=1.2,
        help="Relative height scale (meters) for foreground boosting saturation.",
    )
    parser.add_argument(
        "--frontend-adaptive-feature-boost-strength",
        type=float,
        default=0.2,
        help="Foreground intensity boost strength for feature enhancement.",
    )
    parser.add_argument(
        "--frontend-adaptive-feature-ground-suppress-strength",
        type=float,
        default=0.03,
        help="Ground/background intensity suppression strength for feature enhancement.",
    )
    parser.add_argument(
        "--frontend-adaptive-feature-max-adjust-ratio",
        type=float,
        default=0.35,
        help=(
            "If enhanced-intensity adjusted-point ratio exceeds this threshold, "
            "feature enhancement is skipped."
        ),
    )
    parser.add_argument(
        "--frontend-adaptive-intensity-lower-percentile",
        type=float,
        default=1.0,
        help="Lower percentile used by adaptive intensity auto-repair.",
    )
    parser.add_argument(
        "--frontend-adaptive-intensity-upper-percentile",
        type=float,
        default=99.0,
        help="Upper percentile used by adaptive intensity auto-repair.",
    )
    parser.add_argument(
        "--frontend-adaptive-intensity-trigger-low",
        type=float,
        default=-0.1,
        help="Adaptive intensity repair trigger: lower bound of expected intensity range.",
    )
    parser.add_argument(
        "--frontend-adaptive-intensity-trigger-high",
        type=float,
        default=1.5,
        help="Adaptive intensity repair trigger: upper bound of expected intensity range.",
    )
    return parser.parse_args()


def normalize_angle(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def load_kitti_calibration(calib_path: Path) -> Dict[str, np.ndarray]:
    values: Dict[str, np.ndarray] = {}
    with calib_path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text or ":" not in text:
                continue
            key, raw_data = text.split(":", 1)
            nums = np.array([float(x) for x in raw_data.strip().split()], dtype=np.float32)
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

    if "P2" not in values:
        raise KeyError(f"Missing P2 in {calib_path}")

    return {
        "Tr_velo_to_cam": tr,
        "R0_rect": r0,
        "P2": values["P2"].reshape(3, 4),
    }


def resolve_calib_file(calib_dir: Path, frame_stem: str) -> Path:
    candidates = [calib_dir / f"{frame_stem}.txt"]
    if frame_stem.isdigit():
        frame_idx = int(frame_stem)
        candidates.append(calib_dir / f"{frame_idx:06d}.txt")
    if len(frame_stem) >= 6 and frame_stem[-6:].isdigit():
        candidates.append(calib_dir / f"{frame_stem[-6:]}.txt")

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Missing calibration file for frame {frame_stem}. Tried: "
        + ", ".join(str(p) for p in candidates)
    )


def lidar_to_camera(points_lidar: np.ndarray, calib: Dict[str, np.ndarray]) -> np.ndarray:
    tr = calib["Tr_velo_to_cam"]
    r0 = calib["R0_rect"]
    cam = points_lidar @ tr[:, :3].T + tr[:, 3]
    cam = cam @ r0.T
    return cam.astype(np.float32, copy=False)


def project_points_to_image(points_cam: np.ndarray, p2: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    if points_cam.size == 0:
        return None

    points_h = np.column_stack((points_cam, np.ones((points_cam.shape[0],), dtype=np.float32)))
    proj = points_h @ p2.T
    valid = proj[:, 2] > 1e-3
    if int(np.sum(valid)) < 1:
        return None

    uv = proj[valid, :2] / proj[valid, 2:3]
    return (
        float(np.min(uv[:, 0])),
        float(np.min(uv[:, 1])),
        float(np.max(uv[:, 0])),
        float(np.max(uv[:, 1])),
    )


def corners3d_kitti_camera(h: float, w: float, l: float, x: float, y: float, z: float, ry: float) -> np.ndarray:
    x_corners = np.array([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2], dtype=np.float32)
    y_corners = np.array([0, 0, 0, 0, -h, -h, -h, -h], dtype=np.float32)
    z_corners = np.array([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], dtype=np.float32)

    c = math.cos(ry)
    s = math.sin(ry)
    rot_y = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)

    corners = np.vstack((x_corners, y_corners, z_corners))
    corners = rot_y @ corners
    corners = corners + np.array([[x], [y], [z]], dtype=np.float32)
    return corners.T


def box_lidar_to_kitti_annotation(box_lidar: np.ndarray, calib: Dict[str, np.ndarray]) -> Optional[Dict[str, float]]:
    if box_lidar.shape[0] < 7:
        return None

    x_l, y_l, z_l, dx, dy, dz, heading = [float(v) for v in box_lidar[:7]]

    center_cam = lidar_to_camera(np.array([[x_l, y_l, z_l]], dtype=np.float32), calib)[0]

    h = float(dz)
    w = float(dy)
    l = float(dx)

    x = float(center_cam[0])
    y = float(center_cam[1] + h / 2.0)
    z = float(center_cam[2])

    if z <= 1e-3:
        return None

    ry = normalize_angle(-heading - math.pi / 2.0)
    alpha = normalize_angle(ry - math.atan2(x, z))

    corners = corners3d_kitti_camera(h=h, w=w, l=l, x=x, y=y, z=z, ry=ry)
    bbox = project_points_to_image(corners, calib["P2"])
    if bbox is None:
        return None

    return {
        "alpha": alpha,
        "bbox_left": bbox[0],
        "bbox_top": bbox[1],
        "bbox_right": bbox[2],
        "bbox_bottom": bbox[3],
        "h": h,
        "w": w,
        "l": l,
        "x": x,
        "y": y,
        "z": z,
        "ry": ry,
    }


def format_kitti_line(cls_name: str, anno: Dict[str, float], score: float) -> str:
    return (
        f"{cls_name} "
        f"-1 -1 {anno['alpha']:.4f} "
        f"{anno['bbox_left']:.2f} {anno['bbox_top']:.2f} "
        f"{anno['bbox_right']:.2f} {anno['bbox_bottom']:.2f} "
        f"{anno['h']:.3f} {anno['w']:.3f} {anno['l']:.3f} "
        f"{anno['x']:.3f} {anno['y']:.3f} {anno['z']:.3f} {anno['ry']:.4f} "
        f"{score:.4f}"
    )


def load_data_to_device(batch_dict: Dict[str, object], torch, device) -> None:
    skip_keys = {"frame_id", "metadata", "calib", "image_shape"}
    for key, value in list(batch_dict.items()):
        if key in skip_keys:
            continue
        if isinstance(value, np.ndarray):
            tensor = torch.from_numpy(value)
            if tensor.dtype in (torch.float64, torch.float16):
                tensor = tensor.float()
            batch_dict[key] = tensor.to(device)


def ensure_openpcdet_imports(openpcdet_root: Path):
    if not openpcdet_root.exists():
        raise FileNotFoundError(f"OpenPCDet root not found: {openpcdet_root}")

    root_str = str(openpcdet_root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    try:
        import torch
        from pcdet.config import cfg, cfg_from_yaml_file
        # Import DatasetTemplate directly to avoid optional dataset deps
        # (e.g., Argo2's av2 package) during simple KITTI inference export.
        from pcdet.datasets.dataset import DatasetTemplate
        from pcdet.models import build_network
        from pcdet.utils import common_utils
    except ImportError as exc:
        raise RuntimeError(
            "Failed to import OpenPCDet dependencies. Ensure OpenPCDet is installed "
            "and environment has torch + pcdet available."
        ) from exc

    return torch, cfg, cfg_from_yaml_file, DatasetTemplate, build_network, common_utils


def discover_bin_files(bin_dir: Path, ext: str, limit: int) -> List[Path]:
    files = sorted(bin_dir.glob(f"*{ext}"))
    if limit > 0:
        files = files[:limit]
    return files


def extract_point_cloud_range(dataset_cfg) -> Optional[np.ndarray]:
    value = getattr(dataset_cfg, "POINT_CLOUD_RANGE", None)
    if value is None and isinstance(dataset_cfg, dict):
        value = dataset_cfg.get("POINT_CLOUD_RANGE")
    if value is None:
        return None

    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size != 6:
        raise ValueError(f"POINT_CLOUD_RANGE must have 6 values, got {arr.size}.")
    return arr


def canonical_frame_id(text: str) -> str:
    raw = str(text).strip()
    if raw.isdigit():
        return f"{int(raw):06d}"
    return raw


def parse_pose_file(pose_file: Path) -> Dict[str, np.ndarray]:
    pose_map: Dict[str, np.ndarray] = {}
    with pose_file.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            text = line.strip()
            if not text:
                continue
            parts = text.split()

            frame_id: Optional[str] = None
            data_tokens: Sequence[str] = parts

            if len(parts) in (13, 17):
                frame_id = canonical_frame_id(parts[0])
                data_tokens = parts[1:]

            values = np.asarray([float(v) for v in data_tokens], dtype=np.float32)
            if values.size == 12:
                transform = np.eye(4, dtype=np.float32)
                transform[:3, :4] = values.reshape(3, 4)
            elif values.size == 16:
                transform = values.reshape(4, 4)
            else:
                raise ValueError(
                    f"Invalid pose line {line_idx + 1} in {pose_file}: "
                    f"expected 12/16 values (optionally prefixed by frame id), "
                    f"got {values.size}."
                )

            if frame_id is None:
                frame_id = f"{line_idx:06d}"

            pose_map[frame_id] = transform.astype(np.float32, copy=False)

    return pose_map


def invert_transform(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float32)
    if transform.shape != (4, 4):
        raise ValueError(f"Expected transform shape (4, 4), got {transform.shape}.")
    rot = transform[:3, :3]
    trans = transform[:3, 3]
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = rot.T
    out[:3, 3] = -(rot.T @ trans)
    return out


def make_demo_dataset_class(
    dataset_template_cls,
    file_list: Sequence[Path],
    frontend_profile: str,
    point_cloud_range: Optional[np.ndarray],
    frontend_xyz_clip_abs: float,
    frontend_intensity_clip_abs: float,
    frontend_denoise_voxel_size: float,
    frontend_denoise_neighbor_radius_voxels: int,
    frontend_denoise_min_neighbor_points: int,
    frontend_adaptive_enable_denoise: bool,
    frontend_adaptive_max_denoise_drop_ratio: float,
    frontend_adaptive_enable_feature_enhance: bool,
    frontend_adaptive_feature_grid_size_xy: float,
    frontend_adaptive_feature_min_relative_height: float,
    frontend_adaptive_feature_relative_height_scale: float,
    frontend_adaptive_feature_boost_strength: float,
    frontend_adaptive_feature_ground_suppress_strength: float,
    frontend_adaptive_feature_max_adjust_ratio: float,
    frontend_adaptive_intensity_lower_percentile: float,
    frontend_adaptive_intensity_upper_percentile: float,
    frontend_adaptive_intensity_trigger_low: float,
    frontend_adaptive_intensity_trigger_high: float,
    frontend_temporal_allow_missing_pose: bool,
    temporal_pose_map: Optional[Dict[str, np.ndarray]],
):
    class DemoDataset(dataset_template_cls):
        def __init__(self, dataset_cfg, class_names, training, root_path, logger):
            super().__init__(
                dataset_cfg=dataset_cfg,
                class_names=class_names,
                training=training,
                root_path=root_path,
                logger=logger,
            )
            self.sample_file_list = list(file_list)
            self.frontend_profile = str(frontend_profile).lower()
            self.frontend_xyz_clip_abs = float(frontend_xyz_clip_abs)
            self.frontend_intensity_clip_abs = float(frontend_intensity_clip_abs)
            self.frontend_denoise_voxel_size = float(frontend_denoise_voxel_size)
            self.frontend_denoise_neighbor_radius_voxels = int(
                frontend_denoise_neighbor_radius_voxels
            )
            self.frontend_denoise_min_neighbor_points = int(
                frontend_denoise_min_neighbor_points
            )
            self.frontend_adaptive_enable_denoise = bool(frontend_adaptive_enable_denoise)
            self.frontend_adaptive_max_denoise_drop_ratio = float(
                frontend_adaptive_max_denoise_drop_ratio
            )
            self.frontend_adaptive_enable_feature_enhance = bool(
                frontend_adaptive_enable_feature_enhance
            )
            self.frontend_adaptive_feature_grid_size_xy = float(
                frontend_adaptive_feature_grid_size_xy
            )
            self.frontend_adaptive_feature_min_relative_height = float(
                frontend_adaptive_feature_min_relative_height
            )
            self.frontend_adaptive_feature_relative_height_scale = float(
                frontend_adaptive_feature_relative_height_scale
            )
            self.frontend_adaptive_feature_boost_strength = float(
                frontend_adaptive_feature_boost_strength
            )
            self.frontend_adaptive_feature_ground_suppress_strength = float(
                frontend_adaptive_feature_ground_suppress_strength
            )
            self.frontend_adaptive_feature_max_adjust_ratio = float(
                frontend_adaptive_feature_max_adjust_ratio
            )
            self.frontend_adaptive_intensity_lower_percentile = float(
                frontend_adaptive_intensity_lower_percentile
            )
            self.frontend_adaptive_intensity_upper_percentile = float(
                frontend_adaptive_intensity_upper_percentile
            )
            self.frontend_adaptive_intensity_trigger_low = float(
                frontend_adaptive_intensity_trigger_low
            )
            self.frontend_adaptive_intensity_trigger_high = float(
                frontend_adaptive_intensity_trigger_high
            )
            self.temporal_profile = self.frontend_profile == "temporal_3frame"
            self.temporal_offsets = (-1, 0, 1) if self.temporal_profile else (0,)
            self.frontend_temporal_allow_missing_pose = bool(
                frontend_temporal_allow_missing_pose
            )
            self.temporal_pose_map = temporal_pose_map or {}

            hook_profile = self.frontend_profile
            if self.temporal_profile:
                hook_profile = "adaptive"
            self.frontend_hook = OpenPCDetFrontendHook(
                profile=hook_profile,
                point_cloud_range=point_cloud_range,
                xyz_clip_abs=self.frontend_xyz_clip_abs,
                intensity_clip_abs=self.frontend_intensity_clip_abs,
                denoise_voxel_size=self.frontend_denoise_voxel_size,
                denoise_neighbor_radius_voxels=self.frontend_denoise_neighbor_radius_voxels,
                denoise_min_neighbor_points=self.frontend_denoise_min_neighbor_points,
                adaptive_enable_denoise=self.frontend_adaptive_enable_denoise,
                adaptive_max_denoise_drop_ratio=self.frontend_adaptive_max_denoise_drop_ratio,
                adaptive_enable_feature_enhance=self.frontend_adaptive_enable_feature_enhance,
                adaptive_feature_grid_size_xy=self.frontend_adaptive_feature_grid_size_xy,
                adaptive_feature_min_relative_height=self.frontend_adaptive_feature_min_relative_height,
                adaptive_feature_relative_height_scale=self.frontend_adaptive_feature_relative_height_scale,
                adaptive_feature_boost_strength=self.frontend_adaptive_feature_boost_strength,
                adaptive_feature_ground_suppress_strength=self.frontend_adaptive_feature_ground_suppress_strength,
                adaptive_feature_max_adjust_ratio=self.frontend_adaptive_feature_max_adjust_ratio,
                adaptive_intensity_lower_percentile=self.frontend_adaptive_intensity_lower_percentile,
                adaptive_intensity_upper_percentile=self.frontend_adaptive_intensity_upper_percentile,
                adaptive_intensity_trigger_low=self.frontend_adaptive_intensity_trigger_low,
                adaptive_intensity_trigger_high=self.frontend_adaptive_intensity_trigger_high,
            )
            self.frontend_stats: Dict[int, Dict[str, int]] = {}

        def __len__(self):
            return len(self.sample_file_list)

        def _identity_transform(self) -> np.ndarray:
            return np.eye(4, dtype=np.float32)

        def _get_pose(self, frame_id: str) -> Optional[np.ndarray]:
            key = canonical_frame_id(frame_id)
            pose = self.temporal_pose_map.get(key)
            if pose is not None:
                return pose
            return self.temporal_pose_map.get(str(frame_id))

        def _transform_source_to_anchor(
            self, source_frame_id: str, anchor_frame_id: str
        ) -> Tuple[np.ndarray, bool]:
            source_pose = self._get_pose(source_frame_id)
            anchor_pose = self._get_pose(anchor_frame_id)
            if source_pose is None or anchor_pose is None:
                return self._identity_transform(), False
            return invert_transform(anchor_pose) @ source_pose, True

        def __getitem__(self, index):
            anchor_file = self.sample_file_list[index]
            anchor_frame_id = anchor_file.stem

            stats = make_frontend_stats(include_temporal=True)

            stacked_points: List[np.ndarray] = []
            for offset in self.temporal_offsets:
                src_index = index + int(offset)
                if src_index < 0 or src_index >= len(self.sample_file_list):
                    continue

                src_file = self.sample_file_list[src_index]
                src_points = np.fromfile(str(src_file), dtype=np.float32).reshape(-1, 4)
                src_points, src_stats = self.frontend_hook(src_points)

                merge_frontend_stats(stats, src_stats)
                stats["temporal_frames_used"] += 1

                if offset != 0 and src_points.shape[0] > 0:
                    transform, has_pose = self._transform_source_to_anchor(
                        source_frame_id=src_file.stem,
                        anchor_frame_id=anchor_frame_id,
                    )
                    if not has_pose:
                        stats["temporal_pose_missing_frames"] += 1
                        if not self.frontend_temporal_allow_missing_pose:
                            stats["temporal_frames_skipped_pose"] += 1
                            continue
                    src_points = apply_rigid_transform(src_points, transform)

                if src_points.shape[0] > 0:
                    stacked_points.append(src_points)

            if stacked_points:
                points = np.concatenate(stacked_points, axis=0).astype(np.float32, copy=False)
            else:
                points = np.empty((0, 4), dtype=np.float32)
            stats["num_points_after"] = int(points.shape[0])

            self.frontend_stats[index] = stats
            input_dict = {
                "points": points,
                "frame_id": anchor_frame_id,
            }
            return self.prepare_data(data_dict=input_dict)

        def get_frontend_stats(self, index: int) -> Dict[str, int]:
            return self.frontend_stats.get(index, make_frontend_stats(include_temporal=True))

    return DemoDataset


def main() -> None:
    args = parse_args()

    openpcdet_root = Path(args.openpcdet_root)
    cfg_file = Path(args.cfg_file)
    ckpt_file = Path(args.ckpt)
    bin_dir = Path(args.bin_dir)
    calib_dir = Path(args.calib_dir)
    out_dir = Path(args.out_dir)

    if not cfg_file.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_file}")
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")
    if not bin_dir.exists():
        raise FileNotFoundError(f"BIN directory not found: {bin_dir}")
    if not calib_dir.exists():
        raise FileNotFoundError(f"Calibration directory not found: {calib_dir}")

    files = discover_bin_files(bin_dir, args.ext, args.limit)
    if not files:
        raise FileNotFoundError(f"No {args.ext} files found in {bin_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    torch, cfg, cfg_from_yaml_file, DatasetTemplate, build_network, common_utils = ensure_openpcdet_imports(
        openpcdet_root
    )

    cfg_from_yaml_file(str(cfg_file), cfg)
    class_names = [str(name) for name in cfg.CLASS_NAMES]
    point_cloud_range = extract_point_cloud_range(cfg.DATA_CONFIG)
    if args.frontend_profile in {
        "safe",
        "denoise",
        "adaptive",
        "adaptive_enhance",
        "adaptive_ground",
        "temporal_3frame",
    } and point_cloud_range is None:
        raise ValueError(
            "Frontend profile requires POINT_CLOUD_RANGE in OpenPCDet DATA_CONFIG."
        )

    temporal_pose_map: Optional[Dict[str, np.ndarray]] = None
    temporal_pose_file = str(args.frontend_temporal_pose_file).strip()
    if args.frontend_profile == "temporal_3frame":
        if temporal_pose_file:
            pose_file = Path(temporal_pose_file)
            if not pose_file.exists():
                raise FileNotFoundError(f"Temporal pose file not found: {pose_file}")
            temporal_pose_map = parse_pose_file(pose_file)
            print(f"[openpcdet] temporal poses loaded: {len(temporal_pose_map)} from {pose_file}")
        else:
            print(
                "[openpcdet] temporal_3frame enabled without pose file; "
                "only anchor frame will be used unless "
                "--frontend-temporal-allow-missing-pose is set."
            )

    logger = common_utils.create_logger()
    dataset_class = make_demo_dataset_class(
        DatasetTemplate,
        files,
        frontend_profile=args.frontend_profile,
        point_cloud_range=point_cloud_range,
        frontend_xyz_clip_abs=float(args.frontend_xyz_clip_abs),
        frontend_intensity_clip_abs=float(args.frontend_intensity_clip_abs),
        frontend_denoise_voxel_size=float(args.frontend_denoise_voxel_size),
        frontend_denoise_neighbor_radius_voxels=int(args.frontend_denoise_neighbor_radius_voxels),
        frontend_denoise_min_neighbor_points=int(args.frontend_denoise_min_neighbor_points),
        frontend_adaptive_enable_denoise=bool(args.frontend_adaptive_enable_denoise),
        frontend_adaptive_max_denoise_drop_ratio=float(
            args.frontend_adaptive_max_denoise_drop_ratio
        ),
        frontend_adaptive_enable_feature_enhance=bool(
            args.frontend_adaptive_enable_feature_enhance
            or args.frontend_profile == "adaptive_enhance"
        ),
        frontend_adaptive_feature_grid_size_xy=float(
            args.frontend_adaptive_feature_grid_size_xy
        ),
        frontend_adaptive_feature_min_relative_height=float(
            args.frontend_adaptive_feature_min_relative_height
        ),
        frontend_adaptive_feature_relative_height_scale=float(
            args.frontend_adaptive_feature_relative_height_scale
        ),
        frontend_adaptive_feature_boost_strength=float(
            args.frontend_adaptive_feature_boost_strength
        ),
        frontend_adaptive_feature_ground_suppress_strength=float(
            args.frontend_adaptive_feature_ground_suppress_strength
        ),
        frontend_adaptive_feature_max_adjust_ratio=float(
            args.frontend_adaptive_feature_max_adjust_ratio
        ),
        frontend_adaptive_intensity_lower_percentile=float(
            args.frontend_adaptive_intensity_lower_percentile
        ),
        frontend_adaptive_intensity_upper_percentile=float(
            args.frontend_adaptive_intensity_upper_percentile
        ),
        frontend_adaptive_intensity_trigger_low=float(
            args.frontend_adaptive_intensity_trigger_low
        ),
        frontend_adaptive_intensity_trigger_high=float(
            args.frontend_adaptive_intensity_trigger_high
        ),
        frontend_temporal_allow_missing_pose=bool(args.frontend_temporal_allow_missing_pose),
        temporal_pose_map=temporal_pose_map,
    )
    dataset = dataset_class(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=class_names,
        training=False,
        root_path=bin_dir,
        logger=logger,
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(class_names), dataset=dataset)
    to_cpu = args.device == "cpu"
    model.load_params_from_file(filename=str(ckpt_file), logger=logger, to_cpu=to_cpu)

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no GPU is available.")
        model = model.cuda()

    model.eval()

    summary_frames = []
    total_exported = 0
    frontend_totals = make_frontend_stats(include_temporal=True)

    with torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset[index]
            frame_id = str(sample["frame_id"])
            frontend_stats = dataset.get_frontend_stats(index)
            merge_frontend_stats(frontend_totals, frontend_stats)
            batch_dict = dataset.collate_batch([sample])

            if args.device == "cuda":
                from pcdet.models import load_data_to_gpu

                load_data_to_gpu(batch_dict)
            else:
                load_data_to_device(batch_dict, torch, torch.device("cpu"))

            pred_dicts, _ = model.forward(batch_dict)
            if not pred_dicts:
                pred_boxes = np.empty((0, 7), dtype=np.float32)
                pred_scores = np.empty((0,), dtype=np.float32)
                pred_labels = np.empty((0,), dtype=np.int64)
            else:
                pred = pred_dicts[0]
                pred_boxes = pred["pred_boxes"].detach().cpu().numpy()
                pred_scores = pred["pred_scores"].detach().cpu().numpy()
                pred_labels = pred["pred_labels"].detach().cpu().numpy().astype(np.int64, copy=False)

            calib_path = resolve_calib_file(calib_dir, frame_id)
            calib = load_kitti_calibration(calib_path)

            lines: List[str] = []
            dropped_score = 0
            dropped_geometry = 0

            for det_idx in range(pred_boxes.shape[0]):
                score = float(pred_scores[det_idx])
                if score < args.score_thresh:
                    dropped_score += 1
                    continue

                cls_index = int(pred_labels[det_idx]) - 1
                if cls_index < 0 or cls_index >= len(class_names):
                    continue
                cls_name = class_names[cls_index]

                anno = box_lidar_to_kitti_annotation(pred_boxes[det_idx], calib)
                if anno is None:
                    dropped_geometry += 1
                    continue

                lines.append(format_kitti_line(cls_name, anno, score))

            out_path = out_dir / f"{frame_id}.txt"
            out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

            total_exported += len(lines)
            frame_summary = {
                "frame": frame_id,
                "raw_detections": int(pred_boxes.shape[0]),
                "exported": int(len(lines)),
                "dropped_score": int(dropped_score),
                "dropped_geometry": int(dropped_geometry),
                "frontend_points_in": int(frontend_stats["num_points_in"]),
                "frontend_points_after": int(frontend_stats["num_points_after"]),
                "frontend_dropped_non_finite": int(frontend_stats["dropped_non_finite"]),
                "frontend_dropped_out_of_range": int(frontend_stats["dropped_out_of_range"]),
                "frontend_dropped_denoise": int(frontend_stats["dropped_denoise"]),
                "frontend_intensity_rescaled": int(frontend_stats["intensity_rescaled"]),
                "frontend_adaptive_denoise_applied": int(
                    frontend_stats["adaptive_denoise_applied"]
                ),
                "frontend_adaptive_denoise_skipped": int(
                    frontend_stats["adaptive_denoise_skipped"]
                ),
                "frontend_adaptive_denoise_candidate_drop": int(
                    frontend_stats["adaptive_denoise_candidate_drop"]
                ),
                "frontend_feature_points_adjusted": int(
                    frontend_stats["feature_points_adjusted"]
                ),
                "frontend_feature_points_adjusted_candidate": int(
                    frontend_stats["feature_points_adjusted_candidate"]
                ),
                "frontend_feature_enhance_applied": int(
                    frontend_stats["feature_enhance_applied"]
                ),
                "frontend_feature_enhance_skipped": int(
                    frontend_stats["feature_enhance_skipped"]
                ),
                "frontend_dropped_ground_sparse": int(
                    frontend_stats["dropped_ground_sparse"]
                ),
                "frontend_ground_sparse_candidate_drop": int(
                    frontend_stats["ground_sparse_candidate_drop"]
                ),
                "frontend_ground_sparse_applied": int(
                    frontend_stats["ground_sparse_applied"]
                ),
                "frontend_ground_sparse_skipped": int(
                    frontend_stats["ground_sparse_skipped"]
                ),
                "frontend_temporal_frames_used": int(frontend_stats["temporal_frames_used"]),
                "frontend_temporal_pose_missing_frames": int(
                    frontend_stats["temporal_pose_missing_frames"]
                ),
                "frontend_temporal_frames_skipped_pose": int(
                    frontend_stats["temporal_frames_skipped_pose"]
                ),
            }
            summary_frames.append(frame_summary)
            print(
                f"[openpcdet] frame={index:04d} id={frame_id} "
                f"raw={pred_boxes.shape[0]} exported={len(lines)} "
                f"drop_score={dropped_score} drop_geom={dropped_geometry} "
                f"frontend_in={frontend_stats['num_points_in']} "
                f"frontend_after={frontend_stats['num_points_after']} "
                f"drop_denoise={frontend_stats['dropped_denoise']} "
                f"intensity_rescaled={frontend_stats['intensity_rescaled']} "
                f"feature_adjusted={frontend_stats['feature_points_adjusted']} "
                f"feature_skipped={frontend_stats['feature_enhance_skipped']} "
                f"drop_ground_sparse={frontend_stats['dropped_ground_sparse']} "
                f"temp_frames={frontend_stats['temporal_frames_used']} "
                f"temp_missing_pose={frontend_stats['temporal_pose_missing_frames']} "
                f"temp_skipped_pose={frontend_stats['temporal_frames_skipped_pose']}"
            )

    summary = {
        "num_frames": len(summary_frames),
        "total_exported": int(total_exported),
        "score_thresh": float(args.score_thresh),
        "class_names": class_names,
        "out_dir": str(out_dir),
        "cfg_file": str(cfg_file),
        "ckpt": str(ckpt_file),
        "frontend_profile": args.frontend_profile,
        "frontend_xyz_clip_abs": float(args.frontend_xyz_clip_abs),
        "frontend_intensity_clip_abs": float(args.frontend_intensity_clip_abs),
        "frontend_denoise_voxel_size": float(args.frontend_denoise_voxel_size),
        "frontend_denoise_neighbor_radius_voxels": int(
            args.frontend_denoise_neighbor_radius_voxels
        ),
        "frontend_denoise_min_neighbor_points": int(
            args.frontend_denoise_min_neighbor_points
        ),
        "frontend_adaptive_enable_denoise": bool(args.frontend_adaptive_enable_denoise),
        "frontend_adaptive_enable_feature_enhance": bool(
            args.frontend_adaptive_enable_feature_enhance
            or args.frontend_profile == "adaptive_enhance"
        ),
        "frontend_adaptive_max_denoise_drop_ratio": float(
            args.frontend_adaptive_max_denoise_drop_ratio
        ),
        "frontend_adaptive_feature_grid_size_xy": float(
            args.frontend_adaptive_feature_grid_size_xy
        ),
        "frontend_adaptive_feature_min_relative_height": float(
            args.frontend_adaptive_feature_min_relative_height
        ),
        "frontend_adaptive_feature_relative_height_scale": float(
            args.frontend_adaptive_feature_relative_height_scale
        ),
        "frontend_adaptive_feature_boost_strength": float(
            args.frontend_adaptive_feature_boost_strength
        ),
        "frontend_adaptive_feature_ground_suppress_strength": float(
            args.frontend_adaptive_feature_ground_suppress_strength
        ),
        "frontend_adaptive_feature_max_adjust_ratio": float(
            args.frontend_adaptive_feature_max_adjust_ratio
        ),
        "frontend_adaptive_intensity_lower_percentile": float(
            args.frontend_adaptive_intensity_lower_percentile
        ),
        "frontend_adaptive_intensity_upper_percentile": float(
            args.frontend_adaptive_intensity_upper_percentile
        ),
        "frontend_adaptive_intensity_trigger_low": float(
            args.frontend_adaptive_intensity_trigger_low
        ),
        "frontend_adaptive_intensity_trigger_high": float(
            args.frontend_adaptive_intensity_trigger_high
        ),
        "frontend_temporal_pose_file": temporal_pose_file,
        "frontend_temporal_allow_missing_pose": bool(
            args.frontend_temporal_allow_missing_pose
        ),
        "frontend_totals": frontend_totals,
        "frames": summary_frames,
    }

    summary_path = out_dir.parent / "openpcdet_export_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print(f"Saved KITTI prediction txt files to: {out_dir}")
    print(f"Saved export summary: {summary_path}")


if __name__ == "__main__":
    main()
