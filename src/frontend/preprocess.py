"""Frontend preprocessing utilities."""

from typing import Dict, Tuple

import numpy as np


def safe_preprocess_points(
    points: np.ndarray,
    point_cloud_range: np.ndarray,
    xyz_clip_abs: float,
    intensity_clip_abs: float,
) -> Tuple[np.ndarray, Dict[str, int]]:
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected points shape (N, >=3), got {points.shape}.")

    point_cloud_range = np.asarray(point_cloud_range, dtype=np.float32).reshape(-1)
    if point_cloud_range.size != 6:
        raise ValueError(
            f"point_cloud_range must have 6 values, got {point_cloud_range.size}."
        )

    stats = {
        "num_points_in": int(points.shape[0]),
        "dropped_non_finite": 0,
        "dropped_out_of_range": 0,
        "num_points_after": 0,
    }

    finite_mask = np.isfinite(points).all(axis=1)
    filtered = points[finite_mask]
    stats["dropped_non_finite"] = int(points.shape[0] - filtered.shape[0])
    if filtered.size == 0:
        return filtered.astype(np.float32, copy=False), stats

    filtered = filtered.astype(np.float32, copy=True)
    if xyz_clip_abs > 0:
        np.clip(filtered[:, :3], -xyz_clip_abs, xyz_clip_abs, out=filtered[:, :3])
    if filtered.shape[1] > 3 and intensity_clip_abs > 0:
        np.clip(filtered[:, 3], -intensity_clip_abs, intensity_clip_abs, out=filtered[:, 3])

    x_min, y_min, z_min, x_max, y_max, z_max = [float(v) for v in point_cloud_range]
    range_mask = (
        (filtered[:, 0] >= x_min)
        & (filtered[:, 0] <= x_max)
        & (filtered[:, 1] >= y_min)
        & (filtered[:, 1] <= y_max)
        & (filtered[:, 2] >= z_min)
        & (filtered[:, 2] <= z_max)
    )
    ranged = filtered[range_mask]
    stats["dropped_out_of_range"] = int(filtered.shape[0] - ranged.shape[0])
    stats["num_points_after"] = int(ranged.shape[0])
    return ranged.astype(np.float32, copy=False), stats


def robust_rescale_intensity(
    points: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    trigger_low: float = -0.1,
    trigger_high: float = 1.5,
) -> Tuple[np.ndarray, int]:
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected points shape (N, >=3), got {points.shape}.")

    if points.shape[0] == 0 or points.shape[1] < 4:
        return points.astype(np.float32, copy=False), 0

    lo_pct = float(lower_percentile)
    hi_pct = float(upper_percentile)
    if not (0.0 <= lo_pct <= 100.0 and 0.0 <= hi_pct <= 100.0 and lo_pct < hi_pct):
        raise ValueError(
            "Intensity percentiles must satisfy 0 <= low < high <= 100, "
            f"got low={lo_pct}, high={hi_pct}."
        )

    out = points.astype(np.float32, copy=True)
    intensity = out[:, 3]
    p_lo = float(np.percentile(intensity, lo_pct))
    p_hi = float(np.percentile(intensity, hi_pct))

    if p_lo >= trigger_low and p_hi <= trigger_high:
        return out.astype(np.float32, copy=False), 0

    scale = p_hi - p_lo
    if scale > 1e-6:
        out[:, 3] = (intensity - p_lo) / scale
    else:
        out[:, 3] = intensity
    np.clip(out[:, 3], 0.0, 1.0, out=out[:, 3])
    return out.astype(np.float32, copy=False), 1


def apply_rigid_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected points shape (N, >=3), got {points.shape}.")

    transform = np.asarray(transform, dtype=np.float32)
    if transform.shape != (4, 4):
        raise ValueError(f"Expected transform shape (4, 4), got {transform.shape}.")

    if points.shape[0] == 0:
        return points.astype(np.float32, copy=False)

    out = points.astype(np.float32, copy=True)
    rot = transform[:3, :3]
    trans = transform[:3, 3]
    out[:, :3] = out[:, :3] @ rot.T + trans
    return out
