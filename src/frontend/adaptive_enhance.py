"""Distance-adaptive voxel-stat feature enhancement for point cloud intensity."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class DistanceAdaptiveVoxelSpec:
    """Defines near/mid/far voxel scales used in adaptive enhancement."""

    near_range_m: float = 20.0
    far_range_m: float = 45.0
    near_xy_scale: float = 0.6
    mid_xy_scale: float = 1.0
    far_xy_scale: float = 1.8
    near_z_scale: float = 0.6
    mid_z_scale: float = 0.9
    far_z_scale: float = 1.3
    min_voxel_size_m: float = 0.05

    def sizes(self, base_grid_size_xy: float) -> Tuple[np.ndarray, np.ndarray]:
        base = float(base_grid_size_xy)
        min_size = float(self.min_voxel_size_m)
        xy_sizes = np.asarray(
            [
                max(base * self.near_xy_scale, min_size),
                max(base * self.mid_xy_scale, min_size),
                max(base * self.far_xy_scale, min_size),
            ],
            dtype=np.float32,
        )
        z_sizes = np.asarray(
            [
                max(base * self.near_z_scale, min_size),
                max(base * self.mid_z_scale, min_size),
                max(base * self.far_z_scale, min_size),
            ],
            dtype=np.float32,
        )
        return xy_sizes, z_sizes


def build_distance_adaptive_indices(
    points_xyz: np.ndarray,
    base_grid_size_xy: float,
    voxel_spec: DistanceAdaptiveVoxelSpec,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if points_xyz.ndim != 2 or points_xyz.shape[1] < 3:
        raise ValueError(f"Expected xyz shape (N, >=3), got {points_xyz.shape}.")

    xy_sizes, z_sizes = voxel_spec.sizes(base_grid_size_xy)

    radius = np.linalg.norm(points_xyz[:, :2], axis=1)
    range_bin = np.zeros((points_xyz.shape[0],), dtype=np.int32)
    range_bin[radius >= float(voxel_spec.near_range_m)] = 1
    range_bin[radius >= float(voxel_spec.far_range_m)] = 2

    xy_cell_idx = np.zeros((points_xyz.shape[0], 3), dtype=np.int32)
    xy_cell_idx[:, 0] = range_bin
    voxel_idx = np.zeros((points_xyz.shape[0], 4), dtype=np.int32)
    voxel_idx[:, 0] = range_bin

    for b in (0, 1, 2):
        mask = range_bin == b
        if not np.any(mask):
            continue
        xy_size = float(xy_sizes[b])
        z_size = float(z_sizes[b])
        x_idx = np.floor(points_xyz[mask, 0] / xy_size).astype(np.int32, copy=False)
        y_idx = np.floor(points_xyz[mask, 1] / xy_size).astype(np.int32, copy=False)
        z_idx = np.floor(points_xyz[mask, 2] / z_size).astype(np.int32, copy=False)
        voxel_idx[mask, 1] = x_idx
        voxel_idx[mask, 2] = y_idx
        voxel_idx[mask, 3] = z_idx
        xy_cell_idx[mask, 1] = x_idx
        xy_cell_idx[mask, 2] = y_idx

    return voxel_idx, xy_cell_idx, range_bin, xy_sizes, z_sizes


def enhance_object_saliency_by_local_height(
    points: np.ndarray,
    grid_size_xy: float = 0.8,
    min_relative_height: float = 0.2,
    relative_height_scale: float = 1.2,
    boost_strength: float = 0.2,
    ground_suppress_strength: float = 0.03,
    voxel_spec: DistanceAdaptiveVoxelSpec = DistanceAdaptiveVoxelSpec(),
) -> Tuple[np.ndarray, int]:
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected points shape (N, >=3), got {points.shape}.")

    if points.shape[0] == 0 or points.shape[1] < 4:
        return points.astype(np.float32, copy=False), 0

    grid_size_xy = float(grid_size_xy)
    min_relative_height = float(min_relative_height)
    relative_height_scale = float(relative_height_scale)
    boost_strength = float(boost_strength)
    ground_suppress_strength = float(ground_suppress_strength)

    if grid_size_xy <= 0:
        raise ValueError(f"grid_size_xy must be > 0, got {grid_size_xy}.")
    if relative_height_scale <= 0:
        raise ValueError(
            f"relative_height_scale must be > 0, got {relative_height_scale}."
        )
    if boost_strength < 0:
        raise ValueError(f"boost_strength must be >= 0, got {boost_strength}.")
    if ground_suppress_strength < 0:
        raise ValueError(
            "ground_suppress_strength must be >= 0, "
            f"got {ground_suppress_strength}."
        )

    out = points.astype(np.float32, copy=True)

    voxel_idx, xy_cell_idx, range_bin, xy_sizes, z_sizes = build_distance_adaptive_indices(
        points_xyz=out[:, :3],
        base_grid_size_xy=grid_size_xy,
        voxel_spec=voxel_spec,
    )

    radius = np.linalg.norm(out[:, :2], axis=1)
    _, xy_inverse = np.unique(xy_cell_idx, axis=0, return_inverse=True)
    num_xy_cells = int(np.max(xy_inverse)) + 1

    uniq_voxels, inverse, voxel_counts = np.unique(
        voxel_idx,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    num_voxels = int(uniq_voxels.shape[0])

    # XY stats encode local ground context.
    ground_z = np.full((num_xy_cells,), np.inf, dtype=np.float32)
    ceil_z = np.full((num_xy_cells,), -np.inf, dtype=np.float32)
    np.minimum.at(ground_z, xy_inverse, out[:, 2])
    np.maximum.at(ceil_z, xy_inverse, out[:, 2])
    xy_counts = np.bincount(xy_inverse, minlength=num_xy_cells).astype(np.float32, copy=False)

    # 3D voxel stats encode density and reflectivity structure.
    intensity = out[:, 3]
    intensity_sum = np.zeros((num_voxels,), dtype=np.float32)
    intensity_sq_sum = np.zeros((num_voxels,), dtype=np.float32)
    np.add.at(intensity_sum, inverse, intensity)
    np.add.at(intensity_sq_sum, inverse, intensity * intensity)

    voxel_counts_f = voxel_counts.astype(np.float32, copy=False)
    intensity_mean = intensity_sum / np.maximum(voxel_counts_f, 1.0)
    intensity_var = np.maximum(
        intensity_sq_sum / np.maximum(voxel_counts_f, 1.0) - intensity_mean * intensity_mean,
        0.0,
    )
    intensity_std = np.sqrt(intensity_var)

    voxel_volume_lut = (xy_sizes * xy_sizes) * z_sizes
    voxel_volume = voxel_volume_lut[uniq_voxels[:, 0]]
    voxel_density = voxel_counts_f / np.maximum(voxel_volume, 1e-3)

    relative_height = out[:, 2] - ground_z[xy_inverse]
    local_height_span = np.maximum(ceil_z[xy_inverse] - ground_z[xy_inverse], 1e-3)
    xy_counts_point = xy_counts[xy_inverse]
    xy_area_point = np.maximum(xy_sizes[range_bin] * xy_sizes[range_bin], 1e-3)
    xy_density = xy_counts_point / xy_area_point
    height_score = np.clip(
        (relative_height - min_relative_height) / relative_height_scale,
        0.0,
        1.0,
    )
    span_score = np.clip(
        local_height_span / max(min_relative_height + relative_height_scale, 1e-3),
        0.0,
        1.0,
    )
    density_score = np.clip(
        np.log1p(voxel_density[inverse]) / np.log1p(10.0),
        0.0,
        1.0,
    )
    xy_density_score = np.clip(
        np.log1p(xy_density) / np.log1p(40.0),
        0.0,
        1.0,
    )
    peak_score = np.clip(
        (intensity - intensity_mean[inverse]) / (intensity_std[inverse] + 0.05),
        -1.0,
        2.0,
    )
    peak_score = np.clip((peak_score + 1.0) / 3.0, 0.0, 1.0)

    # Sparse/tall/far branch protects small-object cues.
    sparse_score = 1.0 - np.clip((xy_counts_point - 6.0) / 22.0, 0.0, 1.0)
    far_score = np.clip((radius - 18.0) / 30.0, 0.0, 1.0)
    small_object_preserve = np.clip(
        0.55 * sparse_score + 0.30 * (sparse_score * height_score) + 0.15 * far_score,
        0.0,
        1.0,
    )

    foreground_score = (
        0.60 * height_score
        + 0.20 * (height_score * span_score)
        + 0.15 * peak_score
        + 0.05 * (height_score * (0.5 * density_score + 0.5 * xy_density_score))
    )
    background_score = np.clip(
        (1.0 - height_score)
        * (0.75 + 0.25 * (1.0 - peak_score))
        * (0.7 + 0.3 * (1.0 - span_score)),
        0.0,
        1.2,
    )

    confidence = 1.0 / (1.0 + 1.5 * intensity_std[inverse])
    delta = (
        boost_strength * foreground_score - ground_suppress_strength * background_score
    ) * confidence

    delta_small = (
        (0.45 * boost_strength) * (0.75 * height_score + 0.25 * peak_score)
        - (0.20 * ground_suppress_strength) * (1.0 - height_score)
    ) * confidence
    branch_alpha = np.clip(1.0 - small_object_preserve, 0.25, 1.0)
    delta_blend = branch_alpha * delta + (1.0 - branch_alpha) * delta_small

    small_mask = small_object_preserve > 0.6
    if np.any(small_mask):
        delta_blend[small_mask] = np.maximum(delta_blend[small_mask], -0.01)

    alpha = np.clip(0.95 - 0.55 * small_object_preserve, 0.35, 1.0)
    enhanced_intensity = np.clip(intensity + alpha * delta_blend, 0.0, 1.0)
    meaningful_change_eps = max(
        0.01,
        0.08 * boost_strength,
        0.5 * ground_suppress_strength,
    )
    changed = int(np.sum(np.abs(enhanced_intensity - intensity) > meaningful_change_eps))
    out[:, 3] = enhanced_intensity
    return out.astype(np.float32, copy=False), changed
