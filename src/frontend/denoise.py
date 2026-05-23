"""Voxel-density denoise utilities."""

from typing import Tuple

import numpy as np


def sparse_point_mask_by_voxel_density(
    points: np.ndarray,
    voxel_size: float,
    neighbor_radius_voxels: int,
    min_neighbor_points: int,
) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected points shape (N, >=3), got {points.shape}.")
    if points.shape[0] == 0:
        return np.zeros((0,), dtype=bool)

    voxel_size = float(voxel_size)
    neighbor_radius_voxels = int(neighbor_radius_voxels)
    min_neighbor_points = int(min_neighbor_points)

    if voxel_size <= 0:
        raise ValueError(f"voxel_size must be > 0, got {voxel_size}.")
    if neighbor_radius_voxels < 0:
        raise ValueError(
            f"neighbor_radius_voxels must be >= 0, got {neighbor_radius_voxels}."
        )
    if min_neighbor_points <= 1:
        return np.zeros((points.shape[0],), dtype=bool)

    voxel_idx = np.floor(points[:, :3] / voxel_size).astype(np.int32, copy=False)
    uniq_voxels, inverse, voxel_counts = np.unique(
        voxel_idx,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )

    voxel_to_id = {tuple(v.tolist()): i for i, v in enumerate(uniq_voxels)}
    neighborhood_counts = np.zeros((uniq_voxels.shape[0],), dtype=np.int32)

    offsets = range(-neighbor_radius_voxels, neighbor_radius_voxels + 1)
    for i, voxel in enumerate(uniq_voxels):
        vx, vy, vz = int(voxel[0]), int(voxel[1]), int(voxel[2])
        total = 0
        for dx in offsets:
            for dy in offsets:
                for dz in offsets:
                    neighbor_id = voxel_to_id.get((vx + dx, vy + dy, vz + dz))
                    if neighbor_id is not None:
                        total += int(voxel_counts[neighbor_id])
        neighborhood_counts[i] = total

    keep_voxel_mask = neighborhood_counts >= min_neighbor_points
    keep_mask = keep_voxel_mask[inverse]
    return (~keep_mask).astype(bool, copy=False)


def denoise_sparse_near_ground_points(
    points: np.ndarray,
    voxel_size: float,
    neighbor_radius_voxels: int,
    min_neighbor_points: int,
    ground_grid_size_xy: float,
    max_relative_height: float,
) -> Tuple[np.ndarray, int]:
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected points shape (N, >=3), got {points.shape}.")
    if points.shape[0] == 0:
        return points.astype(np.float32, copy=False), 0

    ground_grid_size_xy = float(ground_grid_size_xy)
    max_relative_height = float(max_relative_height)
    if ground_grid_size_xy <= 0:
        raise ValueError(
            f"ground_grid_size_xy must be > 0, got {ground_grid_size_xy}."
        )

    sparse_mask = sparse_point_mask_by_voxel_density(
        points=points,
        voxel_size=voxel_size,
        neighbor_radius_voxels=neighbor_radius_voxels,
        min_neighbor_points=min_neighbor_points,
    )

    xy_idx = np.floor(points[:, :2] / ground_grid_size_xy).astype(np.int32, copy=False)
    uniq_xy, inverse = np.unique(xy_idx, axis=0, return_inverse=True, return_counts=False)
    num_cells = int(np.max(inverse)) + 1
    ground_z = np.full((num_cells,), np.inf, dtype=np.float32)
    np.minimum.at(ground_z, inverse, points[:, 2])
    cell_to_id = {tuple(v.tolist()): i for i, v in enumerate(uniq_xy)}
    ref_ground = ground_z.copy()
    has_neighbor_ref = np.zeros((num_cells,), dtype=bool)
    for i, cell in enumerate(uniq_xy):
        cx, cy = int(cell[0]), int(cell[1])
        best = np.inf
        found = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nid = cell_to_id.get((cx + dx, cy + dy))
                if nid is not None:
                    found = True
                    if ground_z[nid] < best:
                        best = ground_z[nid]
        if found:
            has_neighbor_ref[i] = True
            ref_ground[i] = best

    relative_height = points[:, 2] - ref_ground[inverse]
    near_ground_mask = (relative_height <= max_relative_height) & has_neighbor_ref[inverse]

    drop_mask = sparse_mask & near_ground_mask
    filtered = points[~drop_mask]
    dropped = int(np.sum(drop_mask))
    return filtered.astype(np.float32, copy=False), dropped


def denoise_points_by_voxel_density(
    points: np.ndarray,
    voxel_size: float,
    neighbor_radius_voxels: int,
    min_neighbor_points: int,
) -> Tuple[np.ndarray, int]:
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected points shape (N, >=3), got {points.shape}.")

    if points.shape[0] == 0:
        return points.astype(np.float32, copy=False), 0

    voxel_size = float(voxel_size)
    neighbor_radius_voxels = int(neighbor_radius_voxels)
    min_neighbor_points = int(min_neighbor_points)

    if voxel_size <= 0:
        raise ValueError(f"voxel_size must be > 0, got {voxel_size}.")
    if neighbor_radius_voxels < 0:
        raise ValueError(
            f"neighbor_radius_voxels must be >= 0, got {neighbor_radius_voxels}."
        )
    if min_neighbor_points <= 1:
        return points.astype(np.float32, copy=False), 0

    voxel_idx = np.floor(points[:, :3] / voxel_size).astype(np.int32, copy=False)
    uniq_voxels, inverse, voxel_counts = np.unique(
        voxel_idx,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )

    voxel_to_id = {tuple(v.tolist()): i for i, v in enumerate(uniq_voxels)}
    neighborhood_counts = np.zeros((uniq_voxels.shape[0],), dtype=np.int32)

    offsets = range(-neighbor_radius_voxels, neighbor_radius_voxels + 1)
    for i, voxel in enumerate(uniq_voxels):
        vx, vy, vz = int(voxel[0]), int(voxel[1]), int(voxel[2])
        total = 0
        for dx in offsets:
            for dy in offsets:
                for dz in offsets:
                    neighbor_id = voxel_to_id.get((vx + dx, vy + dy, vz + dz))
                    if neighbor_id is not None:
                        total += int(voxel_counts[neighbor_id])
        neighborhood_counts[i] = total

    keep_voxel_mask = neighborhood_counts >= min_neighbor_points
    keep_mask = keep_voxel_mask[inverse]
    filtered = points[keep_mask]
    dropped = int(points.shape[0] - filtered.shape[0])
    return filtered.astype(np.float32, copy=False), dropped
