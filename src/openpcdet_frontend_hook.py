"""Backward-compatible frontend hook exports.

The implementation has been refactored into ``src/frontend`` to better support
research iteration on distance-adaptive voxelization and multi-signal features.
This module is kept as a stable compatibility layer for existing imports.
"""

from src.frontend import (
    OpenPCDetFrontendHook,
    apply_rigid_transform,
    denoise_points_by_voxel_density,
    denoise_sparse_near_ground_points,
    enhance_object_saliency_by_local_height,
    robust_rescale_intensity,
    safe_preprocess_points,
    sparse_point_mask_by_voxel_density,
)

__all__ = [
    "safe_preprocess_points",
    "robust_rescale_intensity",
    "enhance_object_saliency_by_local_height",
    "sparse_point_mask_by_voxel_density",
    "denoise_sparse_near_ground_points",
    "denoise_points_by_voxel_density",
    "apply_rigid_transform",
    "OpenPCDetFrontendHook",
]
