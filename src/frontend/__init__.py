"""Modular frontend processing package for OpenPCDet integration."""

from .adaptive_enhance import (
    DistanceAdaptiveVoxelSpec,
    build_distance_adaptive_indices,
    enhance_object_saliency_by_local_height,
)
from .constants import (
    BASE_FRONTEND_STATS_KEYS,
    RANGE_REQUIRED_PROFILES,
    SUPPORTED_FRONTEND_PROFILES,
    TEMPORAL_FRONTEND_STATS_KEYS,
)
from .denoise import (
    denoise_points_by_voxel_density,
    denoise_sparse_near_ground_points,
    sparse_point_mask_by_voxel_density,
)
from .hook import OpenPCDetFrontendHook
from .preprocess import apply_rigid_transform, robust_rescale_intensity, safe_preprocess_points
from .stats import make_frontend_stats, merge_frontend_stats

__all__ = [
    "DistanceAdaptiveVoxelSpec",
    "build_distance_adaptive_indices",
    "enhance_object_saliency_by_local_height",
    "BASE_FRONTEND_STATS_KEYS",
    "TEMPORAL_FRONTEND_STATS_KEYS",
    "SUPPORTED_FRONTEND_PROFILES",
    "RANGE_REQUIRED_PROFILES",
    "denoise_points_by_voxel_density",
    "denoise_sparse_near_ground_points",
    "sparse_point_mask_by_voxel_density",
    "OpenPCDetFrontendHook",
    "apply_rigid_transform",
    "robust_rescale_intensity",
    "safe_preprocess_points",
    "make_frontend_stats",
    "merge_frontend_stats",
]
