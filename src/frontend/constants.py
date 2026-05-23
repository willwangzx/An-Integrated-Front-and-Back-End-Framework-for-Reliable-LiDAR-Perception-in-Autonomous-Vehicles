"""Frontend profile and metrics constants for OpenPCDet integration."""

from typing import Tuple

SUPPORTED_FRONTEND_PROFILES = {
    "none",
    "safe",
    "denoise",
    "adaptive",
    "adaptive_enhance",
    "adaptive_ground",
}

RANGE_REQUIRED_PROFILES = {
    "safe",
    "denoise",
    "adaptive",
    "adaptive_enhance",
    "adaptive_ground",
}

BASE_FRONTEND_STATS_KEYS: Tuple[str, ...] = (
    "num_points_in",
    "dropped_non_finite",
    "dropped_out_of_range",
    "dropped_denoise",
    "intensity_rescaled",
    "adaptive_denoise_applied",
    "adaptive_denoise_skipped",
    "adaptive_denoise_candidate_drop",
    "feature_points_adjusted",
    "feature_points_adjusted_candidate",
    "feature_enhance_applied",
    "feature_enhance_skipped",
    "dropped_ground_sparse",
    "ground_sparse_candidate_drop",
    "ground_sparse_applied",
    "ground_sparse_skipped",
    "num_points_after",
)

TEMPORAL_FRONTEND_STATS_KEYS: Tuple[str, ...] = (
    "temporal_frames_used",
    "temporal_pose_missing_frames",
    "temporal_frames_skipped_pose",
)
