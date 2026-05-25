"""OpenPCDet frontend hook with adaptive voxel-stat enhancement profiles."""

from typing import Dict, Optional, Tuple

import numpy as np

from .adaptive_enhance import enhance_object_saliency_by_local_height
from .constants import RANGE_REQUIRED_PROFILES, SUPPORTED_FRONTEND_PROFILES
from .denoise import denoise_points_by_voxel_density, denoise_sparse_near_ground_points
from .preprocess import robust_rescale_intensity, safe_preprocess_points
from .stats import make_frontend_stats


class OpenPCDetFrontendHook:
    """Conservative frontend hook for OpenPCDet data loading."""

    def __init__(
        self,
        profile: str,
        point_cloud_range: Optional[np.ndarray],
        xyz_clip_abs: float,
        intensity_clip_abs: float,
        denoise_voxel_size: float = 0.25,
        denoise_neighbor_radius_voxels: int = 1,
        denoise_min_neighbor_points: int = 4,
        adaptive_enable_denoise: bool = False,
        adaptive_max_denoise_drop_ratio: float = 0.02,
        adaptive_enable_feature_enhance: bool = False,
        adaptive_feature_grid_size_xy: float = 0.8,
        adaptive_feature_min_relative_height: float = 0.2,
        adaptive_feature_relative_height_scale: float = 1.2,
        adaptive_feature_boost_strength: float = 0.2,
        adaptive_feature_ground_suppress_strength: float = 0.03,
        adaptive_feature_max_adjust_ratio: float = 0.35,
        adaptive_intensity_lower_percentile: float = 1.0,
        adaptive_intensity_upper_percentile: float = 99.0,
        adaptive_intensity_trigger_low: float = -0.1,
        adaptive_intensity_trigger_high: float = 1.5,
    ):
        self.profile = str(profile).lower()
        self.point_cloud_range = (
            None
            if point_cloud_range is None
            else np.asarray(point_cloud_range, dtype=np.float32).reshape(-1)
        )
        self.xyz_clip_abs = float(xyz_clip_abs)
        self.intensity_clip_abs = float(intensity_clip_abs)
        self.denoise_voxel_size = float(denoise_voxel_size)
        self.denoise_neighbor_radius_voxels = int(denoise_neighbor_radius_voxels)
        self.denoise_min_neighbor_points = int(denoise_min_neighbor_points)
        self.adaptive_enable_denoise = bool(adaptive_enable_denoise)
        self.adaptive_max_denoise_drop_ratio = float(adaptive_max_denoise_drop_ratio)
        self.adaptive_enable_feature_enhance = bool(adaptive_enable_feature_enhance)
        self.adaptive_feature_grid_size_xy = float(adaptive_feature_grid_size_xy)
        self.adaptive_feature_min_relative_height = float(adaptive_feature_min_relative_height)
        self.adaptive_feature_relative_height_scale = float(
            adaptive_feature_relative_height_scale
        )
        self.adaptive_feature_boost_strength = float(adaptive_feature_boost_strength)
        self.adaptive_feature_ground_suppress_strength = float(
            adaptive_feature_ground_suppress_strength
        )
        self.adaptive_feature_max_adjust_ratio = float(adaptive_feature_max_adjust_ratio)
        self.adaptive_intensity_lower_percentile = float(adaptive_intensity_lower_percentile)
        self.adaptive_intensity_upper_percentile = float(adaptive_intensity_upper_percentile)
        self.adaptive_intensity_trigger_low = float(adaptive_intensity_trigger_low)
        self.adaptive_intensity_trigger_high = float(adaptive_intensity_trigger_high)

        self._validate_config()

    def _validate_config(self) -> None:
        if self.profile not in SUPPORTED_FRONTEND_PROFILES:
            raise ValueError(f"Unsupported OpenPCDet frontend profile: {self.profile}")

        if self.profile in RANGE_REQUIRED_PROFILES:
            if self.point_cloud_range is None:
                raise ValueError(
                    "frontend_profile in {'safe','denoise','adaptive','adaptive_enhance',"
                    "'adaptive_ground'} requires point_cloud_range."
                )
            if self.point_cloud_range.size != 6:
                raise ValueError(
                    "frontend_profile in {'safe','denoise','adaptive','adaptive_enhance',"
                    "'adaptive_ground'} expects 6-value point_cloud_range."
                )

        if self.adaptive_max_denoise_drop_ratio < 0:
            raise ValueError(
                "adaptive_max_denoise_drop_ratio must be >= 0, "
                f"got {self.adaptive_max_denoise_drop_ratio}."
            )
        if self.adaptive_feature_max_adjust_ratio < 0:
            raise ValueError(
                "adaptive_feature_max_adjust_ratio must be >= 0, "
                f"got {self.adaptive_feature_max_adjust_ratio}."
            )

    def _apply_adaptive_ground(
        self,
        filtered: np.ndarray,
        stats: Dict[str, int],
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        filtered, intensity_rescaled = robust_rescale_intensity(
            points=filtered,
            lower_percentile=self.adaptive_intensity_lower_percentile,
            upper_percentile=self.adaptive_intensity_upper_percentile,
            trigger_low=self.adaptive_intensity_trigger_low,
            trigger_high=self.adaptive_intensity_trigger_high,
        )
        stats["intensity_rescaled"] = int(intensity_rescaled)

        if filtered.shape[0] > 0:
            before = int(filtered.shape[0])
            candidate, dropped_ground = denoise_sparse_near_ground_points(
                points=filtered,
                voxel_size=self.denoise_voxel_size,
                neighbor_radius_voxels=self.denoise_neighbor_radius_voxels,
                min_neighbor_points=self.denoise_min_neighbor_points,
                ground_grid_size_xy=self.adaptive_feature_grid_size_xy,
                max_relative_height=self.adaptive_feature_min_relative_height,
            )
            stats["ground_sparse_candidate_drop"] = int(dropped_ground)
            drop_ratio = float(dropped_ground) / max(1, before)
            if drop_ratio <= self.adaptive_max_denoise_drop_ratio:
                filtered = candidate
                stats["dropped_ground_sparse"] = int(dropped_ground)
                stats["ground_sparse_applied"] = 1
            else:
                stats["ground_sparse_skipped"] = 1
            stats["num_points_after"] = int(filtered.shape[0])

        return filtered, stats

    def _apply_adaptive(
        self,
        filtered: np.ndarray,
        stats: Dict[str, int],
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        filtered, intensity_rescaled = robust_rescale_intensity(
            points=filtered,
            lower_percentile=self.adaptive_intensity_lower_percentile,
            upper_percentile=self.adaptive_intensity_upper_percentile,
            trigger_low=self.adaptive_intensity_trigger_low,
            trigger_high=self.adaptive_intensity_trigger_high,
        )
        stats["intensity_rescaled"] = int(intensity_rescaled)

        enable_feature_enhance = (
            self.adaptive_enable_feature_enhance or self.profile == "adaptive_enhance"
        )
        if enable_feature_enhance and filtered.shape[0] > 0:
            candidate, feature_points_adjusted = enhance_object_saliency_by_local_height(
                points=filtered,
                grid_size_xy=self.adaptive_feature_grid_size_xy,
                min_relative_height=self.adaptive_feature_min_relative_height,
                relative_height_scale=self.adaptive_feature_relative_height_scale,
                boost_strength=self.adaptive_feature_boost_strength,
                ground_suppress_strength=self.adaptive_feature_ground_suppress_strength,
            )
            stats["feature_points_adjusted_candidate"] = int(feature_points_adjusted)
            adjust_ratio = float(feature_points_adjusted) / max(1, int(filtered.shape[0]))
            if adjust_ratio <= self.adaptive_feature_max_adjust_ratio:
                filtered = candidate
                stats["feature_points_adjusted"] = int(feature_points_adjusted)
                stats["feature_enhance_applied"] = 1
            else:
                stats["feature_enhance_skipped"] = 1

        if self.adaptive_enable_denoise and filtered.shape[0] > 0:
            before_denoise = int(filtered.shape[0])
            denoised, dropped_denoise = denoise_points_by_voxel_density(
                points=filtered,
                voxel_size=self.denoise_voxel_size,
                neighbor_radius_voxels=self.denoise_neighbor_radius_voxels,
                min_neighbor_points=self.denoise_min_neighbor_points,
            )
            stats["adaptive_denoise_candidate_drop"] = int(dropped_denoise)
            drop_ratio = float(dropped_denoise) / max(1, before_denoise)
            if drop_ratio <= self.adaptive_max_denoise_drop_ratio:
                filtered = denoised
                stats["dropped_denoise"] = int(dropped_denoise)
                stats["adaptive_denoise_applied"] = 1
            else:
                stats["adaptive_denoise_skipped"] = 1
            stats["num_points_after"] = int(filtered.shape[0])

        return filtered, stats

    def __call__(self, points: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
        if self.profile not in RANGE_REQUIRED_PROFILES:
            stats = make_frontend_stats(include_temporal=False)
            stats["num_points_in"] = int(points.shape[0])
            stats["num_points_after"] = int(points.shape[0])
            return points, stats

        filtered, preprocess_stats = safe_preprocess_points(
            points=points,
            point_cloud_range=self.point_cloud_range,
            xyz_clip_abs=self.xyz_clip_abs,
            intensity_clip_abs=self.intensity_clip_abs,
        )

        stats = make_frontend_stats(include_temporal=False)
        stats.update(preprocess_stats)

        if self.profile == "adaptive_ground":
            return self._apply_adaptive_ground(filtered=filtered, stats=stats)

        if self.profile in {"adaptive", "adaptive_enhance"}:
            return self._apply_adaptive(filtered=filtered, stats=stats)

        if self.profile == "denoise":
            filtered, dropped_denoise = denoise_points_by_voxel_density(
                points=filtered,
                voxel_size=self.denoise_voxel_size,
                neighbor_radius_voxels=self.denoise_neighbor_radius_voxels,
                min_neighbor_points=self.denoise_min_neighbor_points,
            )
            stats["dropped_denoise"] = int(dropped_denoise)
            stats["num_points_after"] = int(filtered.shape[0])
            return filtered, stats

        # safe profile
        return filtered, stats
