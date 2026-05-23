import unittest

import numpy as np

from src.openpcdet_frontend_hook import (
    OpenPCDetFrontendHook,
    apply_rigid_transform,
    denoise_sparse_near_ground_points,
    denoise_points_by_voxel_density,
    enhance_object_saliency_by_local_height,
    robust_rescale_intensity,
    safe_preprocess_points,
)


class TestOpenPCDetFrontendHook(unittest.TestCase):
    def test_safe_preprocess_points_filters_and_clips(self) -> None:
        points = np.array(
            [
                [1.0, 0.0, 0.0, 0.5],
                [np.nan, 0.0, 0.0, 0.1],
                [2.0, 0.0, np.inf, 0.2],
                [20.0, 0.0, 0.0, 2000.0],
                [5.0, 3.0, 0.0, 0.3],
                [5.0, 1.0, 0.0, 2000.0],
                [-20000.0, 0.0, 0.0, 0.2],
            ],
            dtype=np.float32,
        )
        point_cloud_range = np.array([0.0, -2.0, -1.0, 10.0, 2.0, 1.0], dtype=np.float32)

        out, stats = safe_preprocess_points(
            points=points,
            point_cloud_range=point_cloud_range,
            xyz_clip_abs=1e4,
            intensity_clip_abs=1e3,
        )

        self.assertEqual(out.shape, (2, 4))
        np.testing.assert_allclose(out[0], np.array([1.0, 0.0, 0.0, 0.5], dtype=np.float32))
        np.testing.assert_allclose(out[1], np.array([5.0, 1.0, 0.0, 1000.0], dtype=np.float32))

        self.assertEqual(stats["num_points_in"], 7)
        self.assertEqual(stats["dropped_non_finite"], 2)
        self.assertEqual(stats["dropped_out_of_range"], 3)
        self.assertEqual(stats["num_points_after"], 2)

    def test_hook_profile_none_is_noop(self) -> None:
        points = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        hook = OpenPCDetFrontendHook(
            profile="none",
            point_cloud_range=None,
            xyz_clip_abs=1e4,
            intensity_clip_abs=1e3,
        )

        out, stats = hook(points)
        np.testing.assert_allclose(out, points)
        self.assertEqual(stats["num_points_after"], 1)
        self.assertEqual(stats["dropped_non_finite"], 0)
        self.assertEqual(stats["dropped_out_of_range"], 0)

    def test_hook_profile_safe_requires_point_cloud_range(self) -> None:
        with self.assertRaises(ValueError):
            OpenPCDetFrontendHook(
                profile="safe",
                point_cloud_range=None,
                xyz_clip_abs=1e4,
                intensity_clip_abs=1e3,
            )

    def test_denoise_points_by_voxel_density_removes_isolated_points(self) -> None:
        points = np.array(
            [
                [0.00, 0.00, 0.00, 1.0],
                [0.05, 0.02, 0.01, 1.0],
                [0.08, 0.01, 0.00, 1.0],
                [5.00, 5.00, 0.00, 1.0],
            ],
            dtype=np.float32,
        )
        out, dropped = denoise_points_by_voxel_density(
            points=points,
            voxel_size=0.2,
            neighbor_radius_voxels=1,
            min_neighbor_points=3,
        )
        self.assertEqual(dropped, 1)
        self.assertEqual(out.shape[0], 3)

    def test_apply_rigid_transform_keeps_intensity(self) -> None:
        points = np.array([[1.0, 0.0, 0.0, 0.25]], dtype=np.float32)
        transform = np.eye(4, dtype=np.float32)
        transform[:3, 3] = np.array([2.0, -1.0, 0.5], dtype=np.float32)

        out = apply_rigid_transform(points, transform)
        np.testing.assert_allclose(out[0, :3], np.array([3.0, -1.0, 0.5], dtype=np.float32))
        self.assertAlmostEqual(float(out[0, 3]), 0.25, places=6)

    def test_hook_profile_denoise(self) -> None:
        points = np.array(
            [
                [1.0, 0.0, 0.0, 0.1],
                [1.1, 0.0, 0.0, 0.2],
                [1.2, 0.0, 0.0, 0.3],
                [30.0, 30.0, 0.0, 0.9],
            ],
            dtype=np.float32,
        )
        hook = OpenPCDetFrontendHook(
            profile="denoise",
            point_cloud_range=np.array([0.0, -40.0, -3.0, 70.0, 40.0, 1.0], dtype=np.float32),
            xyz_clip_abs=1e4,
            intensity_clip_abs=1e3,
            denoise_voxel_size=0.25,
            denoise_neighbor_radius_voxels=1,
            denoise_min_neighbor_points=3,
        )
        out, stats = hook(points)
        self.assertEqual(out.shape[0], 3)
        self.assertEqual(stats["dropped_denoise"], 1)

    def test_robust_rescale_intensity_scales_out_of_range_values(self) -> None:
        points = np.array(
            [
                [1.0, 0.0, 0.0, 10.0],
                [1.1, 0.0, 0.0, 100.0],
                [1.2, 0.0, 0.0, 255.0],
            ],
            dtype=np.float32,
        )
        out, changed = robust_rescale_intensity(points)
        self.assertEqual(changed, 1)
        self.assertTrue(np.all(out[:, 3] >= 0.0))
        self.assertTrue(np.all(out[:, 3] <= 1.0))

    def test_hook_profile_adaptive_skips_over_aggressive_denoise(self) -> None:
        points = np.array(
            [
                [1.0, 0.0, 0.0, 10.0],
                [1.1, 0.0, 0.0, 20.0],
                [1.2, 0.0, 0.0, 30.0],
                [30.0, 30.0, 0.0, 255.0],
            ],
            dtype=np.float32,
        )
        hook = OpenPCDetFrontendHook(
            profile="adaptive",
            point_cloud_range=np.array([0.0, -40.0, -3.0, 70.0, 40.0, 1.0], dtype=np.float32),
            xyz_clip_abs=1e4,
            intensity_clip_abs=1e3,
            denoise_voxel_size=0.25,
            denoise_neighbor_radius_voxels=1,
            denoise_min_neighbor_points=3,
            adaptive_enable_denoise=True,
            adaptive_max_denoise_drop_ratio=0.1,
        )
        out, stats = hook(points)
        self.assertEqual(out.shape[0], 4)
        self.assertEqual(stats["intensity_rescaled"], 1)
        self.assertEqual(stats["dropped_denoise"], 0)
        self.assertEqual(stats["adaptive_denoise_candidate_drop"], 1)
        self.assertEqual(stats["adaptive_denoise_applied"], 0)
        self.assertEqual(stats["adaptive_denoise_skipped"], 1)
        self.assertTrue(np.all(out[:, 3] >= 0.0))
        self.assertTrue(np.all(out[:, 3] <= 1.0))

    def test_hook_profile_adaptive_applies_denoise_when_drop_ratio_safe(self) -> None:
        points = np.array(
            [
                [1.0, 0.0, 0.0, 10.0],
                [1.1, 0.0, 0.0, 20.0],
                [1.2, 0.0, 0.0, 30.0],
                [30.0, 30.0, 0.0, 255.0],
            ],
            dtype=np.float32,
        )
        hook = OpenPCDetFrontendHook(
            profile="adaptive",
            point_cloud_range=np.array([0.0, -40.0, -3.0, 70.0, 40.0, 1.0], dtype=np.float32),
            xyz_clip_abs=1e4,
            intensity_clip_abs=1e3,
            denoise_voxel_size=0.25,
            denoise_neighbor_radius_voxels=1,
            denoise_min_neighbor_points=3,
            adaptive_enable_denoise=True,
            adaptive_max_denoise_drop_ratio=0.3,
        )
        out, stats = hook(points)
        self.assertEqual(out.shape[0], 3)
        self.assertEqual(stats["adaptive_denoise_candidate_drop"], 1)
        self.assertEqual(stats["dropped_denoise"], 1)
        self.assertEqual(stats["adaptive_denoise_applied"], 1)
        self.assertEqual(stats["adaptive_denoise_skipped"], 0)

    def test_enhance_object_saliency_by_local_height(self) -> None:
        points = np.array(
            [
                [0.00, 0.00, 0.00, 0.50],
                [0.10, 0.05, 0.05, 0.50],
                [0.08, 0.03, 1.00, 0.50],
            ],
            dtype=np.float32,
        )
        out, changed = enhance_object_saliency_by_local_height(
            points=points,
            grid_size_xy=0.5,
            min_relative_height=0.1,
            relative_height_scale=0.6,
            boost_strength=0.2,
            ground_suppress_strength=0.05,
        )
        self.assertGreater(changed, 0)
        self.assertLess(float(out[0, 3]), 0.50)
        self.assertLess(float(out[1, 3]), 0.50)
        self.assertGreater(float(out[2, 3]), 0.50)

    def test_enhance_object_saliency_by_local_height_distance_adaptive(self) -> None:
        points = np.array(
            [
                [5.00, 0.00, 0.00, 0.50],   # near-ground (near range)
                [5.05, 0.00, 1.00, 0.50],   # elevated (near range)
                [60.00, 0.00, 0.00, 0.50],  # near-ground (far range)
                [60.10, 0.00, 1.00, 0.50],  # elevated (far range)
            ],
            dtype=np.float32,
        )
        out, changed = enhance_object_saliency_by_local_height(
            points=points,
            grid_size_xy=0.8,
            min_relative_height=0.1,
            relative_height_scale=0.8,
            boost_strength=0.3,
            ground_suppress_strength=0.05,
        )
        self.assertGreater(changed, 0)
        near_boost = float(out[1, 3] - points[1, 3])
        far_boost = float(out[3, 3] - points[3, 3])
        self.assertGreater(near_boost, far_boost)

    def test_hook_profile_adaptive_enhance(self) -> None:
        points = np.array(
            [
                [0.00, 0.00, 0.00, 10.0],
                [0.10, 0.00, 0.05, 20.0],
                [0.05, 0.00, 0.80, 30.0],
            ],
            dtype=np.float32,
        )
        hook = OpenPCDetFrontendHook(
            profile="adaptive_enhance",
            point_cloud_range=np.array([-5.0, -5.0, -3.0, 5.0, 5.0, 3.0], dtype=np.float32),
            xyz_clip_abs=1e4,
            intensity_clip_abs=1e3,
            adaptive_feature_max_adjust_ratio=1.0,
        )
        out, stats = hook(points)
        self.assertEqual(stats["intensity_rescaled"], 1)
        self.assertEqual(stats["feature_enhance_applied"], 1)
        self.assertEqual(stats["feature_enhance_skipped"], 0)
        self.assertGreaterEqual(stats["feature_points_adjusted"], 0)
        self.assertTrue(np.all(out[:, 3] >= 0.0))
        self.assertTrue(np.all(out[:, 3] <= 1.0))

    def test_denoise_sparse_near_ground_points(self) -> None:
        points = np.array(
            [
                [0.00, 0.00, 0.00, 0.5],
                [0.05, 0.00, 0.02, 0.5],
                [0.10, 0.00, 0.01, 0.5],
                [0.00, 0.05, 0.00, 0.5],
                [0.05, 0.05, 0.02, 0.5],
                [0.90, 0.00, 0.01, 0.5],  # sparse near-ground with neighbor-ground ref -> drop
                [4.00, 4.00, 1.00, 0.5],  # sparse but elevated -> keep
            ],
            dtype=np.float32,
        )
        out, dropped = denoise_sparse_near_ground_points(
            points=points,
            voxel_size=0.2,
            neighbor_radius_voxels=1,
            min_neighbor_points=3,
            ground_grid_size_xy=0.5,
            max_relative_height=0.15,
        )
        self.assertEqual(dropped, 1)
        self.assertEqual(out.shape[0], 6)

    def test_hook_profile_adaptive_ground(self) -> None:
        points = np.array(
            [
                [0.00, 0.00, 0.00, 0.5],
                [0.05, 0.00, 0.02, 0.5],
                [0.10, 0.00, 0.01, 0.5],
                [0.00, 0.05, 0.00, 0.5],
                [0.05, 0.05, 0.02, 0.5],
                [0.90, 0.00, 0.01, 0.5],
                [4.00, 4.00, 1.00, 0.5],
            ],
            dtype=np.float32,
        )
        hook = OpenPCDetFrontendHook(
            profile="adaptive_ground",
            point_cloud_range=np.array([-10.0, -10.0, -3.0, 10.0, 10.0, 3.0], dtype=np.float32),
            xyz_clip_abs=1e4,
            intensity_clip_abs=1e3,
            denoise_voxel_size=0.2,
            denoise_neighbor_radius_voxels=1,
            denoise_min_neighbor_points=3,
            adaptive_feature_grid_size_xy=0.5,
            adaptive_feature_min_relative_height=0.15,
            adaptive_max_denoise_drop_ratio=1.0,
        )
        out, stats = hook(points)
        self.assertEqual(out.shape[0], 6)
        self.assertEqual(stats["ground_sparse_applied"], 1)
        self.assertEqual(stats["ground_sparse_skipped"], 0)
        self.assertEqual(stats["dropped_ground_sparse"], 1)


if __name__ == "__main__":
    unittest.main()
