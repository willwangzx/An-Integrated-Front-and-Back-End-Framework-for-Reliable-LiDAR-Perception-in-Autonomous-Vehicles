import unittest

import numpy as np

from src.bev_map import generate_bev_features
from src.clustering import classify_cluster
from src.object_features import extract_object_features
from src.preprocessing import remove_ground
from src.reflectivity_map import interpolate_reflectivity
from src.tracking import CentroidTracker


class PipelineComponentTests(unittest.TestCase):
    def test_interpolate_reflectivity_fills_center_voxel(self):
        reflectivity = {
            (-1, 0, 0): 0.2,
            (1, 0, 0): 0.4,
            (0, -1, 0): 0.6,
            (0, 1, 0): 0.8,
        }
        stability = {key: 1.0 for key in reflectivity}
        result = interpolate_reflectivity(reflectivity, stability, radius=1, min_neighbors=3)
        self.assertIn((0, 0, 0), result)
        self.assertGreater(result[(0, 0, 0)], 0.0)

    def test_remove_ground_ransac_keeps_object_points(self):
        ground = np.array([[x, y, 0.0] for x in range(-3, 4) for y in range(-3, 4)], dtype=np.float32)
        objects = np.array([[0.0, 0.0, 1.5], [1.0, 1.0, 1.8], [2.0, -1.0, 1.2]], dtype=np.float32)
        points = np.vstack([ground, objects])
        intensity = np.ones(points.shape[0], dtype=np.float32)

        filtered_points, filtered_intensity, ground_model = remove_ground(
            points,
            intensity,
            threshold=-0.2,
            method="ransac",
            max_iterations=60,
            distance_threshold=0.15,
            min_inlier_ratio=0.3,
        )
        self.assertEqual(filtered_points.shape[0], objects.shape[0])
        self.assertEqual(filtered_intensity.shape[0], objects.shape[0])
        self.assertIsNotNone(ground_model)

    def test_classification_tracking_and_features(self):
        vehicle_cluster = np.array(
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [4.0, 1.8, 1.6], [0.0, 1.8, 1.6]],
            dtype=np.float32,
        )
        self.assertEqual(classify_cluster(vehicle_cluster), "vehicle")

        clusters = [{"points": vehicle_cluster, "label": "vehicle"}]
        reflectivity = {(0, 0, 0): 0.5, (20, 9, 8): 0.7}
        stability = {(0, 0, 0): 1.0, (20, 9, 8): 0.5}
        features = extract_object_features(clusters, reflectivity, stability, voxel_size=0.2)
        self.assertEqual(features[0]["semantic_label"], "vehicle")

        tracker = CentroidTracker(max_distance=5.0)
        tracked_first = tracker.update(features)
        tracked_second = tracker.update(features)
        self.assertEqual(tracked_first[0]["track_id"], tracked_second[0]["track_id"])

    def test_generate_bev_features(self):
        points = np.array([[0.0, 0.0, 1.0], [0.1, 0.1, 2.0], [1.0, 1.0, 0.5]], dtype=np.float32)
        intensity = np.array([0.2, 0.4, 0.8], dtype=np.float32)
        bev = generate_bev_features(points, intensity, resolution=0.5)
        self.assertEqual(set(bev.keys()), {"density", "max_height", "mean_intensity"})
        self.assertAlmostEqual(float(bev["density"].sum()), 3.0)


if __name__ == "__main__":
    unittest.main()
