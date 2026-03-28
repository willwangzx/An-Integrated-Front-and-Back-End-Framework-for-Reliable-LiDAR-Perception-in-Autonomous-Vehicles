from collections import deque
from glob import glob
import logging

from src.bev_map import generate_bev, generate_bev_features
from src.clustering import cluster_objects
from src.config import (
    BEV_RESOLUTION,
    CLUSTER_EPS,
    CLUSTER_MIN_POINTS,
    ENABLE_INTENSITY_COMP,
    EWMA_ALPHA,
    FUSION_WINDOW,
    GROUND_METHOD,
    GROUND_THRESHOLD,
    INTERPOLATE_REFLECTIVITY,
    INTERPOLATION_MIN_NEIGHBORS,
    INTERPOLATION_RADIUS,
    INTERPOLATION_STABILITY_THRESHOLD,
    LOG_LEVEL,
    MAX_RANGE,
    RANSAC_DISTANCE_THRESHOLD,
    RANSAC_MAX_ITERATIONS,
    RANSAC_MIN_INLIER_RATIO,
    RANGE_ATTENUATION_ALPHA,
    TRACKING_MAX_DISTANCE,
    USE_EWMA_FUSION,
    VOXEL_SIZE,
)
from src.lidar_loader import load_las
from src.object_features import extract_object_features
from src.preprocessing import compensate_intensity, filter_range, remove_ground
from src.reflectivity_map import build_reflectivity, interpolate_reflectivity
from src.temporal_fusion import fuse_maps
from src.tracking import CentroidTracker
from src.voxelization import voxelize

LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper(), logging.INFO))


class LidarPerceptionPipeline:
    def __init__(self, fusion_window=FUSION_WINDOW, tracker=None):
        self.maps = deque(maxlen=max(fusion_window, 1))
        self.tracker = tracker or CentroidTracker(max_distance=TRACKING_MAX_DISTANCE)

    def process_frame(self, file_path):
        points, intensity = load_las(file_path)
        raw_point_count = int(points.shape[0])

        if ENABLE_INTENSITY_COMP:
            intensity = compensate_intensity(points, intensity, RANGE_ATTENUATION_ALPHA)

        points, intensity = filter_range(points, intensity, MAX_RANGE)
        points, intensity, ground_model = remove_ground(
            points,
            intensity,
            GROUND_THRESHOLD,
            method=GROUND_METHOD,
            max_iterations=RANSAC_MAX_ITERATIONS,
            distance_threshold=RANSAC_DISTANCE_THRESHOLD,
            min_inlier_ratio=RANSAC_MIN_INLIER_RATIO,
        )

        voxels, voxel_intensity, voxel_counts = voxelize(points, intensity, VOXEL_SIZE)
        reflectivity_map = build_reflectivity(voxels, voxel_intensity)
        self.maps.append(reflectivity_map)

        clusters = cluster_objects(points, CLUSTER_EPS, CLUSTER_MIN_POINTS)
        fused_map, stability_map = fuse_maps(
            list(self.maps),
            use_ewma=USE_EWMA_FUSION,
            ewma_alpha=EWMA_ALPHA,
        )
        if INTERPOLATE_REFLECTIVITY:
            fused_map = interpolate_reflectivity(
                fused_map,
                stability_map,
                stability_threshold=INTERPOLATION_STABILITY_THRESHOLD,
                radius=INTERPOLATION_RADIUS,
                min_neighbors=INTERPOLATION_MIN_NEIGHBORS,
            )

        object_features = extract_object_features(
            clusters,
            fused_map,
            stability_map,
            VOXEL_SIZE,
        )
        tracked_objects = self.tracker.update(object_features)
        bev = generate_bev(points, BEV_RESOLUTION)
        bev_features = generate_bev_features(points, intensity, BEV_RESOLUTION)

        LOGGER.info(
            "Processed %s: raw_points=%d filtered_points=%d clusters=%d tracked=%d",
            file_path,
            raw_point_count,
            int(points.shape[0]),
            len(clusters),
            len(tracked_objects),
        )

        return {
            "points": points,
            "intensity": intensity,
            "voxels": voxels,
            "voxel_intensity": voxel_intensity,
            "voxel_counts": voxel_counts,
            "reflectivity_map": reflectivity_map,
            "fused_map": fused_map,
            "stability_map": stability_map,
            "ground_model": ground_model,
            "clusters": clusters,
            "object_features": object_features,
            "tracked_objects": tracked_objects,
            "bev": bev,
            "bev_features": bev_features,
        }


def iter_lidar_frames(pattern="data/velodyne_points/las/*.las"):
    return sorted(glob(pattern))
