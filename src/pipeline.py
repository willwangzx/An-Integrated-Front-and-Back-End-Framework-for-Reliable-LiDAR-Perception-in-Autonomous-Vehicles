from collections import deque
from glob import glob

from src.bev_map import generate_bev
from src.clustering import cluster_objects
from src.config import (
    BEV_RESOLUTION,
    CLUSTER_EPS,
    CLUSTER_MIN_POINTS,
    ENABLE_INTENSITY_COMP,
    EWMA_ALPHA,
    FUSION_WINDOW,
    GROUND_THRESHOLD,
    MAX_RANGE,
    RANGE_ATTENUATION_ALPHA,
    USE_EWMA_FUSION,
    VOXEL_SIZE,
)
from src.lidar_loader import load_las
from src.object_features import extract_object_features
from src.preprocessing import compensate_intensity, filter_range, remove_ground
from src.reflectivity_map import build_reflectivity, interpolate_reflectivity
from src.temporal_fusion import fuse_maps
from src.voxelization import voxelize


class LidarPerceptionPipeline:
    def __init__(self, fusion_window=FUSION_WINDOW):
        self.maps = deque(maxlen=max(fusion_window, 1))

    def process_frame(self, file_path):
        points, intensity = load_las(file_path)

        if ENABLE_INTENSITY_COMP:
            intensity = compensate_intensity(points, intensity, RANGE_ATTENUATION_ALPHA)

        points, intensity = filter_range(points, intensity, MAX_RANGE)
        points, intensity = remove_ground(points, intensity, GROUND_THRESHOLD)

        voxels, voxel_intensity, voxel_counts = voxelize(points, intensity, VOXEL_SIZE)
        reflectivity_map = build_reflectivity(voxels, voxel_intensity)
        self.maps.append(reflectivity_map)

        clusters = cluster_objects(points, CLUSTER_EPS, CLUSTER_MIN_POINTS)
        fused_map, stability_map = fuse_maps(
            list(self.maps),
            use_ewma=USE_EWMA_FUSION,
            ewma_alpha=EWMA_ALPHA,
        )
        fused_map = interpolate_reflectivity(fused_map, stability_map)

        object_features = extract_object_features(
            clusters,
            fused_map,
            stability_map,
            VOXEL_SIZE,
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
            "clusters": clusters,
            "object_features": object_features,
            "bev": generate_bev(points, BEV_RESOLUTION),
        }


def iter_lidar_frames(pattern="data/velodyne_points/las/*.las"):
    return sorted(glob(pattern))
