from collections import deque
from glob import glob

from src.bev_map import generate_bev, generate_bev_features
from src.clustering import classify_cluster, cluster_objects, cluster_objects_adaptive
from src.config import (
    ADAPTIVE_CLUSTER_EPS_SCALES,
    ADAPTIVE_CLUSTER_MIN_SCALES,
    ADAPTIVE_CLUSTER_RANGE_BINS,
    BEV_RESOLUTION,
    CLUSTER_EPS,
    CLUSTER_MIN_POINTS,
    CLUSTER_Z_SCALE,
    ENABLE_INTENSITY_COMP,
    EWMA_ALPHA,
    FUSION_WINDOW,
    GROUND_THRESHOLD,
    MAX_RANGE,
    RANGE_ATTENUATION_ALPHA,
    USE_ADAPTIVE_CLUSTER,
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


class LidarPerceptionPipeline:
    def __init__(self, fusion_window=FUSION_WINDOW):
        self.maps = deque(maxlen=max(fusion_window, 1))
        self.tracker = CentroidTracker()

    def process_frame(self, file_path):
        points, intensity = load_las(file_path)

        if ENABLE_INTENSITY_COMP:
            intensity = compensate_intensity(points, intensity, RANGE_ATTENUATION_ALPHA)

        points, intensity = filter_range(points, intensity, MAX_RANGE)
        points, intensity, _ = remove_ground(points, intensity, GROUND_THRESHOLD)

        voxels, voxel_intensity, voxel_counts = voxelize(points, intensity, VOXEL_SIZE)
        reflectivity_map = build_reflectivity(voxels, voxel_intensity)
        self.maps.append(reflectivity_map)

        if USE_ADAPTIVE_CLUSTER:
            clusters = cluster_objects_adaptive(
                points=points,
                base_eps=CLUSTER_EPS,
                base_min_samples=CLUSTER_MIN_POINTS,
                range_bins=ADAPTIVE_CLUSTER_RANGE_BINS,
                eps_scales=ADAPTIVE_CLUSTER_EPS_SCALES,
                min_samples_scales=ADAPTIVE_CLUSTER_MIN_SCALES,
                z_scale=CLUSTER_Z_SCALE,
            )
        else:
            clusters = cluster_objects(
                points,
                CLUSTER_EPS,
                CLUSTER_MIN_POINTS,
                z_scale=CLUSTER_Z_SCALE,
            )
        fused_map, stability_map = fuse_maps(
            list(self.maps),
            use_ewma=USE_EWMA_FUSION,
            ewma_alpha=EWMA_ALPHA,
        )
        fused_map = interpolate_reflectivity(fused_map, stability_map)

        labeled_clusters = [
            {"points": cluster, "label": classify_cluster(cluster)}
            for cluster in clusters
        ]
        object_features = extract_object_features(
            labeled_clusters,
            fused_map,
            stability_map,
            VOXEL_SIZE,
        )
        tracked_objects = self.tracker.update(object_features)

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
            "labeled_clusters": labeled_clusters,
            "object_features": object_features,
            "tracked_objects": tracked_objects,
            "bev": generate_bev(points, BEV_RESOLUTION),
            "bev_features": generate_bev_features(points, intensity, BEV_RESOLUTION),
        }


def iter_lidar_frames(pattern="data/velodyne_points/las/*.las"):
    return sorted(glob(pattern))
