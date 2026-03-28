import numpy as np


def extract_object_features(clusters, reflectivity_map, stability_map, voxel_size):
    features = []

    for cluster in clusters:
        points = cluster["points"] if isinstance(cluster, dict) else cluster
        semantic_label = cluster.get("label", "unknown") if isinstance(cluster, dict) else "unknown"
        if points.size == 0:
            continue

        centroid = points.mean(axis=0)
        extent = points.max(axis=0) - points.min(axis=0)
        voxel_keys = np.unique(np.floor(points / voxel_size).astype(int), axis=0)

        refl_vals = []
        stab_vals = []
        for voxel in voxel_keys:
            key = tuple(int(axis) for axis in voxel)
            if key in reflectivity_map:
                refl_vals.append(reflectivity_map[key])
            if stability_map and key in stability_map:
                stab_vals.append(stability_map[key])

        refl_vals = np.array(refl_vals, dtype=np.float32) if refl_vals else np.array([], dtype=np.float32)
        stab_vals = np.array(stab_vals, dtype=np.float32) if stab_vals else np.array([], dtype=np.float32)

        features.append(
            {
                "num_points": int(points.shape[0]),
                "centroid": centroid,
                "extent": extent,
                "height": float(extent[2]) if extent.size > 2 else 0.0,
                "footprint_area": float(extent[0] * extent[1]) if extent.size > 1 else 0.0,
                "intensity_mean": float(refl_vals.mean()) if refl_vals.size > 0 else 0.0,
                "intensity_var": float(refl_vals.var()) if refl_vals.size > 0 else 0.0,
                "stability_mean": float(stab_vals.mean()) if stab_vals.size > 0 else 0.0,
                "semantic_label": semantic_label,
            }
        )

    return features
