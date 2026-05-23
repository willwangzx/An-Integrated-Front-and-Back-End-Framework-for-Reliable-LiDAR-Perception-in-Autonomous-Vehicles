from sklearn.cluster import DBSCAN
import numpy as np


def classify_cluster(points):
    if points.size == 0:
        return "unknown"

    extent = points.max(axis=0) - points.min(axis=0)
    height = float(extent[2]) if extent.size > 2 else 0.0
    width = float(min(extent[0], extent[1])) if extent.size > 1 else 0.0
    length = float(max(extent[0], extent[1])) if extent.size > 1 else 0.0

    if 1.0 <= height <= 2.4 and 2.0 <= length <= 6.5 and 1.0 <= width <= 3.0:
        return "vehicle"
    if 1.2 <= height <= 2.2 and length <= 1.2 and width <= 1.2:
        return "pedestrian"
    if 1.0 <= height <= 2.2 and 1.0 <= length <= 2.5 and width <= 1.2:
        return "cyclist"
    return "unknown"


def cluster_objects(points, eps, min_samples, z_scale=0.25):

    if points.shape[0] < min_samples:
        return []

    if points.shape[1] >= 3 and z_scale > 0:
        features = points[:, :3].copy()
        features[:, 2] *= z_scale
    else:
        features = points[:, :2]

    model = DBSCAN(eps=eps, min_samples=min_samples, algorithm="kd_tree", n_jobs=-1)

    labels = model.fit_predict(features)

    valid_idx = np.flatnonzero(labels >= 0)
    if valid_idx.size == 0:
        return []

    ordered = valid_idx[np.argsort(labels[valid_idx], kind="mergesort")]
    ordered_labels = labels[ordered]
    split_idx = np.flatnonzero(np.diff(ordered_labels)) + 1
    clusters = [points[idx] for idx in np.split(ordered, split_idx)]

    return clusters


def cluster_objects_adaptive(
    points,
    base_eps,
    base_min_samples,
    range_bins,
    eps_scales,
    min_samples_scales,
    z_scale=0.25,
):
    if points.shape[0] == 0:
        return []

    if len(range_bins) < 2:
        raise ValueError("range_bins must contain at least two edges.")
    n_bands = len(range_bins) - 1
    if len(eps_scales) != n_bands or len(min_samples_scales) != n_bands:
        raise ValueError(
            "eps_scales and min_samples_scales must have len(range_bins)-1 values."
        )

    radii = np.linalg.norm(points[:, :2], axis=1)
    clusters = []
    for idx in range(n_bands):
        lo = range_bins[idx]
        hi = range_bins[idx + 1]
        if idx == n_bands - 1:
            band_mask = (radii >= lo) & (radii <= hi)
        else:
            band_mask = (radii >= lo) & (radii < hi)

        if not np.any(band_mask):
            continue

        band_points = points[band_mask]
        band_eps = float(base_eps * eps_scales[idx])
        band_min_samples = max(3, int(round(base_min_samples * min_samples_scales[idx])))
        clusters.extend(
            cluster_objects(
                band_points,
                eps=band_eps,
                min_samples=band_min_samples,
                z_scale=z_scale,
            )
        )
    return clusters
