from sklearn.cluster import DBSCAN
import numpy as np


def classify_cluster(cluster):
    if cluster.size == 0:
        return "unknown"

    extent = cluster.max(axis=0) - cluster.min(axis=0)
    length, width = sorted(extent[:2], reverse=True)
    height = extent[2] if extent.size > 2 else 0.0

    if 2.5 <= length <= 6.5 and 1.2 <= width <= 3.0 and 1.0 <= height <= 3.0:
        return "vehicle"
    if 0.2 <= width <= 1.2 and 1.2 <= height <= 2.3:
        return "pedestrian"
    if 0.5 <= width <= 1.5 and 1.0 <= height <= 2.2 and length <= 2.5:
        return "cyclist"
    return "unknown"


def cluster_objects(points, eps, min_samples, z_scale=0.25):
    if points.shape[0] < min_samples:
        return []

    features = points[:, :2]
    if points.shape[1] >= 3 and z_scale > 0:
        features = np.column_stack((points[:, :2], points[:, 2] * z_scale))

    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(features)

    clusters = []
    for label in sorted(set(labels)):
        if label == -1:
            continue

        cluster_points = points[labels == label]
        clusters.append(
            {
                "points": cluster_points,
                "label": classify_cluster(cluster_points),
            }
        )

    return clusters
