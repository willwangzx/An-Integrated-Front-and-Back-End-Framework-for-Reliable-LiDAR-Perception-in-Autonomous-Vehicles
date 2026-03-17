from sklearn.cluster import DBSCAN
import numpy as np

def cluster_objects(points,eps,min_samples,z_scale=0.25):

    if points.shape[0] < min_samples:
        return []

    features = points[:, :2]
    if points.shape[1] >= 3 and z_scale > 0:
        features = np.column_stack((points[:, :2], points[:, 2] * z_scale))

    model = DBSCAN(eps=eps,min_samples=min_samples)

    labels = model.fit_predict(features)

    clusters=[]

    for l in sorted(set(labels)):

        if l==-1:
            continue

        clusters.append(points[labels==l])

    return clusters
