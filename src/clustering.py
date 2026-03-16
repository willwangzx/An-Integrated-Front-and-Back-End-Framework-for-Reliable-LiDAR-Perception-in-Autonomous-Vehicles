from sklearn.cluster import DBSCAN
import numpy as np

def cluster_objects(points,eps,min_samples):

    model = DBSCAN(eps=eps,min_samples=min_samples)

    labels = model.fit_predict(points)

    clusters=[]

    for l in set(labels):

        if l==-1:
            continue

        clusters.append(points[labels==l])

    return clusters