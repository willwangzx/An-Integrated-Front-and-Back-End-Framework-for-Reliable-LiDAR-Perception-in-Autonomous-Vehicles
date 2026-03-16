import numpy as np

def filter_range(points,intensity,max_range):

    dist = np.linalg.norm(points[:,:2],axis=1)

    mask = dist < max_range

    return points[mask],intensity[mask]


def remove_ground(points,intensity,threshold):

    mask = points[:,2] > threshold

    return points[mask],intensity[mask]