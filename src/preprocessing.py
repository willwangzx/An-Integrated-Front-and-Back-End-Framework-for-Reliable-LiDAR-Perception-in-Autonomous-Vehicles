import numpy as np

def filter_range(points,intensity,max_range):

    dist = np.linalg.norm(points[:,:2],axis=1)

    mask = dist < max_range

    return points[mask],intensity[mask]

def compensate_intensity(points,intensity,alpha=0.0,incidence_angles=None):

    ranges = np.linalg.norm(points,axis=1)
    corrected = intensity * (ranges ** 2) * np.exp(alpha * ranges)

    if incidence_angles is not None:
        cos_term = np.cos(incidence_angles)
        corrected = corrected / (cos_term + 1e-6)

    return corrected


def remove_ground(points,intensity,threshold):

    mask = points[:,2] > threshold

    return points[mask],intensity[mask]
