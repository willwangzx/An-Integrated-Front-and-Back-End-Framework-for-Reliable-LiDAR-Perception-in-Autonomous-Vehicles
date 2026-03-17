import numpy as np

def filter_range(points,intensity,max_range):

    if points.size == 0:
        return points, intensity

    dist = np.linalg.norm(points[:,:2],axis=1)

    mask = dist < max_range

    return points[mask],intensity[mask]

def compensate_intensity(points,intensity,alpha=0.0,incidence_angles=None):

    if points.size == 0:
        return intensity

    ranges = np.linalg.norm(points,axis=1)
    corrected = intensity * np.maximum(ranges, 1e-3) ** 2 * np.exp(alpha * ranges)

    if incidence_angles is not None:
        cos_term = np.clip(np.cos(incidence_angles), 1e-3, None)
        corrected = corrected / (cos_term + 1e-6)

    peak = np.max(corrected) if corrected.size else 0.0
    if peak > 0:
        corrected = corrected / peak

    return corrected.astype(np.float32, copy=False)


def remove_ground(points,intensity,threshold):

    if points.size == 0:
        return points, intensity

    mask = points[:,2] > threshold

    return points[mask],intensity[mask]
