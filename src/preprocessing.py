import numpy as np


def filter_range(points, intensity, max_range):
    if points.size == 0:
        return points, intensity

    dist = np.linalg.norm(points[:, :2], axis=1)
    mask = dist < max_range
    return points[mask], intensity[mask]


def compensate_intensity(points, intensity, alpha=0.0, incidence_angles=None):
    if points.size == 0:
        return intensity

    ranges = np.linalg.norm(points, axis=1)
    corrected = intensity * np.maximum(ranges, 1e-3) ** 2 * np.exp(alpha * ranges)

    if incidence_angles is not None:
        cos_term = np.clip(np.cos(incidence_angles), 1e-3, None)
        corrected = corrected / (cos_term + 1e-6)

    peak = np.max(corrected) if corrected.size else 0.0
    if peak > 0:
        corrected = corrected / peak

    return corrected.astype(np.float32, copy=False)


def _fit_plane_from_points(sample_points):
    p0, p1, p2 = sample_points
    normal = np.cross(p1 - p0, p2 - p0)
    norm = np.linalg.norm(normal)
    if norm < 1e-6:
        return None, None

    normal = normal / norm
    d = -np.dot(normal, p0)
    return normal, d


def remove_ground_ransac(
    points,
    intensity,
    threshold,
    max_iterations=80,
    distance_threshold=0.2,
    min_inlier_ratio=0.2,
    random_state=42,
):
    if points.size == 0 or points.shape[0] < 3:
        return points, intensity, None

    rng = np.random.default_rng(random_state)
    best_inliers = None
    best_plane = None
    n_points = points.shape[0]

    for _ in range(max_iterations):
        sample_idx = rng.choice(n_points, size=3, replace=False)
        normal, d = _fit_plane_from_points(points[sample_idx])
        if normal is None:
            continue

        if abs(normal[2]) < 0.7:
            continue

        distances = np.abs(points @ normal + d)
        inliers = distances < distance_threshold

        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_plane = (normal, d)

    if best_inliers is None or best_inliers.mean() < min_inlier_ratio:
        fallback_points, fallback_intensity, _ = remove_ground(points, intensity, threshold, method="threshold")
        return fallback_points, fallback_intensity, None

    plane_points = points[best_inliers]
    centroid = plane_points.mean(axis=0)
    centered = plane_points - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal = normal / max(np.linalg.norm(normal), 1e-6)
    if normal[2] < 0:
        normal = -normal
    d = -np.dot(normal, centroid)

    signed_distances = points @ normal + d
    nonground_mask = signed_distances > distance_threshold

    return points[nonground_mask], intensity[nonground_mask], {
        "normal": normal.astype(np.float32),
        "offset": float(d),
        "inlier_ratio": float(best_inliers.mean()),
    }


def remove_ground(points, intensity, threshold, method="threshold", **kwargs):
    if points.size == 0:
        return points, intensity, None

    if method == "ransac":
        return remove_ground_ransac(points, intensity, threshold, **kwargs)

    mask = points[:, 2] > threshold
    return points[mask], intensity[mask], None
