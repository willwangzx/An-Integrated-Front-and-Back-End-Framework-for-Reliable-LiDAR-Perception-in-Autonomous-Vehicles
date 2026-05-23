import numpy as np


def _grid_parameters(points, resolution):
    x = points[:, 0]
    y = points[:, 1]

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    width = int((xmax - xmin) / resolution) + 1
    height = int((ymax - ymin) / resolution) + 1

    xi = ((x - xmin) / resolution).astype(int)
    yi = ((y - ymin) / resolution).astype(int)
    return (width, height), (xi, yi), (xmin, ymin)


def generate_bev(points, resolution=0.2):
    if points.size == 0:
        return np.zeros((0, 0), dtype=np.int32)

    (width, height), (xi, yi), _ = _grid_parameters(points, resolution)
    bev = np.zeros((width, height), dtype=np.int32)
    np.add.at(bev, (xi, yi), 1)
    return bev


def generate_bev_features(points, intensity, resolution=0.2):
    if points.size == 0:
        empty = np.zeros((0, 0), dtype=np.float32)
        return {
            "density": empty,
            "max_height": empty,
            "mean_intensity": empty,
        }

    (width, height), (xi, yi), _ = _grid_parameters(points, resolution)
    density = np.zeros((width, height), dtype=np.float32)
    max_height = np.full((width, height), -np.inf, dtype=np.float32)
    intensity_sum = np.zeros((width, height), dtype=np.float32)

    np.add.at(density, (xi, yi), 1)
    np.maximum.at(max_height, (xi, yi), points[:, 2])
    np.add.at(intensity_sum, (xi, yi), intensity)

    mean_intensity = np.divide(
        intensity_sum,
        np.maximum(density, 1.0),
        out=np.zeros_like(intensity_sum),
        where=density > 0,
    )
    max_height[max_height == -np.inf] = 0.0

    return {
        "density": density,
        "max_height": max_height,
        "mean_intensity": mean_intensity,
    }
