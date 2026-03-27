import numpy as np

def voxelize(points,intensity,voxel_size):

    if points.size == 0:
        empty_idx = np.empty((0, 3), dtype=np.int32)
        empty_float = np.empty((0,), dtype=np.float32)
        empty_counts = np.empty((0,), dtype=np.int32)
        return empty_idx, empty_float, empty_counts

    voxel_indices = np.floor(points / voxel_size).astype(np.int32, copy=False)
    voxels, inverse = np.unique(voxel_indices, axis=0, return_inverse=True)

    counts = np.bincount(inverse, minlength=voxels.shape[0]).astype(np.int32, copy=False)
    summed_intensity = np.bincount(
        inverse,
        weights=intensity.astype(np.float64, copy=False),
        minlength=voxels.shape[0],
    )

    nonzero = counts > 0
    intensities = np.zeros(voxels.shape[0], dtype=np.float32)
    intensities[nonzero] = (summed_intensity[nonzero] / counts[nonzero]).astype(
        np.float32,
        copy=False,
    )

    return voxels, intensities, counts
