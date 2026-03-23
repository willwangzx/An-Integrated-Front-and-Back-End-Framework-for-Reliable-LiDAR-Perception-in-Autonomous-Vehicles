def build_reflectivity(voxels, intensity):
    reflectivity = {}

    for v, i in zip(voxels, intensity):
        reflectivity[tuple(int(axis) for axis in v)] = float(i)

    return reflectivity


def _neighbor_keys(key, radius):
    x, y, z = key
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                if dx == dy == dz == 0:
                    continue
                yield (x + dx, y + dy, z + dz)


def interpolate_reflectivity(
    reflectivity,
    stability,
    stability_threshold=0.7,
    radius=1,
    min_neighbors=3,
):
    if not reflectivity:
        return reflectivity

    interpolated = dict(reflectivity)
    candidate_keys = set()
    for key, stab in stability.items():
        if stab >= stability_threshold and key not in reflectivity:
            candidate_keys.add(key)
        if stab >= stability_threshold:
            candidate_keys.update(_neighbor_keys(key, radius))

    for key in candidate_keys:
        if key in interpolated:
            continue

        weighted_sum = 0.0
        total_weight = 0.0
        neighbors = 0
        for neighbor in _neighbor_keys(key, radius):
            if neighbor not in reflectivity:
                continue
            dx = neighbor[0] - key[0]
            dy = neighbor[1] - key[1]
            dz = neighbor[2] - key[2]
            dist = (dx * dx + dy * dy + dz * dz) ** 0.5
            if dist == 0:
                continue
            weight = stability.get(neighbor, 0.0) / dist
            if weight <= 0:
                continue
            weighted_sum += reflectivity[neighbor] * weight
            total_weight += weight
            neighbors += 1

        if neighbors >= min_neighbors and total_weight > 0:
            interpolated[key] = float(weighted_sum / total_weight)

    return interpolated
