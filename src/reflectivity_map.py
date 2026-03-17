def build_reflectivity(voxels,intensity):

    reflectivity={}

    for v,i in zip(voxels,intensity):

        reflectivity[tuple(int(axis) for axis in v)]=float(i)

    return reflectivity

def interpolate_reflectivity(reflectivity,stability,stability_threshold=0.7):

    if not reflectivity:
        return reflectivity

    # Placeholder: keep original values unless confident neighbors are available.
    return reflectivity
