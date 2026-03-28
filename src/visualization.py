import open3d as o3d


def visualize_points(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])


def visualize_clusters(clusters):
    geoms = []

    for cluster in clusters:
        points = cluster["points"] if isinstance(cluster, dict) else cluster
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        geoms.append(pcd)

    if geoms:
        o3d.visualization.draw_geometries(geoms)
