import open3d as o3d
import numpy as np

def visualize_points(points):

    pcd=o3d.geometry.PointCloud()

    pcd.points=o3d.utility.Vector3dVector(points)

    o3d.visualization.draw_geometries([pcd])


def visualize_clusters(clusters):

    geoms=[]

    for c in clusters:

        p=o3d.geometry.PointCloud()
        p.points=o3d.utility.Vector3dVector(c)
        geoms.append(p)

    o3d.visualization.draw_geometries(geoms)