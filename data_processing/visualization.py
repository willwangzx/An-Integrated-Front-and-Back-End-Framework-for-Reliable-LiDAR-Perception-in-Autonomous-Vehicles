import open3d as o3d
import laspy
import numpy as np
import matplotlib.pyplot as plt
import os


def visualize_las_open3d_basic(las_file):
    """使用Open3D基本可视化LAS文件"""

    # 1. 读取LAS文件
    print(f"读取文件: {las_file}")
    las = laspy.read(las_file)

    # 2. 提取点坐标
    points = np.vstack((las.x, las.y, las.z)).transpose()

    print(f"总点数: {points.shape[0]}")
    print(f"坐标范围: X[{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    print(f"         Y[{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
    print(f"         Z[{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")

    # 3. 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 4. 根据高程设置颜色
    z_min = points[:, 2].min()
    z_max = points[:, 2].max()

    # 使用matplotlib的colormap
    colormap = plt.cm.viridis  # 也可以选择其他: plasma, summer, winter, hot等
    normalized_z = (points[:, 2] - z_min) / (z_max - z_min)
    colors = colormap(normalized_z)[:, :3]  # 取RGB，忽略alpha

    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 5. 可视化
    print("正在打开可视化窗口...")
    print("使用方法:")
    print("  鼠标左键: 旋转")
    print("  鼠标中键: 平移")
    print("  鼠标右键: 缩放")
    print("  +/-: 调整点大小")
    print("  L: 显示/隐藏坐标轴")
    print("  C: 切换颜色模式")
    print("  Esc: 退出")

    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"点云可视化: {os.path.basename(las_file)}",
                      width=1200, height=800)

    # 添加几何体
    vis.add_geometry(pcd)

    # 设置渲染选项
    render_option = vis.get_render_option()
    render_option.point_size = 1.0  # 点大小
    render_option.background_color = np.array([0.1, 0.1, 0.1])  # 背景颜色
    render_option.show_coordinate_frame = True  # 显示坐标轴

    # 运行可视化
    vis.run()
    vis.destroy_window()


# 使用示例
if __name__ == "__main__":
    las_file = r"C:/Users/willw/PycharmProjects/Lidar/data/velodyne_points/las/0000000021.las"
    visualize_las_open3d_basic(las_file)