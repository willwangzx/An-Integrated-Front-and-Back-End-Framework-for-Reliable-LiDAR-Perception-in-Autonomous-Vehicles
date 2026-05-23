# 面向自动驾驶可靠 LiDAR 感知的一体化前后端框架

本项目实现了一个围绕 **LiDAR 点云感知** 的轻量级原型框架，目标是把点云加载、预处理、体素化、反射率建图、时序融合、聚类、目标特征提取以及 BEV（Bird's‑Eye View）表达串联为一条可复用的处理流水线，用于自动驾驶场景中的环境理解与目标分析。

项目当前采用 Python 实现，核心流程位于 `src/` 目录，并以 `.las` 点云文件作为主要输入格式。整体设计强调“从数据到结果”的端到端可运行性，适合课程项目、论文原型、算法验证和功能扩展。

---

## 0. 当前研究主线（2026-03 重构）

当前项目的核心研究方向已收敛为：

- **前端优先增强**：在不改动后端检测器结构的前提下，利用前端点云空间网格化与多信号统计增强检测鲁棒性；
- **距离自适应体素化**：近处网格更细、远处更粗（但保持在可用分辨率内）；
- **小目标保护机制**：在增强汽车收益的同时，尽量抑制对 Pedestrian / Cyclist 的负面影响；
- **实验可复现**：通过统一参数与 200 帧评估脚本进行快速 A/B 迭代。

对应实现已模块化到 `src/frontend/`，`src/openpcdet_frontend_hook.py` 保持兼容导出。

---

## 1. 项目目标

该框架主要解决以下问题：

- 对 LiDAR 点云进行标准化读取与预处理。
- 将离散点云转化为体素级表示，便于稳健建模。
- 基于点云强度信息构建三维反射率场。
- 融合历史帧信息，提高反射率估计的时序稳定性。
- 使用聚类方法提取潜在目标。
- 为每个目标生成几何与反射率统计特征。
- 生成 BEV 栅格表示，为后续检测、跟踪或可视化提供输入。

---

## 2. 当前实现的整体流程

完整感知管线由 `LidarPerceptionPipeline` 串联，处理步骤如下：

1. **读取 LAS 点云**：从 `.las` 文件中解析三维坐标与强度值。
2. **强度补偿**：根据距离对强度进行可选修正。
3. **距离过滤**：去除超出最大感知范围的点。
4. **地面去除**：通过高度阈值剔除地面点。
5. **体素化**：将点云离散到体素网格中，并统计每个体素的平均强度与点数。
6. **反射率建图**：生成体素级反射率字典。
7. **时序融合**：对多帧反射率图进行平均或 EWMA 融合，并估计稳定性。
8. **目标聚类**：使用 DBSCAN（支持分距离自适应参数）。
9. **特征提取**：统计目标点数、中心、尺寸、高度、占地面积、强度均值/方差、稳定性均值等特征。
10. **BEV 生成**：把点云投影到俯视栅格图中。

---

## 3. 项目结构

```text
.
├── README.md
├── main.py
├── openpcdet_infer_kitti.py     # OpenPCDet 推理并导出 KITTI 预测
├── src/
│   ├── frontend/               # 前端研究模块（配置/网格统计/增强/降噪/统计聚合）
│   │   ├── constants.py
│   │   ├── stats.py
│   │   ├── preprocess.py
│   │   ├── denoise.py
│   │   ├── adaptive_enhance.py
│   │   └── hook.py
│   ├── config.py                # 超参数配置
│   ├── pipeline.py              # 主感知流水线
│   ├── lidar_loader.py          # LAS 点云读取
│   ├── preprocessing.py         # 距离过滤、地面去除、强度补偿
│   ├── voxelization.py          # 体素化
│   ├── reflectivity_map.py      # 反射率图构建/插值接口
│   ├── temporal_fusion.py       # 多帧时序融合
│   ├── clustering.py            # DBSCAN 聚类（含自适应版本）
│   ├── object_features.py       # 目标特征提取
│   ├── bev_map.py               # BEV 栅格图生成
│   ├── visualization.py         # Open3D 可视化
│   └── openpcdet_frontend_hook.py # 兼容入口（重定向到 src/frontend）
├── data_processing/
│   ├── kitti_bin_to_las.py      # KITTI bin 转 LAS
│   ├── pipeline.json            # PDAL 管线示例
│   └── visualization.py
├── data/
│   └── velodyne_points/las/     # 示例 LAS 数据
├── tests/                       # 单元测试
├── docs/
│   └── OPENPCDET_KITTI.md       # OpenPCDet + KITTI 使用说明
└── 3D Voxel-Based Reflectivity Field Reconstruction/
    ├── main.ipynb
    ├── load_pointcloud.ipynb
    └── voxel_reflectivity.ipynb
```

---

## 4. 核心模块说明

### 4.1 `src/config.py`
集中管理系统参数，包括：

- `VOXEL_SIZE`：体素边长。
- `MAX_RANGE`：最大感知半径。
- `GROUND_THRESHOLD`：地面剔除高度阈值。
- `CLUSTER_EPS`、`CLUSTER_MIN_POINTS`：DBSCAN 聚类参数。
- `CLUSTER_Z_SCALE`：Z 轴缩放权重。
- `USE_ADAPTIVE_CLUSTER`：是否启用分距离自适应聚类。
- `ADAPTIVE_CLUSTER_RANGE_BINS`、`ADAPTIVE_CLUSTER_EPS_SCALES`、`ADAPTIVE_CLUSTER_MIN_SCALES`：自适应聚类配置。
- `BEV_RESOLUTION`：BEV 栅格分辨率。
- `ENABLE_INTENSITY_COMP`：是否启用强度补偿。
- `USE_EWMA_FUSION`、`EWMA_ALPHA`、`FUSION_WINDOW`：时序融合配置。
- `ENABLE_VISUALIZATION`：是否显示点云和聚类结果。

### 4.2 `src/lidar_loader.py`
负责读取 `.las` 点云文件：

- 使用 `laspy` 加载数据。
- 输出 `points`（`N x 3`）和归一化后的 `intensity`。
- 若原始点云没有强度字段，则自动填充全 1。

### 4.3 `src/preprocessing.py`
实现三个基础预处理函数：

- `filter_range()`：基于水平距离过滤点。
- `compensate_intensity()`：基于距离和可选入射角做强度补偿。
- `remove_ground()`：按高度阈值去除地面点。

### 4.4 `src/voxelization.py`
将点云映射到三维体素网格：

- 使用 `floor(point / voxel_size)` 计算体素索引。
- 统计每个体素中的平均强度和点数。
- 返回体素索引数组、体素强度数组、体素点数数组。

### 4.5 `src/reflectivity_map.py`
用于体素级反射率建图：

- `build_reflectivity()`：把体素及其强度值写入字典。
- `interpolate_reflectivity()`：目前为占位实现，保留后续邻域插值或置信补全扩展空间。

### 4.6 `src/temporal_fusion.py`
融合多帧反射率图：

- 支持普通平均融合。
- 支持 EWMA（指数加权移动平均）融合。
- 额外生成 `stability_map`，表示某体素在时间窗口中的出现频率。

### 4.7 `src/clustering.py`
使用 `DBSCAN` 做目标级点云聚类：

- 默认结合 `x/y/z` 三维信息聚类。
- 通过 `z_scale` 对高度维度施加较小权重。
- 可选分距离自适应参数配置，以更好适配远近稀疏度差异。

### 4.8 `src/object_features.py`
对每个聚类目标提取特征：

- 点数 `num_points`
- 质心 `centroid`
- 三维包围尺寸 `extent`
- 高度 `height`
- 占地面积 `footprint_area`
- 强度均值与方差
- 稳定性均值

### 4.9 `src/bev_map.py`
将点云投影为俯视角栅格图：

- 统计每个栅格中的点数。
- 适合用作轻量级空间占据表达。

### 4.10 `src/visualization.py`
基于 `Open3D` 提供两种可视化方式：

- 原始/预处理后点云可视化。
- 聚类结果可视化。

---

## 5. 运行环境

建议使用 **Python 3.9+**。

### 推荐依赖

```bash
pip install numpy laspy scikit-learn open3d
```

或直接使用：

```bash
pip install -r requirements.txt
```

如果你需要执行 `data_processing/kitti_bin_to_las.py`，确保已安装 `laspy`（已包含在 `requirements.txt`）。

---

## 6. 数据准备

当前主流程默认从以下目录读取 LAS 文件：

```text
data/velodyne_points/las/*.las
```

你可以把自己的 `.las` 文件放到该目录下，或者修改 `src/pipeline.py` 中 `iter_lidar_frames()` 的路径模式。

### 如果你的数据是 KITTI `.bin`
项目已提供一个转换脚本：

```bash
python data_processing/kitti_bin_to_las.py \
  --bin-dir /path/to/KITTI/training/velodyne \
  --las-dir data/velodyne_points/las \
  --limit 0
```

该脚本会：

- 读取 KITTI 原始四通道点云 `x, y, z, intensity`
- 重组坐标系
- 输出 `.las` 文件（支持批量与 `--overwrite`）

完整 KITTI 使用流程见：`docs/KITTI_FULL_WORKFLOW_ZH.md`

### 一键跑通（推荐）

如果你希望从 KITTI 数据到评估结果全流程一次跑完，直接使用：

```bash
python run_kitti_full_pipeline.py \
  --kitti-root /path/to/KITTI \
  --openpcdet-root /path/to/OpenPCDet \
  --openpcdet-cfg-file /path/to/OpenPCDet/tools/cfgs/kitti_models/pv_rcnn.yaml \
  --openpcdet-ckpt /path/to/checkpoints/pv_rcnn_8369.pth \
  --workspace-dir runs/kitti_full_pipeline \
  --run-name full_pipeline \
  --limit 200
```

该脚本会自动串行执行：

1. OpenPCDet 推理并导出 KITTI 预测  
2. KITTI devkit AP 评估

---

## 7. 快速开始

### 方式一：直接运行主流程

```bash
python main.py
```

程序会遍历 `data/velodyne_points/las/` 下的所有 LAS 文件，并输出类似信息：

```text
Frame 0: points=xxxx clusters=x objects=x
```

同时在开启可视化时，会显示：

- 当前帧点云
- DBSCAN 聚类结果

### 方式二：在代码中调用

```python
from src.pipeline import LidarPerceptionPipeline

pipeline = LidarPerceptionPipeline()
result = pipeline.process_frame("data/velodyne_points/las/0000000000.las")

print(result.keys())
```

返回结果字段包括：

- `points`
- `intensity`
- `voxels`
- `voxel_intensity`
- `voxel_counts`
- `reflectivity_map`
- `fused_map`
- `stability_map`
- `clusters`
- `object_features`
- `bev`

---

## 8. 可配置参数

你可以在 `src/config.py` 中快速调整算法行为。

### 示例参数含义（当前默认值）

| 参数 | 说明 | 默认值 |
|---|---|---:|
| `VOXEL_SIZE` | 体素大小（米） | `0.2` |
| `MAX_RANGE` | 最大感知距离（米） | `60` |
| `GROUND_THRESHOLD` | 地面阈值 | `-1.4` |
| `CLUSTER_EPS` | DBSCAN 邻域半径 | `0.6` |
| `CLUSTER_MIN_POINTS` | 最小聚类点数 | `15` |
| `CLUSTER_Z_SCALE` | 高度维度缩放 | `0.25` |
| `USE_ADAPTIVE_CLUSTER` | 自适应聚类 | `True` |
| `BEV_RESOLUTION` | BEV 分辨率 | `0.05` |
| `USE_EWMA_FUSION` | 是否启用 EWMA | `True` |
| `EWMA_ALPHA` | EWMA 权重 | `0.6` |
| `FUSION_WINDOW` | 融合窗口长度 | `10` |
| `ENABLE_VISUALIZATION` | 是否开启可视化 | `True` |

---

## 9. 输出结果说明

单帧处理后的输出是一个 Python 字典：

| 字段 | 类型 | 含义 |
|---|---|---|
| `points` | `np.ndarray` | 预处理后的点云坐标 |
| `intensity` | `np.ndarray` | 补偿/归一化后的强度 |
| `voxels` | `np.ndarray` | 非空体素索引 |
| `voxel_intensity` | `np.ndarray` | 每个体素的平均强度 |
| `voxel_counts` | `np.ndarray` | 每个体素的点数 |
| `reflectivity_map` | `dict` | 当前帧体素反射率图 |
| `fused_map` | `dict` | 多帧融合后的反射率图 |
| `stability_map` | `dict` | 体素稳定性图 |
| `clusters` | `list[np.ndarray]` | 聚类后的目标点集 |
| `object_features` | `list[dict]` | 每个目标的统计特征 |
| `bev` | `np.ndarray` | 俯视图栅格 |

---

## 10. Notebook 与研究原型

`3D Voxel-Based Reflectivity Field Reconstruction/` 目录下提供了若干 Notebook，适合用于：

- 点云加载实验
- 体素反射率场重建验证
- 研究思路展示与教学演示

如果你希望把 Notebook 中的实验逻辑整合到主工程中，建议逐步迁移到 `src/` 目录的模块化实现里。

---

## 11. 当前局限性

项目目前是一个 **研究/教学原型**，仍存在一些待完善点：

1. `interpolate_reflectivity()` 尚未实现真正的空间插值逻辑。
2. 地面去除仅采用固定阈值，复杂道路场景下鲁棒性有限。
3. 聚类目前只使用 DBSCAN，尚未区分车辆、行人、骑行者等类别。
4. 没有集成目标跟踪模块。
5. 已有基础单元测试，但系统化评估与端到端测试覆盖仍需增强。
6. 依赖 Open3D 的可视化不适合无图形界面的服务器环境。
7. 数据读取当前聚焦 `.las`，对 `.pcd`、`.ply`、ROS bag 等格式支持不足。

---

## 12. 后续可扩展方向

如果你计划继续完善这个框架，建议优先考虑以下方向：

- 引入 **RANSAC / Patchwork / CSF** 等更稳健的地面分割方法。
- 在反射率插值中加入邻域搜索、置信传播或 TSDF/稀疏体素策略。
- 增加 **目标检测 + 跟踪** 模块。
- 将 BEV 输出接入深度学习模型。
- 支持多传感器融合（LiDAR + Camera + IMU）。
- 增加配置文件系统（如 YAML）与命令行接口。
- 补充 benchmark、日志与实验记录机制。

---

## 13. 适用场景

本项目适用于以下场景：

- 自动驾驶感知课程作业/毕业设计
- LiDAR 点云处理算法原型开发
- 反射率场重建研究
- 时序融合与稳健感知实验
- 作为更复杂 3D 感知系统的基础骨架

---

## 14. 致使用者的建议

如果你准备将该项目用于论文、比赛或工程系统，建议你：

- 先关闭可视化，确保处理流程可批量运行。
- 使用更大的真实数据集验证参数鲁棒性。
- 为每个模块补充日志、异常处理和测试用例。
- 在主入口中加入命令行参数，而不是直接修改源码。
- 把数据路径、模型参数和实验结果统一管理。

---

## 15. 测试（新增）

项目新增了 `tests/` 目录，覆盖以下核心模块：

- 预处理：`filter_range` / `remove_ground` / `compensate_intensity`
- 体素化与反射率映射：`voxelize` / `build_reflectivity`
- 时序融合：平均融合与 EWMA 融合
- 聚类：标准 DBSCAN 与分距离自适应 DBSCAN
- 目标特征提取：`extract_object_features`
- OpenPCDet 导出关键几何：LiDAR 3D 框到 KITTI 标注字段的转换

运行方式（标准库 `unittest`）：

```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

## 16. OpenPCDet 3D 识别接入（新增）

项目新增脚本 `openpcdet_infer_kitti.py`，用于：

- 调用 OpenPCDet 模型做 3D 检测推理
- 将预测导出为 KITTI 官方评估格式 `.txt`
- 与 `kitti_devkit_eval.py` 直接衔接做 AP 评估
- 在 `__getitem__` 中提供 OpenPCDet 专用安全前端 hook（读点后、`prepare_data` 前）
  - 默认 `adaptive`：保守清理 + 强度自适应修复（必要时）
  - `adaptive_ground`：结构保留型近地面稀疏噪点清理
  - `adaptive_enhance`：基于局部高度的主体特征增强（抑制地面、强化目标点）
  - 可选 `denoise` / `temporal_3frame` 做前端消融

示例：

```bash
python openpcdet_infer_kitti.py \
  --openpcdet-root /path/to/OpenPCDet \
  --cfg-file /path/to/OpenPCDet/tools/cfgs/kitti_models/pv_rcnn.yaml \
  --ckpt /path/to/checkpoints/pv_rcnn_8369.pth \
  --bin-dir /path/to/KITTI/training/velodyne \
  --calib-dir /path/to/KITTI/training/calib \
  --out-dir kitti_predictions_openpcdet/data \
  --score-thresh 0.1
```

然后执行：

```bash
python kitti_devkit_eval.py \
  --label-dir /path/to/KITTI/training/label_2 \
  --pred-dir kitti_predictions_openpcdet/data \
  --run-name openpcdet_eval \
  --compile
```

更详细说明见：`docs/OPENPCDET_KITTI.md`

注意：`src/preprocessing.py` 中的地面去除/强度补偿属于启发式链路，不建议直接接入
OpenPCDet 推理；若改变输入分布，应配套重训或至少微调。

---

## 17. License

仓库中当前未看到明确的开源许可证文件。

如果你希望公开发布或允许他人复用，建议补充标准许可证（如 MIT、Apache-2.0 或 GPL）。

---

## 18. 一句话总结

这是一个围绕 **“LiDAR 点云 → 体素反射率建图 → 时序融合 → 聚类与目标特征提取”** 的自动驾驶感知原型框架，适合作为研究验证和系统扩展的起点。
