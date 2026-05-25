# 面向自动驾驶可靠 LiDAR 感知的一体化前后端框架：项目梳理与论文描述（草稿）

## 1. 项目框架梳理

### 1.1 总体架构
本项目围绕 KITTI 数据构建了“前端检测 + 后端可靠性处理 + 统一评测”的一体化链路：

1. 前端检测分支（学习型）：`openpcdet_infer_kitti.py` 调用 OpenPCDet（如 PointPillar），完成 3D 检测推理并导出 KITTI 预测文件。
2. 前端安全钩子（工程稳健性）：`src/openpcdet_frontend_hook.py` 在 `__getitem__` 处插入保守预处理（NaN/Inf 清理、极值裁剪、点云范围裁剪），降低异常输入导致的推理不稳定。
3. 后端可靠性分支（几何/统计启发式）：`kitti_export.py` 对点云执行地面去除、ROI/FOV 约束、范围自适应+类别感知 DBSCAN 聚类、伪框构造、类别判别、置信度校准与类内 NMS，输出 KITTI 格式结果。
4. 统一评测：`kitti_devkit_eval.py` 编译并调用 KITTI 官方 devkit C++ evaluator，输出 `kitti_eval_summary.json`。
5. 单入口流程：`run_kitti_full_pipeline.py` 串联“预测导出 + AP 评估”，保证实验可复现。

### 1.2 核心模块对应关系

- 通用 LiDAR 处理主链：`src/pipeline.py`
  - `lidar_loader.py`：点云读取
  - `preprocessing.py`：距离过滤、强度补偿、阈值/RANSAC 地面去除
  - `voxelization.py` + `reflectivity_map.py`：体素化与反射率场构建
  - `temporal_fusion.py`：多帧平均/EWMA 融合与稳定性估计
  - `clustering.py`：标准/分距离自适应/类别感知 DBSCAN
  - `object_features.py`：目标几何与强度统计特征
  - `bev_map.py`：BEV 栅格表达
- KITTI 评测链：`openpcdet_infer_kitti.py` / `kitti_export.py` / `kitti_devkit_eval.py`

### 1.3 设计思想

- 学习型前端负责检测主精度；
- 启发式后端负责可解释约束与可靠性筛选；
- 评测与导出工具统一，保证不同分支可直接做可比实验。

---

## 2. 论文描述（可直接改写使用）

### 2.1 做了什么
本文实现了一个面向自动驾驶场景的 LiDAR 感知一体化框架。与仅关注单一检测器的方案不同，本工作将深度学习前端（OpenPCDet）与几何统计后端（聚类、伪框、置信度校准、NMS）进行工程化整合，并在 KITTI 数据集上构建了从推理导出到官方 AP 评测的完整闭环。具体而言，我们在 OpenPCDet 数据加载阶段引入保守型前端安全处理，以提升异常输入条件下的鲁棒性；同时保留可解释的启发式后端分支，用于候选筛选、类别先验约束与可靠性分析。最终，系统以统一接口输出 KITTI 预测结果并自动生成标准化评测摘要，支持消融、参数扫描与复现实验。

### 2.2 目标是什么
本研究的目标不是单点优化某个算子，而是回答三个系统性问题：

1. 如何将“学习型检测器 + 可靠性后处理”组织为可复现、可比较的一体化前后端框架；
2. 如何在不破坏训练/推理分布一致性的前提下，引入前端安全机制提升工程稳定性；
3. 如何通过统一导出与官方评测流程，客观量化不同后端策略对精度与召回的影响。

### 2.3 效果如何（基于仓库现有实验结果）

#### A. 学习型前端（OpenPCDet）表现
- `results/openpcdet_pointpillar_7481/kitti_eval_summary.json`（7481 帧）
  - 3D AP (moderate): Car 47.706，Pedestrian 28.672，Cyclist 46.344
  - 3D mAP (moderate, 三类平均): **40.907**
  - BEV mAP (moderate): **58.418**
  - 2D mAP (moderate): **52.409**
- `runs/openpcdet_pointpillar_7481/openpcdet_export_summary.json`
  - 导出 242,949 个预测，平均 32.475/frame
  - 与原始检测 245,946 相比，保留率约 **98.78%**（几何无效过滤很少）

#### B. 启发式后端分支表现与消融
- `results/full_eval_7481/kitti_eval_summary.json`（7481 帧）
  - 3D mAP (moderate): 0.016
- `results/full_eval_improved_7481/kitti_eval_summary.json`（7481 帧）
  - 3D mAP (moderate): 0.034
  - 相比 baseline 在 Pedestrian 3D moderate 上约 2.15x 提升（0.047 -> 0.101）
- `results/full_eval_recall_reliable_7481/kitti_eval_summary.json`（7481 帧）
  - 3D mAP (moderate): 0.050
  - Pedestrian 3D moderate 提升到 0.149（相对 baseline 约 3.15x）
- `runs/kitti_full_eval_recall_reliable/predictions/kitti_export_summary.json`
  - 平均导出 9.094/frame，显著高于 `full_eval_improved` 的 6.174/frame，说明召回取向策略提高了候选保留率。

#### C. 关键对比结论
- 在当前实现下，**检测精度主贡献来自 OpenPCDet 前端**；
- 启发式后端虽然在部分指标上可改善召回趋势，但与学习型检测器仍有数量级差距，现阶段更适合作为可靠性分析与候选重排模块，而非独立主检测器。

---

## 3. 可用于论文“贡献点”写法（建议）

1. 提出并实现了一个可复现的 LiDAR 前后端一体化实验框架，覆盖推理、导出、评测全流程。
2. 设计了与 OpenPCDet 解耦且分布友好的前端安全钩子，在不引入激进分布漂移的前提下增强输入稳健性。
3. 构建了具备可解释性的后端可靠性分支（几何先验、置信度校准、类内 NMS），并通过 KITTI 官方评测完成系统级消融分析。

---

## 4. 一段式摘要（可直接放摘要初稿）

本文面向自动驾驶 LiDAR 感知任务，提出了一种一体化前后端框架，将 OpenPCDet 学习型检测前端、保守型输入安全处理、以及可解释的几何统计后端统一到同一评测闭环中。系统支持从 KITTI 点云推理、预测导出到官方 devkit AP 计算的端到端自动化流程。实验表明，学习型前端在 7481 帧上的 3D moderate mAP 达到 40.907，验证了主干检测能力；同时，后端可靠性分支可通过参数化筛选与置信度校准改变候选保留行为并改善部分类别召回趋势，但其独立检测精度仍显著低于深度检测器。该结果说明，后端分支更适合作为可靠性增强与重排序模块，与学习型前端形成互补。

---

## 5. 当前局限（建议在论文中主动说明）

1. 启发式后端对 Car 3D 指标贡献有限，泛化能力受规则先验约束。
2. 体素反射率与时序融合分支尚未深度耦合到学习型检测头进行联合优化。
3. 缺少统一 runtime benchmark 与不同天气/域偏移条件下的系统鲁棒性报告。

