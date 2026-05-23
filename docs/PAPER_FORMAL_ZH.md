# 面向自动驾驶可靠 LiDAR 感知的一体化前后端框架

## 摘要
针对自动驾驶场景中 LiDAR 感知链路“工程可复现性不足、前后端耦合松散、鲁棒性与精度难以统一评估”的问题，本文提出并实现了一种一体化前后端框架。该框架以 OpenPCDet 作为学习型检测前端，在数据加载环节引入保守型安全预处理钩子（非有限值清理、极值裁剪、点云范围裁剪），同时构建可解释的后端可靠性分支（地面去除、ROI/FOV约束、范围自适应与类别感知聚类、几何先验筛选、置信度校准与类内NMS），并通过统一导出与 KITTI 官方 devkit 评测脚本形成端到端闭环。基于仓库中 2026-03-29 至 2026-03-30 的已有实验记录，在 KITTI 7481 帧评测中，OpenPCDet(PointPillar) 分支取得 3D AP(moderate)：Car 47.706、Pedestrian 28.672、Cyclist 46.344，三类平均 mAP 为 40.907；启发式后端独立检测精度显著较低，但在召回导向配置下可提升候选保留率与部分类别召回趋势。结果表明：学习型前端应作为主检测能力来源，可靠性后端更适合以重排序/过滤分支形式与其互补集成。

关键词：自动驾驶；LiDAR 感知；OpenPCDet；可靠性后处理；KITTI 评测

---

## 1. 引言
自动驾驶 3D 感知系统通常由“前端检测器 + 工程后处理”构成，但实际研发中常见两类问题：一是前后端模块分离，导致实验不可复现、难以做统一消融；二是异常输入与分布漂移会放大系统不稳定性。为解决上述问题，本文以工程可复现为核心约束，构建了从推理导出到官方评测的一体化框架，并在同一代码库中统一学习型检测前端与可解释后端分支。

本文贡献如下：

1. 构建了覆盖“推理导出-评测汇总-实验对比”的一体化 LiDAR 感知框架，支持不同前后端策略在同一协议下可比评估。
2. 设计了 OpenPCDet 前端安全钩子，在不引入激进分布扰动的前提下提升输入稳健性。
3. 实现了可解释的后端可靠性分支，并通过 KITTI 官方指标验证其能力边界与适用定位。

---

## 2. 方法

### 2.1 系统总体架构
系统由四个层次组成：

1. 学习型前端：`openpcdet_infer_kitti.py` 调用 OpenPCDet 模型完成 3D 检测并导出 KITTI 预测。
2. 可靠性后端：`kitti_export.py` 基于点云几何与统计规则生成候选框与得分。
3. 通用点云主链：`src/pipeline.py` 统一组织预处理、体素化、反射率建图、时序融合、聚类与特征提取。
4. 评测与编排：`kitti_devkit_eval.py` + `run_kitti_full_pipeline.py` 完成官方 AP 计算与实验闭环。

### 2.2 前端安全钩子
在 OpenPCDet 数据集 `__getitem__` 阶段插入保守预处理：

1. 删除包含 NaN/Inf 的点。
2. 对 xyz 与强度做绝对值裁剪，抑制异常极值。
3. 按 `POINT_CLOUD_RANGE` 进行范围过滤。

该策略仅做“安全清洗”，不引入地面去除等强启发式操作，从而尽量避免训练/推理分布偏移。

### 2.3 后端可靠性分支
后端分支以可解释规则为核心，流程为：

1. 预处理：距离过滤、地面去除（阈值或RANSAC）。
2. 空间约束：ROI 裁剪与相机重叠 FOV 约束。
3. 聚类：分距离自适应 DBSCAN，并可叠加类别感知 profile。
4. 几何建模：通过相机标定将聚类点云映射到相机坐标，估计 3D 伪框与 2D 投影框。
5. 类别判别与打分：结合尺寸先验、点数、距离、类别 margin 等构造原始分数，并支持 logistic/isotonic 标定。
6. 后处理：类别内 2D/BEV IoU NMS，输出 KITTI 标准格式。

### 2.4 统一评测机制
系统统一使用 KITTI devkit 进行 AP 统计，输出 `kitti_eval_summary.json`。所有实验共用同一评测入口，保证跨实验可比性。

---

## 3. 实验设置

### 3.1 数据与指标

1. 数据集：KITTI Object（以 `training` split 中可评测样本为主）。
2. 指标：KITTI 官方 AP（2D/BEV/3D，Easy/Moderate/Hard），本文重点报告 Moderate。
3. 记录来源：仓库内 `results/*/kitti_eval_summary.json` 与 `runs/*/*export_summary.json`。

### 3.2 对比组

1. OpenPCDet 主干：`openpcdet_pointpillar_7481`。
2. 启发式后端基线：`full_eval_7481`。
3. 启发式改进版：`full_eval_improved_7481`。
4. 启发式召回导向版：`full_eval_recall_reliable_7481`。
5. 3740 帧消融对比：`ab_openpcdet_3740`、`ab_old_heuristic_3740_v2`、`ab_new_heuristic_3740`。

### 3.3 图表清单与产物映射（现有数据）

以下图表已基于当前仓库已有结果自动生成，供论文正文直接引用：

1. 图1（系统框架图）：`docs/figures/paper_research/fig1_system_framework.png`
2. 图2（量化误差与离散分辨率示意）：`docs/figures/paper_research/fig2_quantization_resolution.png`
3. 图3（1000帧增强强度trade-off）：`docs/figures/paper_research/fig3_1000_aggressive_tradeoff.png`
4. 图4（7481帧全量策略重跑对比）：`docs/figures/paper_research/fig4_full_strategy_7481.png`
5. 图5（voxel-size sweep曲线）：`docs/figures/paper_research/fig5_voxel_sweep_map.png`
6. 图6（exports/frame vs voxel size）：`docs/figures/paper_research/fig6_exports_vs_voxel.png`
7. 图7（neighbor_radius × min_points热力图）：`docs/figures/paper_research/fig7_neighbor_grid_heatmap.png`
8. 表1（7481帧主结果）：`docs/tables/paper_research/table1_main_results_7481.md`
9. 表2（1000帧增强强度对比）：`docs/tables/paper_research/table2_1000_aggressive_compare.md`
10. 表3（全量策略重跑）：`docs/tables/paper_research/table3_full_strategy_rerun.md`
11. 表4（voxel-size sweep）：`docs/tables/paper_research/table4_voxel_size_sweep.md`
12. 表5（proxy quantization分析）：`docs/tables/paper_research/table5_proxy_quantization_analysis.md`

统一图表导航文档：`docs/PAPER_FIGURES_TABLES_ZH.md`

---

## 4. 结果与分析

### 4.1 7481 帧主结果

| 方案 | n_test_images | Car 3D mod | Ped 3D mod | Cyc 3D mod | 3D mAP mod | BEV mAP mod | 2D mAP mod |
|---|---:|---:|---:|---:|---:|---:|---:|
| OpenPCDet(PointPillar) | 7481 | 47.706 | 28.672 | 46.344 | **40.907** | **58.418** | **52.409** |
| Heuristic Baseline | 7481 | 0.000 | 0.047 | 0.000 | 0.016 | 0.110 | 1.585 |
| Heuristic Improved | 7481 | 0.000 | 0.101 | 0.000 | 0.034 | 0.161 | 2.011 |
| Heuristic Recall-Reliable | 7481 | 0.000 | 0.149 | 0.000 | 0.050 | 0.180 | 1.879 |

分析：

1. OpenPCDet 分支在三类 3D moderate 上显著领先，说明主检测能力主要由学习型前端提供。
2. 启发式分支的改进策略对 Pedestrian 指标有增益（0.047 -> 0.101 -> 0.149），但整体仍与学习型方法存在数量级差距。
3. 召回导向配置提高了候选保留强度：`full_eval_recall_reliable` 平均导出 9.094/frame，高于 `full_eval_improved` 的 6.174/frame。

### 4.2 3740 帧消融结果

| 方案 | n_test_images | 3D mAP mod | BEV mAP mod | 2D mAP mod |
|---|---:|---:|---:|---:|
| ab_openpcdet_3740 | 3740 | **42.477** | **60.065** | **54.371** |
| ab_old_heuristic_3740_v2 | 3740 | 0.035 | 0.199 | 1.979 |
| ab_new_heuristic_3740 | 3740 | 0.000 | 0.005 | 0.057 |

分析：

1. 与启发式分支相比，OpenPCDet 在 3740 帧上的优势依旧显著，结论与 7481 帧结果一致。
2. 启发式不同版本之间虽有筛选行为差异，但未改变“难以替代学习型前端”的总体结论。

### 4.3 导出行为与工程稳定性观察

1. `openpcdet_pointpillar_7481` 导出 242,949 个结果，平均 32.475/frame；相对原始检测 245,946 的保留率约 98.78%，说明几何非法剔除比例低。
2. 启发式后端在全量评测中聚类数量高但有效保留率低（如 `full_eval_improved`: 300,516 clusters -> 46,190 exports），表明其主要作用更偏向“候选过滤与可靠性重排”。

---

## 5. 讨论

1. 可靠性后端适合与学习型检测前端并联/串联，承担风险控制、规则约束和重排序，而不宜作为唯一检测器。
2. 当前框架已具备工程化闭环能力，适合作为后续“时序可靠性分支 + 学习型主干联合优化”的实验底座。
3. 后续可将 `temporal_fusion` 与 `reflectivity_map` 的时序稳定性特征显式注入检测框重打分模块，进一步验证对误检抑制与置信度校准的贡献。

---

## 6. 结论
本文实现了一个面向自动驾驶 LiDAR 感知的一体化前后端框架，并以 KITTI 官方评测验证了各模块的作用边界。实验结果显示，学习型前端（OpenPCDet）是当前精度主来源；启发式后端可提供可解释的可靠性增强，但其独立检测性能仍有限。该结论为后续研究提供了明确方向：以学习型前端为主体，将可靠性后端作为可控、可解释的辅助分支进行融合优化。

---

## 附：可直接复现实验的关键结果文件

1. `results/openpcdet_pointpillar_7481/kitti_eval_summary.json`
2. `results/full_eval_7481/kitti_eval_summary.json`
3. `results/full_eval_improved_7481/kitti_eval_summary.json`
4. `results/full_eval_recall_reliable_7481/kitti_eval_summary.json`
5. `results/ab_openpcdet_3740/kitti_eval_summary.json`
6. `results/ab_old_heuristic_3740_v2/kitti_eval_summary.json`
7. `results/ab_new_heuristic_3740/kitti_eval_summary.json`
8. `runs/openpcdet_pointpillar_7481/openpcdet_export_summary.json`
9. `runs/kitti_full_eval_improved/predictions/kitti_export_summary.json`
10. `runs/kitti_full_eval_recall_reliable/predictions/kitti_export_summary.json`
