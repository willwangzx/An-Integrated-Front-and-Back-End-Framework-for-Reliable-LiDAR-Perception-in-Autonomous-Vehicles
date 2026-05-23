# 前端研究方向重构说明（2026-03-31）

## 1. 重构目标
围绕当前研究方向，将“前端点云增强”从单体脚本重构为可组合模块：

- 距离自适应体素化（近细远粗）；
- 网格多信号统计（点数、反射强度均值/方差、局部高度信息）；
- 小目标保护门控（避免 Pedestrian/Cyclist 被过抑制）；
- 统一统计口径，便于 200 帧 A/B 对比。

## 2. 新架构
新增 `src/frontend/`：

- `constants.py`：前端 profile 与统计键常量；
- `stats.py`：统计初始化/聚合工具；
- `preprocess.py`：安全预处理、强度自适应重标定、刚体变换；
- `denoise.py`：体素邻域稀疏点降噪与近地稀疏降噪；
- `adaptive_enhance.py`：距离自适应网格索引 + 多信号增强策略；
- `hook.py`：`OpenPCDetFrontendHook` 配置校验与 profile 编排。

保留兼容入口：

- `src/openpcdet_frontend_hook.py`：只做稳定导出，不再承载实现细节。

## 3. 工程落地改动
- `openpcdet_infer_kitti.py` 迁移到新包导入；
- 用 `make_frontend_stats/merge_frontend_stats` 统一替换重复统计字典与累加逻辑；
- CLI 参数与 profile 行为保持兼容，已有命令无需改动。

## 4. 兼容性与验证
- 兼容：`tests/test_openpcdet_frontend_hook.py` 旧导入路径不变；
- 已通过：`tests/test_openpcdet_frontend_hook.py` 全部 14 项；
- 全量 `unittest discover` 中 `test_pipeline_components` 仍有历史导入问题（`classify_cluster` 缺失），与本次前端重构无关。

## 5. 后续建议
1. 在 `adaptive_enhance.py` 增加“类别敏感”门控超参集（Car 优先 / 平衡模式）；
2. 已增加 sweep 预设：`car_gain_safe_200`、`car_gain_aggressive_200`、`car_gain_full_200`，可继续扩展场景化模板；
3. 在后端重训阶段将前端统计特征（density/intensity-variance）作为辅助通道并做一致性训练。

## 6. 全量实验化执行（论文导向）
已重构为“一键全量套跑 + 自动汇总报告”流程：

- `scripts/run_full_strategy_rerun.sh`
  - 重跑历史核心策略族（baseline / conservative / balanced / aggressive / full / ultra / adaptive_ground / denoise）。
- `scripts/run_full_voxel_sweep.sh`
  - 全量 `voxel_size` 推理扫参，输出论文表所需字段（含 `exports_per_frame`）。
- `scripts/run_full_neighbor_grid.sh`
  - 全量 `neighbor_radius × min_points` 正交，覆盖 `r=0.35` 与 `r=0.6`。
- `scripts/run_full_all_and_report.sh`
  - 顺序执行以上三类实验并自动生成总报告。
- `scripts/generate_full_research_report.py`
  - 汇总 CSV/JSON 产物，生成 `runs/.../full_research_report_zh.md` 与 `docs/FULL_RESEARCH_REPORT_ZH.md`。

推荐入口命令：

```bash
OUT_ROOT=runs/full_research_suite_v1 LIMIT=0 DEVICE=cpu \
bash scripts/run_full_all_and_report.sh \
  /path/to/KITTI_ROOT \
  /path/to/OpenPCDet \
  /path/to/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml \
  /path/to/checkpoints/pointpillar_7728.pth
```
