# 前端消融报告（全量版）

> 状态：本报告已切换为“全量测试优先”流程，1000 帧临时结论不再作为最终结论。

## 1. 报告入口

全量测试完成后，请优先阅读：

- `docs/FULL_RESEARCH_REPORT_ZH.md`（自动生成的主报告）
- `runs/full_research_suite_v1/full_research_report_zh.md`（运行目录内副本）

## 2. 全量实验产物规范

### 2.1 策略重跑
- `runs/full_research_suite_v1/strategy_rerun/full_strategy_summary.csv`
- 字段核心：
  - `strategy_group`
  - `frontend_profile`
  - `adaptive_feature_max_adjust_ratio`
  - `car_3d_moderate / pedestrian_3d_moderate / cyclist_3d_moderate / mean_3d_moderate`
  - `*_delta_vs_baseline`

### 2.2 voxel_size 推理扫参
- `runs/full_research_suite_v1/voxel_sweep/voxel_sweep_table.csv`
- 字段核心：
  - `exp_id`
  - `voxel_size_xy`
  - `voxel_size_z`
  - `profile`
  - `n_frames`
  - `map_3d_mod`
  - `car_3d_mod / ped_3d_mod / cyc_3d_mod`
  - `exports_per_frame`

### 2.3 neighbor 正交网格
- `runs/full_research_suite_v1/neighbor_grid/neighbor_grid_table.csv`
- 字段核心：
  - `grid_group`
  - `adaptive_feature_max_adjust_ratio`
  - `neighbor_radius_voxels`
  - `min_neighbor_points`
  - `drop_ratio`
  - `map_3d_mod`
  - `ped_3d_mod`

## 3. 运行命令（全量）

```bash
OUT_ROOT=runs/full_research_suite_v1 LIMIT=0 DEVICE=cpu \
bash scripts/run_full_all_and_report.sh \
  /path/to/KITTI_ROOT \
  /path/to/OpenPCDet \
  /path/to/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml \
  /path/to/checkpoints/pointpillar_7728.pth
```

## 4. 说明

- 若使用 CPU，全量套跑耗时较长，建议长时任务运行（screen/tmux/CI runner）。
- 报告由 `scripts/generate_full_research_report.py` 自动生成，避免手工抄录误差。
