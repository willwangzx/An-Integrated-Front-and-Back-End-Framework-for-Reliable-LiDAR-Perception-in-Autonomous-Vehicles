# KITTI 数据集完整使用流程（OpenPCDet-only）

本文给出从数据准备到 AP 评估的完整流程。当前单入口脚本
`run_kitti_full_pipeline.py` 已移除启发式检测后端，仅保留 OpenPCDet。

---

## 0. 单入口脚本

使用仓库根目录脚本：

`run_kitti_full_pipeline.py`

固定执行两步：

1. OpenPCDet 推理并导出 KITTI 预测文本
2. 编译并运行 KITTI devkit 评估，输出 AP

---

## 1. 数据目录准备

确保 KITTI Object 数据目录至少包含：

- `KITTI_ROOT/training/velodyne/*.bin`
- `KITTI_ROOT/training/calib/*.txt`
- `KITTI_ROOT/training/label_2/*.txt`

注意：

- `testing` split 没有 `label_2`，不能算 AP。
- 评估脚本默认按连续帧编号（`000000` 开始）推断评估区间。

---

## 2. 环境与依赖

你需要两个环境：

1. 本仓库运行环境（用于导出脚本与评估脚本）
2. OpenPCDet 环境（按 OpenPCDet 官方要求安装）

---

## 3. 一条命令跑完全流程

Linux/macOS（推荐）：

```bash
python run_kitti_full_pipeline.py \
  --kitti-root /path/to/KITTI \
  --openpcdet-root /path/to/OpenPCDet \
  --openpcdet-cfg-file /path/to/OpenPCDet/tools/cfgs/kitti_models/pv_rcnn.yaml \
  --openpcdet-ckpt /path/to/checkpoints/pv_rcnn_8369.pth \
  --workspace-dir runs/kitti_full_pipeline \
  --run-name openpcdet_full_pipeline \
  --limit 200
```

Windows（MSYS2）：

```bash
python run_kitti_full_pipeline.py \
  --kitti-root D:/KITTI \
  --openpcdet-root D:/OpenPCDet \
  --openpcdet-cfg-file D:/OpenPCDet/tools/cfgs/kitti_models/pv_rcnn.yaml \
  --openpcdet-ckpt D:/checkpoints/pv_rcnn_8369.pth \
  --workspace-dir runs/kitti_full_pipeline \
  --run-name openpcdet_full_pipeline \
  --limit 200 \
  --bash-path C:/msys64/usr/bin/bash.exe
```

输出结果：

- `runs/kitti_full_pipeline/predictions/data/*.txt`
- `results/<run-name>/kitti_eval_summary.json`
- `runs/kitti_full_pipeline/full_pipeline_summary.json`

---

## 4. OpenPCDet 前端 hook 参数

可选参数：

- `--openpcdet-frontend-profile {none,safe,denoise,adaptive,adaptive_ground,adaptive_enhance,temporal_3frame}`（默认 `adaptive`）
- `--openpcdet-frontend-xyz-clip-abs`
- `--openpcdet-frontend-intensity-clip-abs`
- `--openpcdet-frontend-denoise-voxel-size`（默认 `0.25`）
- `--openpcdet-frontend-denoise-neighbor-radius-voxels`（默认 `1`）
- `--openpcdet-frontend-denoise-min-neighbor-points`（默认 `4`）
- `--openpcdet-frontend-temporal-pose-file`（可选）
- `--openpcdet-frontend-temporal-allow-missing-pose`（可选，不建议默认开启）
- `--openpcdet-frontend-adaptive-enable-denoise`（可选）
- `--openpcdet-frontend-adaptive-enable-feature-enhance`（可选）
- `--openpcdet-frontend-adaptive-max-denoise-drop-ratio`（默认 `0.02`）
- `--openpcdet-frontend-adaptive-feature-grid-size-xy`（默认 `0.8`）
- `--openpcdet-frontend-adaptive-feature-min-relative-height`（默认 `0.2`）
- `--openpcdet-frontend-adaptive-feature-relative-height-scale`（默认 `1.2`）
- `--openpcdet-frontend-adaptive-feature-boost-strength`（默认 `0.2`）
- `--openpcdet-frontend-adaptive-feature-ground-suppress-strength`（默认 `0.03`）
- `--openpcdet-frontend-adaptive-feature-max-adjust-ratio`（默认 `0.35`，超过则跳过增强）
- `--openpcdet-frontend-adaptive-intensity-lower-percentile`（默认 `1.0`）
- `--openpcdet-frontend-adaptive-intensity-upper-percentile`（默认 `99.0`）
- `--openpcdet-frontend-adaptive-intensity-trigger-low`（默认 `-0.1`）
- `--openpcdet-frontend-adaptive-intensity-trigger-high`（默认 `1.5`）

说明：

- `safe`：仅做保守预处理（去 NaN/Inf、极值裁剪、按 `POINT_CLOUD_RANGE` 裁剪）。
- `denoise`：在 `safe` 基础上做体素邻域去噪。
- `adaptive`：`safe` + 强度自适应修复（当强度分布明显偏离 `[0,1]` 时自动归一化），可选受控去噪。
- `adaptive_ground`：`adaptive` + 近地面稀疏噪点清理（结构保留型，默认带删除比例保护）。
- `adaptive_enhance`：在 `adaptive` 基础上加入“局部高度主体增强”（抑制地面、提升潜在目标点显著性）。
- `temporal_3frame`：默认要求邻帧有位姿才会堆叠；若缺位姿会跳过邻帧。可用 `--openpcdet-frontend-temporal-allow-missing-pose` 恢复旧行为。

---

## 5. 常见问题排查

1. 报 `Missing calibration file`  
检查 `calib` 文件命名是否与帧号匹配（支持 10 位或 6 位编号）。

2. 报 `No indexed prediction files`  
检查输出文件名是否是纯数字（如 `000123.txt`）。

3. `--n-test-images` 自动推断失败  
说明标签和预测不是从 `000000` 连续重叠。  
若你使用了 `--limit`，脚本会自动把同样的数量传给评估。

4. Linux/macOS 上评估失败提示找不到可执行文件  
确认本机可用 `g++`，并重试。

---

## 6. 推荐实践

1. 每次先用 `--limit 200` 做小规模 sanity check。
2. 固定参数后再跑全量，避免重复大规模试错。
3. 每次实验使用独立 `--run-name`，防止结果覆盖。
4. 保留每次导出的 summary JSON，用于后续对比。
