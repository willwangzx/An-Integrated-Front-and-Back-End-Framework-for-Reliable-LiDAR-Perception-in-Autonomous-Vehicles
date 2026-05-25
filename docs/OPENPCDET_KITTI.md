# OpenPCDet + KITTI 3D Detection Integration

This document describes how to run OpenPCDet inference and export KITTI-format
prediction files that can be evaluated by this repository's KITTI devkit wrapper.

## 1) Environment

Install OpenPCDet in a dedicated environment (recommended):

```bash
# Example only; follow your OpenPCDet version requirements.
pip install torch torchvision
pip install numpy
```

Then set up OpenPCDet according to its official instructions.

## 2) Run OpenPCDet Inference Export

Use the helper script in this repository:

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

Optional arguments:

- `--limit 200`: only export first 200 frames.
- `--device cpu`: run inference on CPU.
- `--frontend-profile adaptive`: recommended default frontend hook before `prepare_data` (`adaptive` is default).
  - Includes all `safe` steps below.
  - Auto-repairs intensity to `[0,1]` when percentile range clearly falls outside expected bounds.
  - Supports guarded denoise when `--frontend-adaptive-enable-denoise` is set.
- `--frontend-profile adaptive_ground`: adaptive + sparse near-ground denoise with guard.
  - Removes points that are both voxel-sparse and close to local ground.
  - Keeps elevated sparse points to reduce small-object recall damage.
- `--frontend-profile adaptive_enhance`: adaptive + local-height foreground feature enhancement.
  - Uses local minimum ground reference per XY grid cell.
  - Suppresses near-ground/background intensity and boosts points above local ground.
- `--frontend-profile safe`: conservative frontend hook before `prepare_data`.
  - Drops rows containing NaN/Inf.
  - Clips extreme xyz/intensity values.
  - Filters points to `POINT_CLOUD_RANGE` from OpenPCDet `DATA_CONFIG`.
- `--frontend-profile denoise`: `safe` + voxel-neighborhood outlier suppression.
- `--frontend-profile temporal_3frame`: stack `t-1, t, t+1` points before detection.
  - Optional ego-motion compensation via `--frontend-temporal-pose-file`.
  - By default, neighbor frames without pose are skipped (to avoid ghosting).
  - Use `--frontend-temporal-allow-missing-pose` to force no-pose stacking.
- `--frontend-xyz-clip-abs 10000`
- `--frontend-intensity-clip-abs 1000`
- `--frontend-denoise-voxel-size 0.25`
- `--frontend-denoise-neighbor-radius-voxels 1`
- `--frontend-denoise-min-neighbor-points 4`
- `--frontend-adaptive-enable-denoise`
- `--frontend-adaptive-enable-feature-enhance`
- `--frontend-adaptive-max-denoise-drop-ratio 0.02`
- `--frontend-adaptive-feature-grid-size-xy 0.8`
- `--frontend-adaptive-feature-min-relative-height 0.2`
- `--frontend-adaptive-feature-relative-height-scale 1.2`
- `--frontend-adaptive-feature-boost-strength 0.2`
- `--frontend-adaptive-feature-ground-suppress-strength 0.03`
- `--frontend-adaptive-feature-max-adjust-ratio 0.35`
- `--frontend-adaptive-intensity-lower-percentile 1.0`
- `--frontend-adaptive-intensity-upper-percentile 99.0`
- `--frontend-adaptive-intensity-trigger-low -0.1`
- `--frontend-adaptive-intensity-trigger-high 1.5`
- `--frontend-temporal-pose-file /path/to/poses.txt`
  - Pose file supports either:
    - KITTI odometry style: one line per frame, 12 floats (`3x4`), line index = frame id.
    - `frame_id + 12 floats`.

Important: keep frontend changes conservative. Do not directly reuse heuristic
pipeline operators from `src/preprocessing.py` (e.g., aggressive ground removal)
in OpenPCDet inference unless you retrain or at least fine-tune with the same
distribution shift.

For AP gains, treat `denoise` and `temporal_3frame` as train-time + infer-time
paired changes. Inference-only enablement is mainly for ablation and robustness
checks.

## 3) Evaluate with KITTI Devkit Wrapper

After export:

```bash
python kitti_devkit_eval.py \
  --label-dir /path/to/KITTI/training/label_2 \
  --pred-dir kitti_predictions_openpcdet/data \
  --run-name openpcdet_eval \
  --compile \
  --exe-path cpp/evaluate_object
```

On Windows with MSYS2, use:

```bash
python kitti_devkit_eval.py \
  --label-dir D:/KITTI/training/label_2 \
  --pred-dir kitti_predictions_openpcdet/data \
  --run-name openpcdet_eval \
  --compile \
  --bash-path C:/msys64/usr/bin/bash.exe \
  --exe-path cpp/evaluate_object.exe
```

Outputs:

- `results/openpcdet_eval/kitti_eval_summary.json`
- `results/openpcdet_eval/data/*.txt`

## 4) Notes

- OpenPCDet checkpoint and config must match each other.
- Exported class names follow `cfg.CLASS_NAMES` order.
- If your checkpoint predicts classes outside KITTI official 3 classes, evaluation
  should be interpreted accordingly.
- Current frontend hook insertion point is in `__getitem__`: right after points are
  loaded, before `prepare_data`.
- Frontend implementation is now modular under `src/frontend/`:
  - `hook.py` (profile orchestration)
  - `adaptive_enhance.py` (distance-adaptive voxel-stat enhancement)
  - `denoise.py` / `preprocess.py` / `stats.py`
  - `src/openpcdet_frontend_hook.py` is kept as a compatibility export layer.

## 5) Frontend Roadmap (for Real Gains)

- Multi-frame frontend: 3-5 frame temporal stacking with ego-motion compensation.
- Point-cloud denoising: suppress statistical outliers and motion artifacts while
  preserving edges.
- Intensity calibration: normalize by range and laser ring, and keep train/infer
  processing consistent.
- Domain augmentation: rain/fog/sparsification/intensity perturbation during
  training for robustness.
- Reuse this repo's `temporal_fusion.py` and `reflectivity_map.py` as a reliability
  branch to re-score OpenPCDet boxes, instead of replacing the detector.

## 6) Frontend A/B Sweep Script

Use the automated sweep helper to compare frontend profiles/params:

```bash
python scripts/sweep_openpcdet_frontend.py \
  --kitti-root /path/to/KITTI_ROOT \
  --openpcdet-root /path/to/OpenPCDet \
  --cfg-file /path/to/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml \
  --ckpt /path/to/checkpoints/pointpillar_7728.pth \
  --limit 200 \
  --profile-list none,safe,adaptive,adaptive_enhance,denoise \
  --denoise-voxel-size-list 0.2,0.25,0.3 \
  --denoise-min-neighbor-points-list 3,4,5 \
  --compile-eval
```

Preset mode (recommended for quick 200-frame reproduction of this research line):

```bash
python scripts/sweep_openpcdet_frontend.py \
  --kitti-root /path/to/KITTI_ROOT \
  --openpcdet-root /path/to/OpenPCDet \
  --cfg-file /path/to/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml \
  --ckpt /path/to/checkpoints/pointpillar_7728.pth \
  --preset car_gain_safe_200 \
  --compile-eval
```

Or use the one-shot wrapper:

```bash
bash scripts/run_frontend_preset_200.sh \
  /path/to/KITTI_ROOT \
  /path/to/OpenPCDet \
  /path/to/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml \
  /path/to/checkpoints/pointpillar_7728.pth \
  car_gain_safe_200
```

Optional temporal sweep (with pose compensation):

```bash
python scripts/sweep_openpcdet_frontend.py \
  --kitti-root /path/to/KITTI_ROOT \
  --openpcdet-root /path/to/OpenPCDet \
  --cfg-file /path/to/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml \
  --ckpt /path/to/checkpoints/pointpillar_7728.pth \
  --limit 200 \
  --profile-list temporal_3frame \
  --temporal-pose-file-list /path/to/poses.txt
```

Outputs:

- `<out-root>/sweep_summary.json`
- `<out-root>/sweep_summary.csv`
- per-run export/eval summaries referenced in the above files.

## 7) Full-Set Research Suite (Paper-Oriented)

For full KITTI test-set style reruns (all frames, not 200/500/1000 subsets), use:

```bash
OUT_ROOT=runs/full_research_suite_v1 \
LIMIT=0 \
DEVICE=cpu \
bash scripts/run_full_all_and_report.sh \
  /path/to/KITTI_ROOT \
  /path/to/OpenPCDet \
  /path/to/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml \
  /path/to/checkpoints/pointpillar_7728.pth
```

This suite runs three experiment groups and then auto-generates a report:

- full strategy rerun: baseline / conservative / balanced / aggressive / full / ultra / ground / denoise;
- full voxel sweep: `voxel_size` inference sweep (fixed neighbor settings);
- full neighbor orthogonal grid: `neighbor_radius × min_points` under `r=0.35` and `r=0.6`.

Generated artifacts:

- `runs/full_research_suite_v1/strategy_rerun/full_strategy_summary.csv`
- `runs/full_research_suite_v1/voxel_sweep/voxel_sweep_table.csv`
- `runs/full_research_suite_v1/neighbor_grid/neighbor_grid_table.csv`
- `runs/full_research_suite_v1/full_research_report_zh.md`
- `docs/FULL_RESEARCH_REPORT_ZH.md`
