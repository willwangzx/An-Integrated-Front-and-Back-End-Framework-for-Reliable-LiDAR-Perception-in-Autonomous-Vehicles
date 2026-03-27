# KITTI Evaluation Runbook

This repository now includes `kitti_export.py` to export per-frame KITTI prediction files.
It also includes `kitti_devkit_eval.py` to run the bundled C++ KITTI devkit and summarize AP.

## 1) KITTI-calibrated export (recommended)

Use original KITTI object benchmark files:

```bash
python kitti_export.py \
  --bin-dir "<KITTI_ROOT>/training/velodyne" \
  --calib-dir "<KITTI_ROOT>/training/calib" \
  --out-dir "kitti_predictions/data"
```

Optional debug subset:

```bash
python kitti_export.py \
  --bin-dir "<KITTI_ROOT>/training/velodyne" \
  --calib-dir "<KITTI_ROOT>/training/calib" \
  --out-dir "kitti_predictions/data" \
  --limit 200
```

## 2) Uncalibrated debug export (not for AP)

Only for quick local checks:

```bash
python kitti_export.py \
  --pattern "data/velodyne_points/las/*.las" \
  --out-dir "kitti_predictions/data" \
  --allow-uncalibrated
```

This mode can emit KITTI-invalid fields (`alpha=-10`, invalid 2D bbox), so do not use it for official AP.

## 3) Run KITTI devkit evaluation

After export, run:

```bash
python kitti_devkit_eval.py \
  --label-dir "<KITTI_ROOT>/training/label_2" \
  --pred-dir "kitti_predictions/data" \
  --run-name "exp01" \
  --compile
```

`--n-test-images` defaults to `0` and is auto-inferred from the contiguous overlap of labels/predictions.

This prepares:

- `data/object/label_2/*.txt` (devkit input)
- `results/exp01/data/*.txt` (predictions)

Then it runs `cpp/evaluate_object.exe exp01` and writes:

- `results/exp01/kitti_eval_summary.json`

If you only want to evaluate a small debug set:

```bash
python kitti_devkit_eval.py \
  --label-dir "<KITTI_ROOT>/training/label_2" \
  --pred-dir "kitti_predictions/data" \
  --run-name "debug_small" \
  --n-test-images 200
```

## Output format

Each prediction line follows KITTI detection format:

`type truncation occlusion alpha bbox_left bbox_top bbox_right bbox_bottom h w l x y z ry score`

## Important caveat

Current project logic is unsupervised clustering + heuristic class assignment, not a trained detector.
Scores and classes are approximate and intended for baseline evaluation.
