# KITTI Evaluation Runbook

This repository now includes `kitti_export.py` to export per-frame KITTI prediction files.
It also includes `kitti_devkit_eval.py` to run the bundled C++ KITTI devkit and summarize AP.

## 1) Quick export from existing LAS frames

Use this when you want a fast local check and already have converted `.las` frames:

```bash
python kitti_export.py   --pattern "data/velodyne_points/las/*.las"   --out-dir "kitti_predictions/data"
```

This creates:

- `kitti_predictions/data/<frame>.txt`
- `kitti_predictions/kitti_export_summary.json`

Note: this mode does not use camera calibration and exports pseudo boxes in LiDAR frame.

## 2) KITTI-style camera-coordinate export (recommended for real evaluation)

Use original KITTI object benchmark files:

- point clouds: `training/velodyne/*.bin`
- calibration: `training/calib/*.txt`

Run:

```bash
python kitti_export.py   --bin-dir "<KITTI_ROOT>/training/velodyne"   --calib-dir "<KITTI_ROOT>/training/calib"   --out-dir "kitti_predictions/data"
```

You can limit frames while debugging:

```bash
python kitti_export.py   --bin-dir "<KITTI_ROOT>/training/velodyne"   --calib-dir "<KITTI_ROOT>/training/calib"   --out-dir "kitti_predictions/data"   --limit 50
```

## 3) Run KITTI devkit evaluation

After export, run:

```bash
python kitti_devkit_eval.py \
  --label-dir "<KITTI_ROOT>/training/label_2" \
  --pred-dir "kitti_predictions/data" \
  --run-name "exp01" \
  --compile
```

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
