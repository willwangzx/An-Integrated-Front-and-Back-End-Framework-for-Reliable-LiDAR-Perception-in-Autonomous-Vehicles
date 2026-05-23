# AGENTS.md

## Project Overview

This repository implements an integrated LiDAR perception prototype for autonomous-driving research. The main line of work is a front-end-first reliability framework: point-cloud preprocessing, voxel/statistical enhancement, OpenPCDet-compatible hooks, KITTI export/evaluation utilities, and a lightweight LAS-based perception pipeline.

The codebase is research-oriented. Prefer small, reproducible changes that preserve experiment comparability over broad refactors.

## Key Source Code

Read these files first before making changes:

- `README.md`: project description, pipeline overview, and current research direction.
- `src/config.py`: global pipeline parameters for voxelization, clustering, filtering, BEV, intensity compensation, and temporal fusion.
- `src/pipeline.py`: `LidarPerceptionPipeline` orchestration for LAS frame processing.
- `main.py`: simple LAS pipeline entry point.
- `src/openpcdet_frontend_hook.py`: backward-compatible public exports for OpenPCDet integration.
- `src/frontend/hook.py`: `OpenPCDetFrontendHook` implementation and profile dispatch.
- `src/frontend/preprocess.py`: safe point filtering, clipping, intensity rescaling, and rigid transforms.
- `src/frontend/adaptive_enhance.py`: distance-adaptive voxel-stat feature enhancement.
- `src/frontend/denoise.py`: voxel-density and near-ground sparse point denoising.
- `src/frontend/stats.py`: frontend stats initialization and aggregation.
- `openpcdet_infer_kitti.py`: OpenPCDet inference runner for KITTI `.bin` frames and KITTI-format prediction export.
- `run_kitti_full_pipeline.py`: end-to-end KITTI conversion, inference, evaluation, and report workflow.
- `benchmark.py`: profiled LAS pipeline benchmark.
- `benchmark_param_sweep.py`: parameter sweep benchmark utilities.
- `scripts/sweep_openpcdet_frontend.py`: OpenPCDet frontend experiment sweep driver.
- `tests/test_openpcdet_frontend_hook.py`: unit coverage for the OpenPCDet frontend hook and preprocessing utilities.

Supporting source groups:

- `src/lidar_loader.py`, `src/preprocessing.py`, `src/voxelization.py`, `src/reflectivity_map.py`, `src/temporal_fusion.py`, `src/clustering.py`, `src/object_features.py`, `src/bev_map.py`, `src/visualization.py`: standalone LAS perception pipeline stages.
- `data_processing/kitti_bin_to_las.py`: KITTI `.bin` to LAS conversion.
- `kitti_export.py`, `kitti_devkit_eval.py`, `evaluate_effectiveness.py`: KITTI-format export/evaluation helpers.
- `cpp/`: C++ evaluation support.
- `matlab/`: MATLAB visualization and KITTI helper scripts.
- `docs/`: experiment reports, paper drafts, and workflow notes.

## Data Layout

Expected local data locations are intentionally not fully committed:

- LAS pipeline input: `data/velodyne_points/las/*.las`
- KITTI velodyne input: passed with `--bin-dir`, usually `KITTI/training/velodyne`
- KITTI labels/calibration: passed with `--label-dir` / `--calib-dir` depending on script
- Mock labels for lightweight checks: `data/mock_label_2/`

Do not commit large raw datasets, generated prediction folders, checkpoints, or OpenPCDet workspaces unless explicitly requested.

## Environment

Recommended Python version: `3.9+`.

The README references `requirements.txt`, but this repository currently does not include one. Install dependencies explicitly when needed:

```bash
python -m pip install numpy laspy scikit-learn open3d matplotlib
```

OpenPCDet workflows additionally require an external OpenPCDet checkout, model config, checkpoint, and matching KITTI data. Do not vendor those dependencies into this repository.

## Common Commands

Run the lightweight unit tests:

```bash
python -m unittest tests/test_openpcdet_frontend_hook.py
```

Run the LAS pipeline:

```bash
python main.py
```

Benchmark LAS frames:

```bash
python benchmark.py --pattern 'data/velodyne_points/las/*.las' --limit 0 --out-dir benchmark_results
```

Convert KITTI `.bin` to LAS:

```bash
python data_processing/kitti_bin_to_las.py \
  --bin-dir /path/to/KITTI/training/velodyne \
  --las-dir data/velodyne_points/las \
  --limit 0
```

Run OpenPCDet KITTI inference with the adaptive frontend:

```bash
python openpcdet_infer_kitti.py \
  --openpcdet-root /path/to/OpenPCDet \
  --cfg-file /path/to/OpenPCDet/tools/cfgs/kitti_models/pv_rcnn.yaml \
  --ckpt /path/to/checkpoints/pv_rcnn_8369.pth \
  --bin-dir /path/to/KITTI/training/velodyne \
  --calib-dir /path/to/KITTI/training/calib \
  --out-dir runs/openpcdet_predictions/data \
  --frontend-profile adaptive_enhance \
  --limit 200
```

Run the full KITTI pipeline:

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

## Development Rules

- Keep `src/openpcdet_frontend_hook.py` backward compatible. New frontend logic should live under `src/frontend/` and be re-exported only when needed.
- Keep frontend changes guarded and measurable. Profiles such as `adaptive`, `adaptive_enhance`, and `adaptive_ground` use drop/adjust ratio guards to avoid damaging small-object recall.
- Preserve deterministic behavior for tests and benchmark comparisons. Avoid hidden randomness unless a seed is exposed.
- Prefer vectorized NumPy operations for point-cloud transforms; avoid per-point Python loops in hot paths.
- When modifying point arrays, preserve shape `(N, >=3)` and keep intensity in column 3 when present.
- Do not silently change KITTI coordinate assumptions or output format. KITTI export files must remain compatible with the devkit-style evaluators in this repo.
- Do not remove docs or experiment artifacts unless explicitly asked; many are used as project deliverables.

## Validation Expectations

Before committing changes, run the most relevant available check:

- Frontend/preprocessing changes: `python -m unittest tests/test_openpcdet_frontend_hook.py`
- LAS pipeline changes: run `python main.py` only when LAS sample data is available.
- Benchmark or report script changes: run the script with a small `--limit` if local data and dependencies exist.
- OpenPCDet changes: run with `--limit` on a small KITTI subset when external OpenPCDet assets are available.

If data, checkpoints, GUI/Open3D support, or external OpenPCDet assets are missing, state that explicitly in the final handoff instead of fabricating validation.
