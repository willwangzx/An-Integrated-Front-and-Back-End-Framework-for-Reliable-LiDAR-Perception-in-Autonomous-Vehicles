# Evaluation and Benchmarking

Run the project benchmark with:

```bash
python benchmark.py
```

Useful options:

```bash
python benchmark.py --limit 10 --warmup 1 --out-dir benchmark_results
python benchmark.py --pattern "data/velodyne_points/las/*.las"
```

Generated outputs (in `benchmark_results/` by default):

- `benchmark_metrics.csv`: per-frame stage timings and scene metrics
- `benchmark_summary.json`: aggregate stats (mean, p95, fps, stage means)
- `runtime_per_frame.png`
- `stage_runtime_breakdown.png`
- `runtime_vs_points.png`
- `scene_complexity.png`

Notes:

- `--warmup` frames run normally but are excluded from final metrics.
- Visualizations are generated with `matplotlib`.

## Parameter Sweep Benchmark

To evaluate runtime cost against voxel grid size and other tunable parameters:

```bash
python benchmark_param_sweep.py
```

By default, this runs one-dimensional sweeps for:

- `voxel_size`
- `cluster_eps`
- `cluster_min_points`
- `fusion_window`
- `bev_resolution`

Useful commands:

```bash
# Focus only on voxel size
python benchmark_param_sweep.py --sweep voxel_size=0.1,0.2,0.3,0.4 --limit 12 --warmup 1

# Sweep voxel size + fusion window with custom values
python benchmark_param_sweep.py --sweep voxel_size=0.1,0.2,0.3 --sweep fusion_window=1,5,10,20

# Include repeat runs for stability
python benchmark_param_sweep.py --repeats 3 --limit 10
```

Generated outputs (in `benchmark_param_sweep/` by default):

- `sweep_frame_metrics.csv`: per-frame timings for every experiment
- `sweep_summary.csv`: aggregated runtime metrics per parameter value
- `sweep_summary.json`: metadata + fastest setting summary
- `runtime_vs_<param>.png`: runtime curve for each swept parameter
