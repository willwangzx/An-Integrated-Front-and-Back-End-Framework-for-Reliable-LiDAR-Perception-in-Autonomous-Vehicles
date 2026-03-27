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
