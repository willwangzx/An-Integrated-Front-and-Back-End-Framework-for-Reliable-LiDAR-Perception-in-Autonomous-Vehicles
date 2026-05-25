# Effectiveness Evaluation

Run:

```bash
python evaluate_effectiveness.py
```

Options:

```bash
python evaluate_effectiveness.py --limit 10 --warmup 1 --sample-size 4000 --out-dir effectiveness_results
```

Outputs (default: `effectiveness_results/`):

- `effectiveness_metrics.csv` (per-frame metrics)
- `effectiveness_summary.json` (aggregate metrics)
- `retention_coverage.png`
- `clustering_quality.png`
- `temporal_stability.png`
- `effectiveness_score.png`

Important:

- This evaluation uses unsupervised proxy metrics (no ground-truth labels).
- Use it for trend tracking and regression checks, not absolute detection accuracy.
