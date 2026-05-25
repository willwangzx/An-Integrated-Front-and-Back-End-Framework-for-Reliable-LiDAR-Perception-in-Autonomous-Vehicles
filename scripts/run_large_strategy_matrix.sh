#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

KITTI_ROOT="${1:-${KITTI_ROOT:-${REPO_ROOT}/KITTI_ROOT}}"
OPENPCDET_ROOT="${2:-${OPENPCDET_ROOT:-/home/willw/OpenPCDet}}"
CFG_FILE="${3:-${CFG_FILE:-/home/willw/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml}}"
CKPT="${4:-${CKPT:-${REPO_ROOT}/artifacts/openpcdet_ckpt/pointpillar_7728.pth}}"
LIMIT="${LIMIT:-500}"
DEVICE="${DEVICE:-cpu}"
SCORE_THRESH="${SCORE_THRESH:-0.1}"
OUT_BASE="${OUT_BASE:-${REPO_ROOT}/runs/large_strategy_matrix_${LIMIT}}"

if [[ -x "${REPO_ROOT}/.venv_openpcdet/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv_openpcdet/bin/python}"
elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
fi

mkdir -p "${OUT_BASE}"

run_sweep() {
  local name="$1"
  shift
  local out_root="${OUT_BASE}/${name}"
  local run_prefix="large_${LIMIT}_${name}"

  echo "[strategy] ${name}"
  echo "[out] ${out_root}"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/sweep_openpcdet_frontend.py" \
    --kitti-root "${KITTI_ROOT}" \
    --openpcdet-root "${OPENPCDET_ROOT}" \
    --cfg-file "${CFG_FILE}" \
    --ckpt "${CKPT}" \
    --out-root "${out_root}" \
    --run-prefix "${run_prefix}" \
    --limit "${LIMIT}" \
    --device "${DEVICE}" \
    --score-thresh "${SCORE_THRESH}" \
    "$@"
}

# 1) Baseline vs guarded enhancement (safe).
run_sweep "enhance_safe" \
  --profile-list "adaptive,adaptive_enhance" \
  --adaptive-feature-max-adjust-ratio 0.35

# 2) Baseline vs stronger enhancement.
run_sweep "enhance_aggressive" \
  --profile-list "adaptive,adaptive_enhance" \
  --adaptive-feature-max-adjust-ratio 0.6

# 3) Baseline vs full enhancement (upper bound).
run_sweep "enhance_full" \
  --profile-list "adaptive,adaptive_enhance" \
  --adaptive-feature-max-adjust-ratio 1.0

# 4) Denoise-oriented strategies.
run_sweep "ground_denoise" \
  --profile-list "adaptive,adaptive_ground,denoise" \
  --denoise-voxel-size-list 0.25 \
  --denoise-neighbor-radius-list 1 \
  --denoise-min-neighbor-points-list 4

# Consolidate summaries.
export OUT_BASE
"${PYTHON_BIN}" - <<'PY'
import csv
import json
import os
from pathlib import Path

out_base = Path(os.environ["OUT_BASE"])
summary_rows = []
for sweep_dir in sorted(out_base.glob("*")):
    csv_path = sweep_dir / "sweep_summary.csv"
    if not csv_path.exists():
        continue
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = dict(row)
            row["strategy_group"] = sweep_dir.name
            summary_rows.append(row)

if not summary_rows:
    raise SystemExit("No sweep summaries found.")

# Sort by mean AP descending.
summary_rows.sort(key=lambda r: float(r["mean_3d_moderate"]), reverse=True)

json_path = out_base / "matrix_summary.json"
csv_path = out_base / "matrix_summary.csv"
json_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

fieldnames = [
    "strategy_group",
    "preset",
    "run_tag",
    "frontend_profile",
    "adaptive_feature_max_adjust_ratio",
    "adaptive_enable_denoise",
    "adaptive_enable_feature_enhance",
    "num_frames",
    "total_exported",
    "car_3d_moderate",
    "pedestrian_3d_moderate",
    "cyclist_3d_moderate",
    "mean_3d_moderate",
    "export_summary_json",
    "eval_summary_json",
]
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in summary_rows:
        writer.writerow({k: row.get(k, "") for k in fieldnames})

print(f"[saved] {json_path}")
print(f"[saved] {csv_path}")
print("[best]", summary_rows[0])
PY

echo "[done] large strategy matrix complete"
