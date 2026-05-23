#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

KITTI_ROOT="${1:-${KITTI_ROOT:-${REPO_ROOT}/KITTI_ROOT}}"
OPENPCDET_ROOT="${2:-${OPENPCDET_ROOT:-/home/willw/OpenPCDet}}"
CFG_FILE="${3:-${CFG_FILE:-/home/willw/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml}}"
CKPT="${4:-${CKPT:-${REPO_ROOT}/artifacts/openpcdet_ckpt/pointpillar_7728.pth}}"

LIMIT="${LIMIT:-0}"  # 0 => full dataset
DEVICE="${DEVICE:-cpu}"
SCORE_THRESH="${SCORE_THRESH:-0.1}"
RADIUS_LIST="${RADIUS_LIST:-1,2}"
MIN_POINTS_LIST="${MIN_POINTS_LIST:-3,4,5}"
VOXEL_SIZE_LIST="${VOXEL_SIZE_LIST:-0.25}"
OUT_BASE="${OUT_BASE:-${REPO_ROOT}/runs/full_neighbor_grid}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

if [[ -x "${REPO_ROOT}/.venv_openpcdet/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv_openpcdet/bin/python}"
elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
fi

mkdir -p "${OUT_BASE}"

run_ratio_grid() {
  local ratio="$1"
  local name="$2"
  local out_root="${OUT_BASE}/${name}"
  local done_csv="${out_root}/sweep_summary.csv"

  if [[ "${SKIP_EXISTING}" == "1" && -s "${done_csv}" ]]; then
    echo "[skip] ${name} already completed: ${done_csv}"
    return 0
  fi

  echo "[grid] ${name} ratio=${ratio}"
  echo "[out] ${out_root}"

  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/sweep_openpcdet_frontend.py" \
    --kitti-root "${KITTI_ROOT}" \
    --openpcdet-root "${OPENPCDET_ROOT}" \
    --cfg-file "${CFG_FILE}" \
    --ckpt "${CKPT}" \
    --out-root "${out_root}" \
    --run-prefix "full_neighbor_${name}" \
    --limit "${LIMIT}" \
    --device "${DEVICE}" \
    --score-thresh "${SCORE_THRESH}" \
    --profile-list "adaptive" \
    --adaptive-enable-denoise \
    --adaptive-enable-feature-enhance \
    --adaptive-grid-denoise-params \
    --adaptive-feature-max-adjust-ratio "${ratio}" \
    --denoise-voxel-size-list "${VOXEL_SIZE_LIST}" \
    --denoise-neighbor-radius-list "${RADIUS_LIST}" \
    --denoise-min-neighbor-points-list "${MIN_POINTS_LIST}"
}

run_ratio_grid "0.35" "safe_r0p35"
run_ratio_grid "0.60" "aggr_r0p60"

export OUT_BASE
"${PYTHON_BIN}" - <<'PY'
import csv
import json
import os
from pathlib import Path

out_base = Path(os.environ["OUT_BASE"])
rows = []
for gdir in sorted(out_base.glob("*")):
    csv_path = gdir / "sweep_summary.csv"
    if not csv_path.exists():
        continue
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row = dict(row)
            row["grid_group"] = gdir.name
            rows.append(row)

if not rows:
    raise SystemExit("No neighbor-grid results found.")

for r in rows:
    exp_json = r.get("export_summary_json", "")
    drop_ratio = None
    if exp_json and Path(exp_json).exists():
        payload = json.loads(Path(exp_json).read_text(encoding="utf-8"))
        totals = payload.get("frontend_totals", {})
        num_in = float(totals.get("num_points_in", 0.0) or 0.0)
        num_after = float(totals.get("num_points_after", 0.0) or 0.0)
        if num_in > 0:
            drop_ratio = max(0.0, min(1.0, (num_in - num_after) / num_in))
    r["drop_ratio"] = drop_ratio if drop_ratio is not None else ""

    r["neighbor_radius_voxels"] = int(float(r["denoise_neighbor_radius"]))
    r["min_neighbor_points"] = int(float(r["denoise_min_neighbor_points"]))
    r["map_3d_mod"] = float(r["mean_3d_moderate"])
    r["ped_3d_mod"] = float(r["pedestrian_3d_moderate"])

rows.sort(key=lambda x: (x["grid_group"], x["neighbor_radius_voxels"], x["min_neighbor_points"]))

csv_path = out_base / "neighbor_grid_table.csv"
json_path = out_base / "neighbor_grid_table.json"
md_path = out_base / "neighbor_grid_table.md"

fields = [
    "grid_group",
    "run_tag",
    "adaptive_feature_max_adjust_ratio",
    "neighbor_radius_voxels",
    "min_neighbor_points",
    "drop_ratio",
    "map_3d_mod",
    "ped_3d_mod",
    "car_3d_moderate",
    "cyclist_3d_moderate",
    "num_frames",
    "total_exported",
    "export_summary_json",
    "eval_summary_json",
]
with csv_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in fields})

json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

lines = [
    "# Full Neighbor Grid (Adaptive Denoise + Enhance)",
    "",
    "| group | ratio | radius | min_points | drop_ratio | map_3d_mod | ped_3d_mod |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    drop = r["drop_ratio"]
    drop_text = f"{float(drop):.4f}" if drop != "" else ""
    lines.append(
        f"| {r['grid_group']} | {float(r['adaptive_feature_max_adjust_ratio']):.2f} | "
        f"{r['neighbor_radius_voxels']} | {r['min_neighbor_points']} | {drop_text} | "
        f"{float(r['map_3d_mod']):.3f} | {float(r['ped_3d_mod']):.3f} |"
    )
md_path.write_text("\n".join(lines), encoding="utf-8")

print(f"[saved] {csv_path}")
print(f"[saved] {json_path}")
print(f"[saved] {md_path}")
PY

echo "[done] full neighbor grid complete"
