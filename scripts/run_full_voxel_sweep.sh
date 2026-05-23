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
VOXEL_LIST="${VOXEL_LIST:-0.15,0.20,0.25,0.30,0.35}"
OUT_BASE="${OUT_BASE:-${REPO_ROOT}/runs/full_voxel_sweep}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

if [[ -x "${REPO_ROOT}/.venv_openpcdet/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv_openpcdet/bin/python}"
elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
fi

mkdir -p "${OUT_BASE}"

if [[ "${SKIP_EXISTING}" == "1" && -s "${OUT_BASE}/sweep_summary.csv" ]]; then
  echo "[skip] voxel sweep already completed: ${OUT_BASE}/sweep_summary.csv"
else
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/sweep_openpcdet_frontend.py" \
    --kitti-root "${KITTI_ROOT}" \
    --openpcdet-root "${OPENPCDET_ROOT}" \
    --cfg-file "${CFG_FILE}" \
    --ckpt "${CKPT}" \
    --out-root "${OUT_BASE}" \
    --run-prefix "full_voxel_sweep" \
    --limit "${LIMIT}" \
    --device "${DEVICE}" \
    --score-thresh "${SCORE_THRESH}" \
    --profile-list "denoise" \
    --denoise-voxel-size-list "${VOXEL_LIST}" \
    --denoise-neighbor-radius-list 1 \
    --denoise-min-neighbor-points-list 4
fi

export OUT_BASE
"${PYTHON_BIN}" - <<'PY'
import csv
import json
import os
from pathlib import Path

out_base = Path(os.environ["OUT_BASE"])
src = out_base / "sweep_summary.csv"
if not src.exists():
    raise SystemExit(f"Missing summary: {src}")

rows = []
for row in csv.DictReader(src.open("r", encoding="utf-8")):
    num_frames = int(float(row["num_frames"]))
    total_exported = float(row["total_exported"])
    vox = float(row["denoise_voxel_size"])
    rows.append({
        "exp_id": row["run_tag"],
        "voxel_size_xy": vox,
        "voxel_size_z": vox,
        "profile": row["frontend_profile"],
        "n_frames": num_frames,
        "map_3d_mod": float(row["mean_3d_moderate"]),
        "car_3d_mod": float(row["car_3d_moderate"]),
        "ped_3d_mod": float(row["pedestrian_3d_moderate"]),
        "cyc_3d_mod": float(row["cyclist_3d_moderate"]),
        "exports_per_frame": (total_exported / num_frames) if num_frames > 0 else 0.0,
        "total_exported": int(total_exported),
        "eval_summary_json": row.get("eval_summary_json", ""),
    })

rows.sort(key=lambda r: r["map_3d_mod"], reverse=True)

csv_path = out_base / "voxel_sweep_table.csv"
json_path = out_base / "voxel_sweep_table.json"
md_path = out_base / "voxel_sweep_table.md"

fields = [
    "exp_id", "voxel_size_xy", "voxel_size_z", "profile", "n_frames",
    "map_3d_mod", "car_3d_mod", "ped_3d_mod", "cyc_3d_mod",
    "exports_per_frame", "total_exported", "eval_summary_json"
]
with csv_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows:
        w.writerow(r)

json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

lines = [
    "# Full Voxel Sweep (Inference)",
    "",
    "| exp_id | voxel_xy | voxel_z | map_3d_mod | car | ped | cyc | exports/frame |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    lines.append(
        "| {exp_id} | {voxel_size_xy:.2f} | {voxel_size_z:.2f} | {map_3d_mod:.3f} | "
        "{car_3d_mod:.3f} | {ped_3d_mod:.3f} | {cyc_3d_mod:.3f} | {exports_per_frame:.3f} |".format(**r)
    )
md_path.write_text("\n".join(lines), encoding="utf-8")

print(f"[saved] {csv_path}")
print(f"[saved] {json_path}")
print(f"[saved] {md_path}")
PY

echo "[done] full voxel sweep complete"
