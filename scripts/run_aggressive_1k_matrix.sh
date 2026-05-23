#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

KITTI_ROOT="${1:-${KITTI_ROOT:-${REPO_ROOT}/KITTI_ROOT}}"
OPENPCDET_ROOT="${2:-${OPENPCDET_ROOT:-/home/willw/OpenPCDet}}"
CFG_FILE="${3:-${CFG_FILE:-/home/willw/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml}}"
CKPT="${4:-${CKPT:-${REPO_ROOT}/artifacts/openpcdet_ckpt/pointpillar_7728.pth}}"

LIMIT="${LIMIT:-1000}"
DEVICE="${DEVICE:-cpu}"
SCORE_THRESH="${SCORE_THRESH:-0.1}"
OUT_BASE="${OUT_BASE:-${REPO_ROOT}/runs/aggressive_matrix_${LIMIT}}"

if [[ -x "${REPO_ROOT}/.venv_openpcdet/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv_openpcdet/bin/python}"
elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
fi

mkdir -p "${OUT_BASE}"

run_case() {
  local name="$1"
  local profile="$2"
  local max_adjust_ratio="$3"
  local boost_strength="$4"
  local ground_suppress_strength="$5"

  local out_root="${OUT_BASE}/${name}"
  local run_prefix="aggr_${LIMIT}_${name}"

  echo "[case] ${name}"
  echo "[profile] ${profile}"
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
    --profile-list "${profile}" \
    --adaptive-feature-max-adjust-ratio "${max_adjust_ratio}" \
    --adaptive-feature-boost-strength "${boost_strength}" \
    --adaptive-feature-ground-suppress-strength "${ground_suppress_strength}"
}

# Baseline (no enhancement)
run_case "baseline_adaptive" "adaptive" "0.35" "0.2" "0.03"

# Aggressive tiers
run_case "aggressive_r0p6" "adaptive_enhance" "0.6" "0.2" "0.03"
run_case "aggressive_r0p8" "adaptive_enhance" "0.8" "0.2" "0.03"
run_case "aggressive_r1p0" "adaptive_enhance" "1.0" "0.2" "0.03"
run_case "aggressive_r1p0_boost0p3" "adaptive_enhance" "1.0" "0.3" "0.01"

export OUT_BASE
"${PYTHON_BIN}" - <<'PY'
import csv
import json
import os
from pathlib import Path

out_base = Path(os.environ["OUT_BASE"])
rows = []
for case_dir in sorted(out_base.glob("*")):
    csv_path = case_dir / "sweep_summary.csv"
    if not csv_path.exists():
        continue
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            r = dict(row)
            r["case"] = case_dir.name
            rows.append(r)

if not rows:
    raise SystemExit("No sweep summaries found.")

baseline = next((r for r in rows if r.get("case") == "baseline_adaptive"), None)
if baseline is None:
    raise SystemExit("Missing baseline_adaptive result.")

metrics = [
    "car_3d_moderate",
    "pedestrian_3d_moderate",
    "cyclist_3d_moderate",
    "mean_3d_moderate",
]
base_vals = {k: float(baseline[k]) for k in metrics}

for r in rows:
    for k in metrics:
        r[k] = float(r[k])
        r[f"{k}_delta_vs_baseline"] = r[k] - base_vals[k]

rows.sort(key=lambda r: r["mean_3d_moderate"], reverse=True)

json_path = out_base / "aggressive_1k_summary.json"
csv_path = out_base / "aggressive_1k_summary.csv"
md_path = out_base / "aggressive_1k_summary.md"

json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

fieldnames = [
    "case",
    "frontend_profile",
    "adaptive_feature_max_adjust_ratio",
    "adaptive_feature_boost_strength",
    "adaptive_feature_ground_suppress_strength",
    "num_frames",
    "total_exported",
    "car_3d_moderate",
    "car_3d_moderate_delta_vs_baseline",
    "pedestrian_3d_moderate",
    "pedestrian_3d_moderate_delta_vs_baseline",
    "cyclist_3d_moderate",
    "cyclist_3d_moderate_delta_vs_baseline",
    "mean_3d_moderate",
    "mean_3d_moderate_delta_vs_baseline",
    "eval_summary_json",
]
with csv_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in fieldnames})

lines = [
    "# Aggressive 1K Matrix",
    "",
    "| case | profile | ratio | boost | suppress | car | ped | cyc | mean |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    lines.append(
        "| {case} | {profile} | {ratio:.2f} | {boost:.2f} | {suppress:.2f} | "
        "{car:.3f} ({dcar:+.3f}) | {ped:.3f} ({dped:+.3f}) | "
        "{cyc:.3f} ({dcyc:+.3f}) | {mean:.3f} ({dmean:+.3f}) |".format(
            case=r["case"],
            profile=r["frontend_profile"],
            ratio=float(r["adaptive_feature_max_adjust_ratio"]),
            boost=float(r["adaptive_feature_boost_strength"]),
            suppress=float(r["adaptive_feature_ground_suppress_strength"]),
            car=r["car_3d_moderate"],
            dcar=r["car_3d_moderate_delta_vs_baseline"],
            ped=r["pedestrian_3d_moderate"],
            dped=r["pedestrian_3d_moderate_delta_vs_baseline"],
            cyc=r["cyclist_3d_moderate"],
            dcyc=r["cyclist_3d_moderate_delta_vs_baseline"],
            mean=r["mean_3d_moderate"],
            dmean=r["mean_3d_moderate_delta_vs_baseline"],
        )
    )
md_path.write_text("\n".join(lines), encoding="utf-8")

print(f"[saved] {json_path}")
print(f"[saved] {csv_path}")
print(f"[saved] {md_path}")
best_car = max(rows, key=lambda r: r["car_3d_moderate"])
print("[best_car]", best_car["case"], best_car["car_3d_moderate"])
PY

echo "[done] aggressive 1k matrix complete"
