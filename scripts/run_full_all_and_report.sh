#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

KITTI_ROOT="${1:-${KITTI_ROOT:-${REPO_ROOT}/KITTI_ROOT}}"
OPENPCDET_ROOT="${2:-${OPENPCDET_ROOT:-/home/willw/OpenPCDet}}"
CFG_FILE="${3:-${CFG_FILE:-/home/willw/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml}}"
CKPT="${4:-${CKPT:-${REPO_ROOT}/artifacts/openpcdet_ckpt/pointpillar_7728.pth}}"

LIMIT="${LIMIT:-0}"  # 0 => full
DEVICE="${DEVICE:-cpu}"
SCORE_THRESH="${SCORE_THRESH:-0.1}"

OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/runs/full_research_suite}"
STRATEGY_OUT="${STRATEGY_OUT:-${OUT_ROOT}/strategy_rerun}"
VOXEL_OUT="${VOXEL_OUT:-${OUT_ROOT}/voxel_sweep}"
NEIGHBOR_OUT="${NEIGHBOR_OUT:-${OUT_ROOT}/neighbor_grid}"

mkdir -p "${OUT_ROOT}"

echo "[suite] full strategy rerun"
OUT_BASE="${STRATEGY_OUT}" LIMIT="${LIMIT}" DEVICE="${DEVICE}" SCORE_THRESH="${SCORE_THRESH}" \
  bash "${REPO_ROOT}/scripts/run_full_strategy_rerun.sh" \
  "${KITTI_ROOT}" "${OPENPCDET_ROOT}" "${CFG_FILE}" "${CKPT}"

echo "[suite] full voxel sweep"
OUT_BASE="${VOXEL_OUT}" LIMIT="${LIMIT}" DEVICE="${DEVICE}" SCORE_THRESH="${SCORE_THRESH}" \
  bash "${REPO_ROOT}/scripts/run_full_voxel_sweep.sh" \
  "${KITTI_ROOT}" "${OPENPCDET_ROOT}" "${CFG_FILE}" "${CKPT}"

echo "[suite] full neighbor grid"
OUT_BASE="${NEIGHBOR_OUT}" LIMIT="${LIMIT}" DEVICE="${DEVICE}" SCORE_THRESH="${SCORE_THRESH}" \
  bash "${REPO_ROOT}/scripts/run_full_neighbor_grid.sh" \
  "${KITTI_ROOT}" "${OPENPCDET_ROOT}" "${CFG_FILE}" "${CKPT}"

echo "[suite] generate report"
"${REPO_ROOT}/.venv_openpcdet/bin/python" "${REPO_ROOT}/scripts/generate_full_research_report.py" \
  --strategy-csv "${STRATEGY_OUT}/full_strategy_summary.csv" \
  --voxel-csv "${VOXEL_OUT}/voxel_sweep_table.csv" \
  --neighbor-csv "${NEIGHBOR_OUT}/neighbor_grid_table.csv" \
  --out-md "${OUT_ROOT}/full_research_report_zh.md" \
  --out-doc "${REPO_ROOT}/docs/FULL_RESEARCH_REPORT_ZH.md"

echo "[done] full research suite complete"
