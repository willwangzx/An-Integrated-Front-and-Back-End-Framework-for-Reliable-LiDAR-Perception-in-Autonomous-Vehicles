#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <kitti_root> <openpcdet_root> <cfg_file> <ckpt> [preset]" >&2
  echo "Preset options: none, car_gain_safe_200, car_gain_aggressive_200, car_gain_full_200" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

KITTI_ROOT="$1"
OPENPCDET_ROOT="$2"
CFG_FILE="$3"
CKPT="$4"
PRESET="${5:-${PRESET:-car_gain_safe_200}}"

if [[ -x "${REPO_ROOT}/.venv_openpcdet/bin/python" ]]; then
  DEFAULT_PYTHON="${REPO_ROOT}/.venv_openpcdet/bin/python"
elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  DEFAULT_PYTHON="${REPO_ROOT}/.venv/bin/python"
else
  DEFAULT_PYTHON="$(command -v python3 || command -v python || true)"
fi

PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON}}"
OUT_ROOT="${OUT_ROOT:-runs/sweep_openpcdet_frontend}"
RUN_PREFIX="${RUN_PREFIX:-sweep_openpcdet_frontend}"
DEVICE="${DEVICE:-cpu}"
SCORE_THRESH="${SCORE_THRESH:-0.1}"
LIMIT="${LIMIT:-200}"

if [[ -z "${PYTHON_BIN}" ]]; then
  echo "No python interpreter found in PATH." >&2
  exit 1
fi

CMD=(
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/sweep_openpcdet_frontend.py"
  --kitti-root "${KITTI_ROOT}"
  --openpcdet-root "${OPENPCDET_ROOT}"
  --cfg-file "${CFG_FILE}"
  --ckpt "${CKPT}"
  --out-root "${OUT_ROOT}"
  --run-prefix "${RUN_PREFIX}"
  --limit "${LIMIT}"
  --device "${DEVICE}"
  --score-thresh "${SCORE_THRESH}"
  --preset "${PRESET}"
)

if [[ "${COMPILE_EVAL:-1}" == "1" ]]; then
  CMD+=(--compile-eval)
fi

printf '[cmd] %q ' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
