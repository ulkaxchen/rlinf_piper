#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
DEFAULT_REPO_PATH=$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)
REPO_PATH=${REPO_PATH:-${DEFAULT_REPO_PATH}}
REPO_PATH=$(cd -- "${REPO_PATH}" && pwd -P)

# Server defaults for the Piper LeRobot v2.1 dataset. Override any value by
# exporting it before invoking this script.
PIPER_DATASET_PATH=${PIPER_DATASET_PATH:-/project/peilab/yuyangcheng/dreamdojo-distill/datasets/piper_insert_mouse_battery_lerobot}
RESET_DATA_PATH=${RESET_DATA_PATH:-${REPO_PATH}/checkpoints/piper_initial_frames_36}
PYTHON_BIN=${PYTHON_BIN:-${REPO_PATH}/requirements/dreamdojo-piper/.venv/bin/python}
NUM_EPISODES=${NUM_EPISODES:-64}
FRAMES_PER_FILE=${FRAMES_PER_FILE:-36}

[[ -d "${PIPER_DATASET_PATH}" ]] || {
  echo "Missing Piper LeRobot dataset: ${PIPER_DATASET_PATH}" >&2
  exit 2
}
[[ -f "${PIPER_DATASET_PATH}/meta/episodes_stats.jsonl" || -f "${PIPER_DATASET_PATH}/meta/stats.json" ]] || {
  echo "Missing LeRobot action statistics under ${PIPER_DATASET_PATH}/meta" >&2
  exit 2
}
[[ -x "${PYTHON_BIN}" ]] || {
  echo "Missing uv Python: ${PYTHON_BIN}" >&2
  echo "Run 'uv sync --project requirements/dreamdojo-piper --frozen' in ${REPO_PATH}." >&2
  echo "Alternatively, export PYTHON_BIN for another compatible environment." >&2
  exit 2
}

echo "Piper dataset: ${PIPER_DATASET_PATH}"
echo "Reset output: ${RESET_DATA_PATH}"
echo "Python: ${PYTHON_BIN}"
echo "Episodes: ${NUM_EPISODES}; records per episode: ${FRAMES_PER_FILE}"

"${PYTHON_BIN}" "${REPO_PATH}/rlinf/envs/world_model/convert_piper_to_initial_npy.py" \
  --dataset-path "${PIPER_DATASET_PATH}" \
  --out-dir "${RESET_DATA_PATH}" \
  --num-episodes "${NUM_EPISODES}" \
  --frames-per-file "${FRAMES_PER_FILE}" \
  "$@"

echo "Reset trajectories: ${RESET_DATA_PATH}"
echo "Action statistics: ${RESET_DATA_PATH}/action_stats.json"
