#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO_PATH=${REPO_PATH:-$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)}
EMBODIED_PATH=${EMBODIED_PATH:-${REPO_PATH}/examples/embodiment}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-${REPO_PATH}/checkpoints}
PYTHON_BIN=${PYTHON_BIN:-${REPO_PATH}/requirements/dreamdojo-piper/.venv/bin/python}
MEMORY_MODE=${MEMORY_MODE:-}

if [[ "${MEMORY_MODE}" != "resident" && "${MEMORY_MODE}" != "alternating" ]]; then
  echo "Set MEMORY_MODE to resident or alternating." >&2
  exit 2
fi

DREAMDOJO_REPO_PATH=${DREAMDOJO_REPO_PATH:-/project/peilab/srk/wmpo_workspace/DreamDojo}
KAI0_REPO_PATH=${KAI0_REPO_PATH:-/project/peilab/srk/wmpo_workspace/kai0}
STUDENT_CKPT_PATH=${STUDENT_CKPT_PATH:-${CHECKPOINT_ROOT}/dreamdojo_student_distill/iter_000008000}
COSMOS_TOKENIZER_PATH=${COSMOS_TOKENIZER_PATH:-${CHECKPOINT_ROOT}/cosmos-predict2.5-2B/tokenizer.pth}
COSMOS_REASON1_PATH=${COSMOS_REASON1_PATH:-${CHECKPOINT_ROOT}/Cosmos-Reason1-7B}
CR1_EMBEDDINGS_PATH=${CR1_EMBEDDINGS_PATH:-${CHECKPOINT_ROOT}/cosmos-predict2.5-2B/robot/action-cond/cr1_empty_string_text_embeddings.pt}
VLA_CKPT_PATH=${VLA_CKPT_PATH:-${CHECKPOINT_ROOT}/vla_policy/5000}
RESET_DATA_PATH=${RESET_DATA_PATH:-${CHECKPOINT_ROOT}/piper_initial_frames_36}
ACTION_STATS_PATH=${ACTION_STATS_PATH:-${RESET_DATA_PATH}/action_stats.json}

EPISODE_INDEX=${EPISODE_INDEX:-0}
# One complete GRPO episode: 240 policy actions / 12 actions per chunk.
NUM_CHUNKS=${NUM_CHUNKS:-20}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-4}
FPS=${FPS:-10}
SEED=${SEED:-0}
CUDA_DEVICE=${CUDA_DEVICE:-0}
ROLLOUT_INITIAL_IMAGE=${ROLLOUT_INITIAL_IMAGE:-}
ROLLOUT_INSTRUCTION=${ROLLOUT_INSTRUCTION:-}
APPLY_OPENPI_TRANSFORMERS_OVERLAY=${APPLY_OPENPI_TRANSFORMERS_OVERLAY:-1}
OUTPUT_ROOT=${OUTPUT_ROOT:-${REPO_PATH}/rollout_outputs}
OUTPUT_DIR=${OUTPUT_DIR:-${OUTPUT_ROOT}/$(date +'%Y%m%d-%H%M%S')-${MEMORY_MODE}}

mkdir -p "${OUTPUT_DIR}"
exec > >(tee "${OUTPUT_DIR}/run.log") 2>&1

[[ -x "${PYTHON_BIN}" ]] || {
  echo "Missing unified uv Python: ${PYTHON_BIN}" >&2
  echo "Run: uv sync --project requirements/dreamdojo-piper --frozen" >&2
  exit 2
}
[[ -d "${DREAMDOJO_REPO_PATH}" ]] || {
  echo "Missing DreamDojo repo: ${DREAMDOJO_REPO_PATH}" >&2
  exit 2
}
[[ -d "${KAI0_REPO_PATH}" ]] || {
  echo "Missing kai0 repo: ${KAI0_REPO_PATH}" >&2
  exit 2
}
[[ -f "${STUDENT_CKPT_PATH}/model/.metadata" ]] || {
  echo "Invalid student DCP root: ${STUDENT_CKPT_PATH}" >&2
  exit 2
}
[[ -d "${VLA_CKPT_PATH}" ]] || {
  echo "Missing VLA checkpoint: ${VLA_CKPT_PATH}" >&2
  exit 2
}
[[ -f "${COSMOS_TOKENIZER_PATH}" ]] || {
  echo "Missing Cosmos tokenizer: ${COSMOS_TOKENIZER_PATH}" >&2
  exit 2
}
[[ -d "${COSMOS_REASON1_PATH}" ]] || {
  echo "Missing Cosmos-Reason1-7B snapshot: ${COSMOS_REASON1_PATH}" >&2
  exit 2
}
[[ -f "${CR1_EMBEDDINGS_PATH}" ]] || {
  echo "Missing CR1 compatibility embedding: ${CR1_EMBEDDINGS_PATH}" >&2
  exit 2
}
[[ -d "${RESET_DATA_PATH}" ]] || {
  echo "Missing reset data: ${RESET_DATA_PATH}" >&2
  exit 2
}
[[ -f "${ACTION_STATS_PATH}" ]] || {
  echo "Missing action stats: ${ACTION_STATS_PATH}" >&2
  exit 2
}

# PyTorch's torchvision wheel carries a build-time RPATH. Make the uv-owned
# Torch and CUDA libraries discoverable before importing Transformers/OpenPI.
PYTHON_SITE_PACKAGES=$("${PYTHON_BIN}" -c \
  'import sysconfig; print(sysconfig.get_paths()["purelib"])')
RUNTIME_LIBRARY_DIRS=()
[[ -d "${PYTHON_SITE_PACKAGES}/torch/lib" ]] && \
  RUNTIME_LIBRARY_DIRS+=("${PYTHON_SITE_PACKAGES}/torch/lib")
for lib_dir in "${PYTHON_SITE_PACKAGES}"/nvidia/*/lib; do
  [[ -d "${lib_dir}" ]] && RUNTIME_LIBRARY_DIRS+=("${lib_dir}")
done
if ((${#RUNTIME_LIBRARY_DIRS[@]} > 0)); then
  RUNTIME_LIBRARY_PATH=$(IFS=:; echo "${RUNTIME_LIBRARY_DIRS[*]}")
  export LD_LIBRARY_PATH="${RUNTIME_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
if [[ -d "${PYTHON_SITE_PACKAGES}/nvidia/cuda_nvrtc" ]]; then
  export CUDA_HOME=${CUDA_HOME:-${PYTHON_SITE_PACKAGES}/nvidia/cuda_nvrtc}
fi

export EMBODIED_PATH DREAMDOJO_REPO_PATH KAI0_REPO_PATH
export HF_HOME=${HF_HOME:-${REPO_PATH}/.cache/huggingface}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export JAX_PLATFORMS=${JAX_PLATFORMS:-cpu}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export DREAMDOJO_DISABLE_SAMPLE_TQDM=${DREAMDOJO_DISABLE_SAMPLE_TQDM:-1}
export PYTHONPATH="${REPO_PATH}/examples/embodiment/compat_site:${KAI0_REPO_PATH}/src:${KAI0_REPO_PATH}/packages/openpi-client/src:${DREAMDOJO_REPO_PATH}/packages/cosmos-cuda:${DREAMDOJO_REPO_PATH}/packages/cosmos-oss:${DREAMDOJO_REPO_PATH}:${REPO_PATH}:${PYTHONPATH:-}"

if [[ "${APPLY_OPENPI_TRANSFORMERS_OVERLAY}" == "1" ]]; then
  OVERLAY=${KAI0_REPO_PATH}/src/openpi/models_pytorch/transformers_replace
  [[ -d "${OVERLAY}" ]] || {
    echo "Missing OpenPI Transformers overlay: ${OVERLAY}" >&2
    exit 2
  }
  TRANSFORMERS_DIR=$("${PYTHON_BIN}" -c \
    'import importlib.util; from pathlib import Path; print(Path(importlib.util.find_spec("transformers").origin).parent)')
  cp -a "${OVERLAY}/." "${TRANSFORMERS_DIR}/"
  echo "Applied kai0 Transformers overlay to: ${TRANSFORMERS_DIR}"
fi

CMD=(
  "${PYTHON_BIN}"
  "${EMBODIED_PATH}/rollout_dreamdojo_piper_student.py"
  --memory-mode "${MEMORY_MODE}"
  --vla-checkpoint "${VLA_CKPT_PATH}"
  --student-checkpoint "${STUDENT_CKPT_PATH}"
  --dreamdojo-repo "${DREAMDOJO_REPO_PATH}"
  --cosmos-tokenizer "${COSMOS_TOKENIZER_PATH}"
  --cosmos-reason1 "${COSMOS_REASON1_PATH}"
  --cr1-embeddings "${CR1_EMBEDDINGS_PATH}"
  --reset-data "${RESET_DATA_PATH}"
  --action-stats "${ACTION_STATS_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --episode-index "${EPISODE_INDEX}"
  --num-chunks "${NUM_CHUNKS}"
  --num-inference-steps "${NUM_INFERENCE_STEPS}"
  --fps "${FPS}"
  --seed "${SEED}"
  --cuda-device "${CUDA_DEVICE}"
)
[[ -n "${ROLLOUT_INITIAL_IMAGE}" ]] && \
  CMD+=(--initial-image "${ROLLOUT_INITIAL_IMAGE}")
[[ -n "${ROLLOUT_INSTRUCTION}" ]] && \
  CMD+=(--instruction "${ROLLOUT_INSTRUCTION}")
CMD+=("$@")

echo "Memory mode: ${MEMORY_MODE}"
echo "VLA checkpoint: ${VLA_CKPT_PATH}"
echo "Student checkpoint: ${STUDENT_CKPT_PATH}"
echo "Reset data: ${RESET_DATA_PATH} (episode ${EPISODE_INDEX})"
echo "Output video: ${OUTPUT_DIR}/vla_dreamdojo_rollout.mp4"
"${CMD[@]}"
