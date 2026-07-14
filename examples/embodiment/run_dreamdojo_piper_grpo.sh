#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=${WORKSPACE:-/project/peilab/srk/wmpo_workspace}
REPO_PATH=${REPO_PATH:-${WORKSPACE}/RLinf}
EMBODIED_PATH=${EMBODIED_PATH:-${REPO_PATH}/examples/embodiment}
APPTAINER_IMAGE=${APPTAINER_IMAGE:-${WORKSPACE}/rlinf-dreamdojo-openpi-cu128.sandbox}
APPTAINER_PYTHON=${APPTAINER_PYTHON:-/opt/venv/dreamdojo-openpi/bin/python}
CONFIG_NAME=${CONFIG_NAME:-dreamdojo_piper_grpo}

# ---------------------------------------------------------------------------
# Server deployment paths. Edit these defaults here, or export variables with
# the same names before launching. Command-line Hydra overrides are applied
# last and therefore still take precedence over all values in this block.
# ---------------------------------------------------------------------------
DREAMDOJO_REPO_PATH=${DREAMDOJO_REPO_PATH:-${WORKSPACE}/DreamDojo}
KAI0_REPO_PATH=${KAI0_REPO_PATH:-${WORKSPACE}/kai0}
STUDENT_CKPT_PATH=${STUDENT_CKPT_PATH:-${REPO_PATH}/checkpoints/dreamdojo_distill_3000}
CR1_EMBEDDINGS_PATH=${CR1_EMBEDDINGS_PATH:-${REPO_PATH}/checkpoints/cosmos-predict2.5-2B/robot/action-cond/cr1_empty_string_text_embeddings.pt}
VLA_CKPT_PATH=${VLA_CKPT_PATH:-${KAI0_REPO_PATH}/checkpoints/pi05_piper_insert_mouse_battery_normal/piper_insert_mouse_battery_run2/30000_pytorch}
REWARD_CKPT_PATH=${REWARD_CKPT_PATH:-${WORKSPACE}/piper_data/insert-mouse-battery/reward_model/full_weights.pt}
RESET_DATA_PATH=${RESET_DATA_PATH:-${WORKSPACE}/piper_data/insert-mouse-battery/piper_initial_frames_36}
ACTION_STATS_PATH=${ACTION_STATS_PATH:-${WORKSPACE}/piper_data/insert-mouse-battery/piper_insert_mouse_battery_lerobot/meta/stats.json}

# Comma-separated Apptainer bind specifications. Add server storage roots here
# when the paths above are outside /project, for example /data:/data.
APPTAINER_BIND_PATHS=${APPTAINER_BIND_PATHS:-/project:/project}
SKIP_PATH_CHECKS=${SKIP_PATH_CHECKS:-0}

HF_HOME_DIR=${HF_HOME_DIR:-${DREAMDOJO_REPO_PATH}/.cache/huggingface}
DREAMDOJO_SITE=${DREAMDOJO_SITE:-${WORKSPACE}/.dreamdojo_site_min}
DREAMDOJO_DISABLE_SAMPLE_TQDM=${DREAMDOJO_DISABLE_SAMPLE_TQDM:-1}
RLINF_RAY_INCLUDE_DASHBOARD=${RLINF_RAY_INCLUDE_DASHBOARD:-1}
LOG_ROOT=${LOG_ROOT:-${REPO_PATH}/logs}
LOG_DIR=${LOG_DIR:-${LOG_ROOT}/$(date +'%Y%m%d-%H%M%S')-${CONFIG_NAME}}

mkdir -p "${LOG_DIR}"
exec > >(tee "${LOG_DIR}/run.log") 2>&1

INNER_CMD=$(cat <<'EOF'
set -euo pipefail

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy

export WORKSPACE=${WORKSPACE}
export REPO_PATH=${REPO_PATH}
export EMBODIED_PATH=${EMBODIED_PATH}
export HF_HOME=${HF_HOME_DIR}
export HUGGINGFACE_HUB_CACHE=${HF_HOME_DIR}/hub
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export DREAMDOJO_DISABLE_SAMPLE_TQDM=${DREAMDOJO_DISABLE_SAMPLE_TQDM}
export RLINF_RAY_INCLUDE_DASHBOARD=${RLINF_RAY_INCLUDE_DASHBOARD}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

export PYTHONPATH=${REPO_PATH}/examples/embodiment/compat_site:${KAI0_REPO_PATH}/src:${KAI0_REPO_PATH}/packages/openpi-client/src:${DREAMDOJO_REPO_PATH}/packages/cosmos-cuda:${DREAMDOJO_REPO_PATH}/packages/cosmos-oss:${DREAMDOJO_REPO_PATH}:${REPO_PATH}:${PYTHONPATH:-}

if [[ "${SKIP_PATH_CHECKS}" != "1" ]]; then
  [[ -d "${DREAMDOJO_REPO_PATH}" ]] || { echo "Missing DreamDojo repo: ${DREAMDOJO_REPO_PATH}" >&2; exit 2; }
  [[ -d "${KAI0_REPO_PATH}" ]] || { echo "Missing OpenPI/kai0 repo: ${KAI0_REPO_PATH}" >&2; exit 2; }
  if [[ "${STUDENT_CKPT_PATH}" != s3://* && "${STUDENT_CKPT_PATH}" != msc://* ]]; then
    [[ -f "${STUDENT_CKPT_PATH}/model/.metadata" ]] || { echo "Invalid student DCP root: ${STUDENT_CKPT_PATH}/model/.metadata is missing" >&2; exit 2; }
  fi
  [[ -e "${VLA_CKPT_PATH}" ]] || { echo "Missing VLA checkpoint: ${VLA_CKPT_PATH}" >&2; exit 2; }
  [[ -f "${CR1_EMBEDDINGS_PATH}" ]] || { echo "Missing CR1 embedding cache: ${CR1_EMBEDDINGS_PATH}" >&2; exit 2; }
  [[ -f "${REWARD_CKPT_PATH}" ]] || { echo "Missing reward checkpoint: ${REWARD_CKPT_PATH}" >&2; exit 2; }
  [[ -d "${RESET_DATA_PATH}" ]] || { echo "Missing 36-frame reset data: ${RESET_DATA_PATH}" >&2; exit 2; }
  [[ -f "${ACTION_STATS_PATH}" ]] || { echo "Missing action statistics: ${ACTION_STATS_PATH}" >&2; exit 2; }
fi

HYDRA_PATH_OVERRIDES=(
  actor.model.model_path="${VLA_CKPT_PATH}"
  rollout.model.model_path="${VLA_CKPT_PATH}"
  env.train.dreamdojo_repo_path="${DREAMDOJO_REPO_PATH}"
  env.eval.dreamdojo_repo_path="${DREAMDOJO_REPO_PATH}"
  env.train.dreamdojo_ckpt_path="${STUDENT_CKPT_PATH}"
  env.eval.dreamdojo_ckpt_path="${STUDENT_CKPT_PATH}"
  env.train.cr1_embeddings_path="${CR1_EMBEDDINGS_PATH}"
  env.eval.cr1_embeddings_path="${CR1_EMBEDDINGS_PATH}"
  env.train.reward_model.model_path="${REWARD_CKPT_PATH}"
  env.eval.reward_model.model_path="${REWARD_CKPT_PATH}"
  env.train.initial_image_path="${RESET_DATA_PATH}"
  env.eval.initial_image_path="${RESET_DATA_PATH}"
  env.train.action_norm_stats_path="${ACTION_STATS_PATH}"
  env.eval.action_norm_stats_path="${ACTION_STATS_PATH}"
)

"${APPTAINER_PYTHON}" "${EMBODIED_PATH}/train_embodied_agent.py" \
  --config-path "${EMBODIED_PATH}/config" \
  --config-name "${CONFIG_NAME}" \
  runner.logger.log_path="${LOG_DIR}" \
  "${HYDRA_PATH_OVERRIDES[@]}" \
  "$@"
EOF
)

export WORKSPACE REPO_PATH EMBODIED_PATH HF_HOME_DIR DREAMDOJO_SITE DREAMDOJO_DISABLE_SAMPLE_TQDM RLINF_RAY_INCLUDE_DASHBOARD CONFIG_NAME LOG_DIR
export DREAMDOJO_REPO_PATH KAI0_REPO_PATH STUDENT_CKPT_PATH CR1_EMBEDDINGS_PATH VLA_CKPT_PATH REWARD_CKPT_PATH RESET_DATA_PATH ACTION_STATS_PATH SKIP_PATH_CHECKS
export APPTAINERENV_WORKSPACE="${WORKSPACE}"
export APPTAINERENV_REPO_PATH="${REPO_PATH}"
export APPTAINERENV_EMBODIED_PATH="${EMBODIED_PATH}"
export APPTAINERENV_HF_HOME_DIR="${HF_HOME_DIR}"
export APPTAINERENV_DREAMDOJO_SITE="${DREAMDOJO_SITE}"
export APPTAINERENV_DREAMDOJO_DISABLE_SAMPLE_TQDM="${DREAMDOJO_DISABLE_SAMPLE_TQDM}"
export APPTAINERENV_RLINF_RAY_INCLUDE_DASHBOARD="${RLINF_RAY_INCLUDE_DASHBOARD}"
export APPTAINERENV_CONFIG_NAME="${CONFIG_NAME}"
export APPTAINERENV_LOG_DIR="${LOG_DIR}"
export APPTAINERENV_APPTAINER_PYTHON="${APPTAINER_PYTHON}"
export APPTAINERENV_DREAMDOJO_REPO_PATH="${DREAMDOJO_REPO_PATH}"
export APPTAINERENV_KAI0_REPO_PATH="${KAI0_REPO_PATH}"
export APPTAINERENV_STUDENT_CKPT_PATH="${STUDENT_CKPT_PATH}"
export APPTAINERENV_CR1_EMBEDDINGS_PATH="${CR1_EMBEDDINGS_PATH}"
export APPTAINERENV_VLA_CKPT_PATH="${VLA_CKPT_PATH}"
export APPTAINERENV_REWARD_CKPT_PATH="${REWARD_CKPT_PATH}"
export APPTAINERENV_RESET_DATA_PATH="${RESET_DATA_PATH}"
export APPTAINERENV_ACTION_STATS_PATH="${ACTION_STATS_PATH}"
export APPTAINERENV_SKIP_PATH_CHECKS="${SKIP_PATH_CHECKS}"

echo "Using image: ${APPTAINER_IMAGE}"
echo "Using config: ${CONFIG_NAME}"
echo "Using log dir: ${LOG_DIR}"
echo "Student DCP: ${STUDENT_CKPT_PATH}"
echo "CR1 cache: ${CR1_EMBEDDINGS_PATH}"
echo "VLA checkpoint: ${VLA_CKPT_PATH}"
echo "Reward checkpoint: ${REWARD_CKPT_PATH}"
echo "Reset data: ${RESET_DATA_PATH}"
echo "Action stats: ${ACTION_STATS_PATH}"
echo "Extra Hydra overrides: $*"
echo "Saving full stdout/stderr to: ${LOG_DIR}/run.log"

APPTAINER_CMD=(
  apptainer exec --nv
  -B "${APPTAINER_BIND_PATHS}"
  "${APPTAINER_IMAGE}"
  bash -lc "${INNER_CMD}" bash "$@"
)

if [[ -n "${JOB_ID:-}" ]]; then
  echo "Launching through srun job ${JOB_ID}"
  srun --jobid="${JOB_ID}" --overlap --nodes=1 --ntasks=1 \
    --gres="${GRES:-gpu:8}" --cpus-per-task="${CPUS_PER_TASK:-16}" \
    bash -lc "module load apptainer && exec \"\$@\"" bash "${APPTAINER_CMD[@]}"
else
  module load apptainer
  "${APPTAINER_CMD[@]}"
fi
