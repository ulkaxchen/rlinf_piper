#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=${WORKSPACE:-/project/peilab/srk/wmpo_workspace}
REPO_PATH=${REPO_PATH:-${WORKSPACE}/RLinf}
EMBODIED_PATH=${EMBODIED_PATH:-${REPO_PATH}/examples/embodiment}
APPTAINER_IMAGE=${APPTAINER_IMAGE:-${WORKSPACE}/rlinf-dreamdojo-openpi-cu128.sandbox}
APPTAINER_PYTHON=${APPTAINER_PYTHON:-/opt/venv/dreamdojo-openpi/bin/python}
CONFIG_NAME=${CONFIG_NAME:-dreamdojo_piper_grpo}

HF_HOME_DIR=${HF_HOME_DIR:-${WORKSPACE}/DreamDojo/.cache/huggingface}
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

export PYTHONPATH=${REPO_PATH}/examples/embodiment/compat_site:${WORKSPACE}/kai0/src:${WORKSPACE}/kai0/packages/openpi-client/src:${WORKSPACE}/DreamDojo/packages/cosmos-cuda:${WORKSPACE}/DreamDojo/packages/cosmos-oss:${WORKSPACE}/DreamDojo:${REPO_PATH}:${PYTHONPATH:-}

"${APPTAINER_PYTHON}" "${EMBODIED_PATH}/train_embodied_agent.py" \
  --config-path "${EMBODIED_PATH}/config" \
  --config-name "${CONFIG_NAME}" \
  runner.logger.log_path="${LOG_DIR}" \
  "$@"
EOF
)

export WORKSPACE REPO_PATH EMBODIED_PATH HF_HOME_DIR DREAMDOJO_SITE DREAMDOJO_DISABLE_SAMPLE_TQDM RLINF_RAY_INCLUDE_DASHBOARD CONFIG_NAME LOG_DIR
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

echo "Using image: ${APPTAINER_IMAGE}"
echo "Using config: ${CONFIG_NAME}"
echo "Using log dir: ${LOG_DIR}"
echo "Extra Hydra overrides: $*"
echo "Saving full stdout/stderr to: ${LOG_DIR}/run.log"

APPTAINER_CMD=(
  apptainer exec --nv
  -B /project:/project
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
