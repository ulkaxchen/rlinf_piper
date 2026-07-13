#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_REPO_PATH=$(cd "${SCRIPT_DIR}/../.." && pwd)

REPO_PATH=${REPO_PATH:-${DEFAULT_REPO_PATH}}
WORKSPACE=${WORKSPACE:-$(cd "${REPO_PATH}/.." && pwd)}
EMBODIED_PATH=${EMBODIED_PATH:-${REPO_PATH}/examples/embodiment}
APPTAINER_IMAGE=${APPTAINER_IMAGE:-${WORKSPACE}/rlinf-dreamdojo-openpi-cu128.sandbox}
APPTAINER_PYTHON=${APPTAINER_PYTHON:-/opt/venv/dreamdojo-openpi/bin/python}
LOCAL_PYTHON=${LOCAL_PYTHON:-}
USE_APPTAINER=${USE_APPTAINER:-auto}

DREAMDOJO_VARIANT=${DREAMDOJO_VARIANT:-teacher}
if [[ $# -gt 0 && ( "$1" == "teacher" || "$1" == "student" ) ]]; then
  DREAMDOJO_VARIANT=$1
  shift
fi

case "${DREAMDOJO_VARIANT}" in
  teacher)
    DEFAULT_CONFIG_NAME=dreamdojo_piper_teacher_grpo
    POLICY_ACTIONS_PER_CHUNK=36
    DEFAULT_TRAIN_INFERENCE_STEPS=35
    DEFAULT_EVAL_INFERENCE_STEPS=5
    ;;
  student)
    DEFAULT_CONFIG_NAME=dreamdojo_piper_student_grpo
    POLICY_ACTIONS_PER_CHUNK=12
    DEFAULT_TRAIN_INFERENCE_STEPS=4
    DEFAULT_EVAL_INFERENCE_STEPS=4
    ;;
  *)
    echo "DREAMDOJO_VARIANT must be 'teacher' or 'student', got: ${DREAMDOJO_VARIANT}" >&2
    exit 2
    ;;
esac
CONFIG_NAME=${CONFIG_NAME:-${DEFAULT_CONFIG_NAME}}

# A sparse success reward needs a complete task trajectory. Keep the same
# policy-chunk horizon for teacher and student while respecting their distinct
# action chunk sizes: 36 actions/chunk for teacher, 12 for student.
ACTION_CHUNKS_PER_TRAJECTORY=${ACTION_CHUNKS_PER_TRAJECTORY:-32}
if ! [[ "${ACTION_CHUNKS_PER_TRAJECTORY}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ACTION_CHUNKS_PER_TRAJECTORY must be a positive integer" >&2
  exit 2
fi
ROLLOUT_ACTION_STEPS=$((ACTION_CHUNKS_PER_TRAJECTORY * POLICY_ACTIONS_PER_CHUNK))

CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-${REPO_PATH}/checkpoints}
BASE_POLICY_CKPT=${BASE_POLICY_CKPT:-${CHECKPOINT_ROOT}/base_policy_30000}

resolve_student_dreamdojo_ckpt() {
  local root=${STUDENT_DREAMDOJO_WM_ROOT}
  local candidate=${DREAMDOJO_WM_CKPT:-}
  local latest=

  if [[ -z "${candidate}" ]]; then
    if [[ -f "${root}/model/.metadata" ]]; then
      candidate=${root}
    elif [[ "${root}" == s3://* || "${root}" == msc://* ]]; then
      candidate=${root}
    elif [[ -d "${root}" ]]; then
      while IFS= read -r -d '' iter_dir; do
        if [[ -f "${iter_dir}/model/.metadata" ]]; then
          latest=${iter_dir}
        fi
      done < <(
        find "${root}" -mindepth 1 -maxdepth 1 -type d \
          -name 'iter_*' -print0 | sort -zV
      )
      candidate=${latest}
    fi
  fi

  if [[ -z "${candidate}" ]]; then
    echo "No valid student DCP checkpoint found under ${root}." >&2
    echo "Expected ${root}/model/.metadata or ${root}/iter_*/model/.metadata." >&2
    return 2
  fi

  if [[ "${candidate}" != s3://* && "${candidate}" != msc://* \
    && ! -f "${candidate}/model/.metadata" ]]; then
    echo "Invalid student DCP checkpoint: ${candidate}/model/.metadata is missing." >&2
    return 2
  fi

  DREAMDOJO_WM_CKPT=${candidate}
}

if [[ "${DREAMDOJO_VARIANT}" == "student" ]]; then
  STUDENT_DREAMDOJO_WM_ROOT=${STUDENT_DREAMDOJO_WM_ROOT:-${CHECKPOINT_ROOT}/dreamdojo_distill_3000}
  resolve_student_dreamdojo_ckpt
else
  STUDENT_DREAMDOJO_WM_ROOT=${STUDENT_DREAMDOJO_WM_ROOT:-${CHECKPOINT_ROOT}/dreamdojo_distill_3000}
  DREAMDOJO_WM_CKPT=${DREAMDOJO_WM_CKPT:-${CHECKPOINT_ROOT}/dreamdojo_wm/model_ema_bf16.pt}
fi
REWARD_MODEL_CKPT=${REWARD_MODEL_CKPT:-${CHECKPOINT_ROOT}/reward_model/full_weights.pt}
DREAMDOJO_DISABLE_ONLINE_TEXT_ENCODER=${DREAMDOJO_DISABLE_ONLINE_TEXT_ENCODER:-True}
DREAMDOJO_TEXT_EMBED_CACHE=${DREAMDOJO_TEXT_EMBED_CACHE:-${CHECKPOINT_ROOT}/cosmos-predict2.5-2B/robot/action-cond/cr1_empty_string_text_embeddings.pt}
DREAMDOJO_NEG_TEXT_EMBED_CACHE=${DREAMDOJO_NEG_TEXT_EMBED_CACHE:-${DREAMDOJO_TEXT_EMBED_CACHE}}

DREAMDOJO_REPO_PATH=${DREAMDOJO_REPO_PATH:-${WORKSPACE}/DreamDojo}
KAI0_PATH=${KAI0_PATH:-${WORKSPACE}/kai0}
PIPER_DATA_ROOT=${PIPER_DATA_ROOT:-${WORKSPACE}/data}
INITIAL_IMAGE_PATH=${INITIAL_IMAGE_PATH:-${PIPER_DATA_ROOT}/piper_initial_frames}

if [[ -z "${LOCAL_PYTHON}" ]]; then
  if [[ -x "${REPO_PATH}/.venv-dreamdojo/bin/python" ]]; then
    LOCAL_PYTHON=${REPO_PATH}/.venv-dreamdojo/bin/python
  elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    LOCAL_PYTHON=${VIRTUAL_ENV}/bin/python
  elif [[ -x "${DREAMDOJO_REPO_PATH}/.venv/bin/python" ]]; then
    LOCAL_PYTHON=${DREAMDOJO_REPO_PATH}/.venv/bin/python
  else
    LOCAL_PYTHON=python3
  fi
fi

ACTION_NORM_SOURCE=${ACTION_NORM_SOURCE:-${BASE_POLICY_CKPT}/assets/norm_stats.json}
ACTION_NORM_STATS_PATH=${ACTION_NORM_STATS_PATH:-${BASE_POLICY_CKPT}/assets/dreamdojo_action_stats.json}

HF_HOME_DIR=${HF_HOME_DIR:-${CHECKPOINT_ROOT}/huggingface}
HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
DREAMDOJO_SITE=${DREAMDOJO_SITE:-${WORKSPACE}/.dreamdojo_site_min}
DREAMDOJO_DISABLE_SAMPLE_TQDM=${DREAMDOJO_DISABLE_SAMPLE_TQDM:-1}
RLINF_RAY_INCLUDE_DASHBOARD=${RLINF_RAY_INCLUDE_DASHBOARD:-1}
VRAM_PRESET=${VRAM_PRESET:-32g}
SINGLE_GPU_SERIAL_OFFLOAD=${SINGLE_GPU_SERIAL_OFFLOAD:-True}
# After a complete trajectory is delivered to the actor, unload the inactive
# DreamDojo pipeline from CPU as well as GPU. This avoids the actor update
# overlapping with the world-model's CPU weights on a 64 GiB host.
SINGLE_GPU_SERIAL_UNLOAD_ENV=${SINGLE_GPU_SERIAL_UNLOAD_ENV:-True}
# Rebuild the actor from resumable streamed state after each full rollout. This
# is deliberately slow, but it prevents the actor and DreamDojo CPU weights
# from coexisting on a 64 GiB host and supports continuous multi-step training.
SINGLE_GPU_SERIAL_LAZY_ACTOR_INIT=${SINGLE_GPU_SERIAL_LAZY_ACTOR_INIT:-True}
# OpenPI and Cosmos can each invoke torch.compile internally. On this
# single-GPU setup, their independent Inductor pools fork up to 24 workers
# apiece and exhaust host RAM before the first rollout. Eager execution is
# slower but is the reliable default here; set DISABLE_TORCH_COMPILE=False to
# opt back into compilation on a larger host.
DISABLE_TORCH_COMPILE=${DISABLE_TORCH_COMPILE:-True}
TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-1}
SMOKE=${SMOKE:-0}
# Long teacher and student trajectories both keep Ray workers alive for much
# longer, so guard host memory by default. Set this to 0 to disable it.
# Keep enough reclaimable RAM for the next model/offload transfer. The kernel
# can otherwise select a Ray worker before the two-second guard poll runs.
HOST_MEMORY_GUARD_GB=${HOST_MEMORY_GUARD_GB:-12}
HOST_MEMORY_POLL_SECONDS=${HOST_MEMORY_POLL_SECONDS:-2}
LOG_ROOT=${LOG_ROOT:-${REPO_PATH}/logs}
LOG_DIR=${LOG_DIR:-${LOG_ROOT}/$(date +'%Y%m%d-%H%M%S')-${CONFIG_NAME}}

if [[ "${DISABLE_TORCH_COMPILE}" =~ ^([Tt]rue|1|yes|YES)$ ]]; then
  export TORCH_COMPILE_DISABLE=1
else
  unset TORCH_COMPILE_DISABLE
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  DEFAULT_HF_TOKEN_PATH=${DEFAULT_HF_TOKEN_PATH:-${HOME}/.cache/huggingface/token}
  if [[ -f "${DEFAULT_HF_TOKEN_PATH}" ]]; then
    HF_TOKEN=$(<"${DEFAULT_HF_TOKEN_PATH}")
    export HF_TOKEN
  fi
fi

if [[ ! -f "${ACTION_NORM_STATS_PATH}" && -f "${ACTION_NORM_SOURCE}" ]]; then
  "${LOCAL_PYTHON}" -c '
import json
import os
import sys

src, dst = sys.argv[1], sys.argv[2]
with open(src, "r", encoding="utf-8") as f:
    data = json.load(f)
actions = data.get("norm_stats", {}).get("actions")
if actions is None:
    raise KeyError(f"{src} does not contain norm_stats.actions")
act_min = actions.get("min", actions.get("q01"))
act_max = actions.get("max", actions.get("q99"))
if act_min is None or act_max is None:
    raise KeyError(f"{src} actions entry must contain min/max or q01/q99")
os.makedirs(os.path.dirname(dst), exist_ok=True)
with open(dst, "w", encoding="utf-8") as f:
    json.dump({"action": {"min": act_min[:14], "max": act_max[:14]}}, f, indent=2)
    f.write("\n")
' "${ACTION_NORM_SOURCE}" "${ACTION_NORM_STATS_PATH}"
fi

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
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE}
export DREAMDOJO_DISABLE_SAMPLE_TQDM=${DREAMDOJO_DISABLE_SAMPLE_TQDM}
export RLINF_RAY_INCLUDE_DASHBOARD=${RLINF_RAY_INCLUDE_DASHBOARD}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TORCHINDUCTOR_COMPILE_THREADS
if [[ "${DISABLE_TORCH_COMPILE}" =~ ^([Tt]rue|1|yes|YES)$ ]]; then
  export TORCH_COMPILE_DISABLE=1
else
  unset TORCH_COMPILE_DISABLE
fi

export PYTHONPATH=${REPO_PATH}/examples/embodiment/compat_site:${KAI0_PATH}/src:${KAI0_PATH}/packages/openpi-client/src:${DREAMDOJO_REPO_PATH}/packages/cosmos-cuda:${DREAMDOJO_REPO_PATH}/packages/cosmos-oss:${DREAMDOJO_REPO_PATH}:${REPO_PATH}:${PYTHONPATH:-}

if [[ -n "${H800_NUM_GPUS:-}" ]]; then
  VISIBLE_GPU_COUNT=$("${APPTAINER_PYTHON}" -c \
    'import torch; print(torch.cuda.device_count())')
  if [[ "${VISIBLE_GPU_COUNT}" != "${H800_NUM_GPUS}" ]]; then
    echo "Expected ${H800_NUM_GPUS} visible H800 GPUs, but PyTorch sees ${VISIBLE_GPU_COUNT}." >&2
    echo "Check the Slurm GRES request or CUDA_VISIBLE_DEVICES before launching." >&2
    exit 2
  fi
fi

HYDRA_OVERRIDES=(
  runner.logger.log_path="${LOG_DIR}"
  actor.model.model_path="${BASE_POLICY_CKPT}"
  rollout.model.model_path="${BASE_POLICY_CKPT}"
  env.train.dreamdojo_repo_path="${DREAMDOJO_REPO_PATH}"
  env.eval.dreamdojo_repo_path="${DREAMDOJO_REPO_PATH}"
  env.train.dreamdojo_ckpt_path="${DREAMDOJO_WM_CKPT}"
  env.eval.dreamdojo_ckpt_path="${DREAMDOJO_WM_CKPT}"
  env.train.initial_image_path="${INITIAL_IMAGE_PATH}"
  env.eval.initial_image_path="${INITIAL_IMAGE_PATH}"
  env.train.disable_online_text_encoder="${DREAMDOJO_DISABLE_ONLINE_TEXT_ENCODER}"
  env.eval.disable_online_text_encoder="${DREAMDOJO_DISABLE_ONLINE_TEXT_ENCODER}"
  env.train.action_norm_stats_path="${ACTION_NORM_STATS_PATH}"
  env.eval.action_norm_stats_path="${ACTION_NORM_STATS_PATH}"
  env.train.reward_model.model_path="${REWARD_MODEL_CKPT}"
  env.eval.reward_model.model_path="${REWARD_MODEL_CKPT}"
  +rollout.sampling_params.max_new_tokens="${ROLLOUT_MAX_NEW_TOKENS:-1}"
  env.train.max_steps_per_rollout_epoch="${TRAIN_MAX_STEPS_PER_ROLLOUT_EPOCH:-${ROLLOUT_ACTION_STEPS}}"
  env.train.max_episode_steps="${TRAIN_MAX_EPISODE_STEPS:-${ROLLOUT_ACTION_STEPS}}"
  env.train.num_inference_steps="${TRAIN_INFERENCE_STEPS:-${DEFAULT_TRAIN_INFERENCE_STEPS}}"
  env.eval.max_steps_per_rollout_epoch="${EVAL_MAX_STEPS_PER_ROLLOUT_EPOCH:-${ROLLOUT_ACTION_STEPS}}"
  env.eval.max_episode_steps="${EVAL_MAX_EPISODE_STEPS:-${ROLLOUT_ACTION_STEPS}}"
  env.eval.num_inference_steps="${EVAL_INFERENCE_STEPS:-${DEFAULT_EVAL_INFERENCE_STEPS}}"
)

if [[ "${DREAMDOJO_VARIANT}" == "student" ]]; then
  HYDRA_OVERRIDES+=(
    env.train.cr1_embeddings_path="${DREAMDOJO_TEXT_EMBED_CACHE}"
    env.eval.cr1_embeddings_path="${DREAMDOJO_TEXT_EMBED_CACHE}"
  )
fi

if [[ "${DREAMDOJO_DISABLE_ONLINE_TEXT_ENCODER}" =~ ^([Tt]rue|1|yes|YES)$ ]]; then
  HYDRA_OVERRIDES+=(
    env.train.text_embedding_cache_path="${DREAMDOJO_TEXT_EMBED_CACHE}"
    env.eval.text_embedding_cache_path="${DREAMDOJO_TEXT_EMBED_CACHE}"
    env.train.negative_text_embedding_cache_path="${DREAMDOJO_NEG_TEXT_EMBED_CACHE}"
    env.eval.negative_text_embedding_cache_path="${DREAMDOJO_NEG_TEXT_EMBED_CACHE}"
  )
fi

if [[ "${VRAM_PRESET}" == "32g" ]]; then
  if [[ "${SINGLE_GPU_SERIAL_OFFLOAD}" =~ ^([Tt]rue|1|yes|YES)$ ]]; then
    export RAY_memory_monitor_refresh_ms=${RAY_memory_monitor_refresh_ms:-0}
  fi
  HYDRA_OVERRIDES+=(
    env.train.total_num_envs="${TRAIN_NUM_ENVS:-2}"
    env.train.video_cfg.save_video="${TRAIN_SAVE_VIDEO:-False}"
    env.eval.total_num_envs="${EVAL_NUM_ENVS:-1}"
    env.eval.video_cfg.save_video="${EVAL_SAVE_VIDEO:-False}"
    runner.single_gpu_serial_offload="${SINGLE_GPU_SERIAL_OFFLOAD}"
    runner.single_gpu_serial_unload_env="${SINGLE_GPU_SERIAL_UNLOAD_ENV}"
    runner.single_gpu_serial_lazy_actor_init="${SINGLE_GPU_SERIAL_LAZY_ACTOR_INIT}"
    actor.fsdp_config.save_full_model_weights=False
    actor.model.precision="${MODEL_PRECISION:-bf16}"
    rollout.model.precision="${MODEL_PRECISION:-bf16}"
    actor.micro_batch_size="${ACTOR_MICRO_BATCH_SIZE:-1}"
    actor.global_batch_size="${ACTOR_GLOBAL_BATCH_SIZE:-2}"
    runner.logger.logger_backends="${LOGGER_BACKENDS:-[]}"
  )
fi

# Hardware-specific YAMLs provide their own defaults. Still honor explicit
# environment overrides so H800 smoke/tuning commands can adjust data parallel
# batch sizes without spelling Hydra keys.
if [[ "${VRAM_PRESET}" != "32g" ]]; then
  if [[ -n "${TRAIN_NUM_ENVS:-}" ]]; then
    HYDRA_OVERRIDES+=(env.train.total_num_envs="${TRAIN_NUM_ENVS}")
  fi
  if [[ -n "${EVAL_NUM_ENVS:-}" ]]; then
    HYDRA_OVERRIDES+=(env.eval.total_num_envs="${EVAL_NUM_ENVS}")
  fi
  if [[ -n "${ACTOR_MICRO_BATCH_SIZE:-}" ]]; then
    HYDRA_OVERRIDES+=(actor.micro_batch_size="${ACTOR_MICRO_BATCH_SIZE}")
  fi
  if [[ -n "${ACTOR_GLOBAL_BATCH_SIZE:-}" ]]; then
    HYDRA_OVERRIDES+=(actor.global_batch_size="${ACTOR_GLOBAL_BATCH_SIZE}")
  fi
fi

if [[ "${SMOKE}" == "1" ]]; then
  HYDRA_OVERRIDES+=(
    runner.max_epochs=1
    runner.save_interval=-1
  )
fi

echo "Resolved base policy: ${BASE_POLICY_CKPT}"
echo "DreamDojo variant: ${DREAMDOJO_VARIANT}"
echo "Resolved DreamDojo WM: ${DREAMDOJO_WM_CKPT}"
echo "Resolved reward model: ${REWARD_MODEL_CKPT}"
echo "Resolved DreamDojo repo: ${DREAMDOJO_REPO_PATH}"
echo "Resolved initial image path: ${INITIAL_IMAGE_PATH}"
echo "Resolved action norm stats: ${ACTION_NORM_STATS_PATH}"
echo "DreamDojo online text encoder disabled: ${DREAMDOJO_DISABLE_ONLINE_TEXT_ENCODER}"
echo "Resolved DreamDojo text embedding cache: ${DREAMDOJO_TEXT_EMBED_CACHE}"
echo "Resolved DreamDojo negative text embedding cache: ${DREAMDOJO_NEG_TEXT_EMBED_CACHE}"
echo "Trajectory horizon: ${ACTION_CHUNKS_PER_TRAJECTORY} policy chunks (${ROLLOUT_ACTION_STEPS} 30Hz actions; ${POLICY_ACTIONS_PER_CHUNK} actions/chunk)"
echo "DreamDojo denoise steps: train=${TRAIN_INFERENCE_STEPS:-${DEFAULT_TRAIN_INFERENCE_STEPS}}, eval=${EVAL_INFERENCE_STEPS:-${DEFAULT_EVAL_INFERENCE_STEPS}}"
echo "VRAM preset: ${VRAM_PRESET} (set VRAM_PRESET=none to use the YAML defaults)"
echo "Ray memory monitor refresh: ${RAY_memory_monitor_refresh_ms:-default}"
echo "Unload inactive DreamDojo pipeline: ${SINGLE_GPU_SERIAL_UNLOAD_ENV}"
echo "Lazy actor initialization: ${SINGLE_GPU_SERIAL_LAZY_ACTOR_INIT}"
echo "Host memory guard: ${HOST_MEMORY_GUARD_GB} GiB"
echo "Torch compile disabled: ${DISABLE_TORCH_COMPILE} (Inductor threads: ${TORCHINDUCTOR_COMPILE_THREADS})"

TRAIN_CMD=(
  "${APPTAINER_PYTHON}" "${EMBODIED_PATH}/train_embodied_agent.py"
  --config-path "${EMBODIED_PATH}/config" \
  --config-name "${CONFIG_NAME}"
  "${HYDRA_OVERRIDES[@]}"
  "$@"
)

if ! [[ "${HOST_MEMORY_GUARD_GB}" =~ ^[0-9]+$ ]]; then
  echo "HOST_MEMORY_GUARD_GB must be a non-negative integer" >&2
  exit 2
fi

if (( HOST_MEMORY_GUARD_GB == 0 )); then
  "${TRAIN_CMD[@]}"
  exit $?
fi

MEMORY_LOG_PATH="${LOG_DIR}/memory.csv"
printf 'timestamp,mem_available_kib,swap_free_kib,ray_rss_kib,gpu_used_mib\n' >"${MEMORY_LOG_PATH}"
"${TRAIN_CMD[@]}" &
TRAIN_PID=$!

monitor_host_memory() {
  local threshold_kib=$((HOST_MEMORY_GUARD_GB * 1024 * 1024))
  while kill -0 "${TRAIN_PID}" 2>/dev/null; do
    local available_kib swap_free_kib ray_rss_kib gpu_used_mib
    available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    swap_free_kib=$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)
    ray_rss_kib=$(ps -eo rss=,args= | awk \
      '$0 ~ /ray::|train_embodied_agent.py/ {sum += $1} END {print sum + 0}')
    if command -v nvidia-smi >/dev/null 2>&1; then
      gpu_used_mib=$(nvidia-smi --query-compute-apps=used_memory \
        --format=csv,noheader,nounits 2>/dev/null | \
        awk '{sum += $1} END {print sum + 0}')
    else
      gpu_used_mib=0
    fi
    printf '%s,%s,%s,%s,%s\n' "$(date --iso-8601=seconds)" \
      "${available_kib}" "${swap_free_kib}" "${ray_rss_kib}" \
      "${gpu_used_mib}" >>"${MEMORY_LOG_PATH}"

    if (( available_kib < threshold_kib )); then
      echo "Host memory guard stopping training: MemAvailable fell below ${HOST_MEMORY_GUARD_GB} GiB." >&2
      ps -eo pid,ppid,rss,vsz,comm,args --sort=-rss | head -n 25 \
        >"${LOG_DIR}/memory-guard-processes.log"
      kill -TERM "${TRAIN_PID}" 2>/dev/null || true
      if command -v ray >/dev/null 2>&1; then
        ray stop --force >/dev/null 2>&1 || true
      fi
      sleep 2
      kill -KILL "${TRAIN_PID}" 2>/dev/null || true
      return
    fi
    sleep "${HOST_MEMORY_POLL_SECONDS}"
  done
}

monitor_host_memory &
GUARD_PID=$!
set +e
wait "${TRAIN_PID}"
TRAIN_STATUS=$?
set -e
kill "${GUARD_PID}" 2>/dev/null || true
wait "${GUARD_PID}" 2>/dev/null || true
exit "${TRAIN_STATUS}"
EOF
)

export WORKSPACE REPO_PATH EMBODIED_PATH HF_HOME_DIR HF_HUB_OFFLINE DREAMDOJO_SITE DREAMDOJO_DISABLE_SAMPLE_TQDM RLINF_RAY_INCLUDE_DASHBOARD CONFIG_NAME LOG_DIR DREAMDOJO_VARIANT ACTION_CHUNKS_PER_TRAJECTORY POLICY_ACTIONS_PER_CHUNK ROLLOUT_ACTION_STEPS DEFAULT_TRAIN_INFERENCE_STEPS DEFAULT_EVAL_INFERENCE_STEPS DISABLE_TORCH_COMPILE TORCHINDUCTOR_COMPILE_THREADS
export CHECKPOINT_ROOT BASE_POLICY_CKPT DREAMDOJO_WM_CKPT STUDENT_DREAMDOJO_WM_ROOT REWARD_MODEL_CKPT DREAMDOJO_DISABLE_ONLINE_TEXT_ENCODER DREAMDOJO_TEXT_EMBED_CACHE DREAMDOJO_NEG_TEXT_EMBED_CACHE DREAMDOJO_REPO_PATH KAI0_PATH PIPER_DATA_ROOT INITIAL_IMAGE_PATH ACTION_NORM_STATS_PATH VRAM_PRESET SINGLE_GPU_SERIAL_OFFLOAD SINGLE_GPU_SERIAL_UNLOAD_ENV SINGLE_GPU_SERIAL_LAZY_ACTOR_INIT SMOKE HOST_MEMORY_GUARD_GB HOST_MEMORY_POLL_SECONDS
export APPTAINERENV_WORKSPACE="${WORKSPACE}"
export APPTAINERENV_REPO_PATH="${REPO_PATH}"
export APPTAINERENV_EMBODIED_PATH="${EMBODIED_PATH}"
export APPTAINERENV_HF_HOME_DIR="${HF_HOME_DIR}"
export APPTAINERENV_HF_HUB_OFFLINE="${HF_HUB_OFFLINE}"
export APPTAINERENV_DREAMDOJO_SITE="${DREAMDOJO_SITE}"
export APPTAINERENV_DREAMDOJO_DISABLE_SAMPLE_TQDM="${DREAMDOJO_DISABLE_SAMPLE_TQDM}"
export APPTAINERENV_RLINF_RAY_INCLUDE_DASHBOARD="${RLINF_RAY_INCLUDE_DASHBOARD}"
export APPTAINERENV_CONFIG_NAME="${CONFIG_NAME}"
export APPTAINERENV_DREAMDOJO_VARIANT="${DREAMDOJO_VARIANT}"
export APPTAINERENV_ACTION_CHUNKS_PER_TRAJECTORY="${ACTION_CHUNKS_PER_TRAJECTORY}"
export APPTAINERENV_POLICY_ACTIONS_PER_CHUNK="${POLICY_ACTIONS_PER_CHUNK}"
export APPTAINERENV_ROLLOUT_ACTION_STEPS="${ROLLOUT_ACTION_STEPS}"
export APPTAINERENV_DEFAULT_TRAIN_INFERENCE_STEPS="${DEFAULT_TRAIN_INFERENCE_STEPS}"
export APPTAINERENV_DEFAULT_EVAL_INFERENCE_STEPS="${DEFAULT_EVAL_INFERENCE_STEPS}"
export APPTAINERENV_LOG_DIR="${LOG_DIR}"
export APPTAINERENV_APPTAINER_PYTHON="${APPTAINER_PYTHON}"
export APPTAINERENV_CHECKPOINT_ROOT="${CHECKPOINT_ROOT}"
export APPTAINERENV_BASE_POLICY_CKPT="${BASE_POLICY_CKPT}"
export APPTAINERENV_DREAMDOJO_WM_CKPT="${DREAMDOJO_WM_CKPT}"
export APPTAINERENV_STUDENT_DREAMDOJO_WM_ROOT="${STUDENT_DREAMDOJO_WM_ROOT}"
export APPTAINERENV_REWARD_MODEL_CKPT="${REWARD_MODEL_CKPT}"
export APPTAINERENV_DREAMDOJO_DISABLE_ONLINE_TEXT_ENCODER="${DREAMDOJO_DISABLE_ONLINE_TEXT_ENCODER}"
export APPTAINERENV_DREAMDOJO_TEXT_EMBED_CACHE="${DREAMDOJO_TEXT_EMBED_CACHE}"
export APPTAINERENV_DREAMDOJO_NEG_TEXT_EMBED_CACHE="${DREAMDOJO_NEG_TEXT_EMBED_CACHE}"
export APPTAINERENV_DREAMDOJO_REPO_PATH="${DREAMDOJO_REPO_PATH}"
export APPTAINERENV_KAI0_PATH="${KAI0_PATH}"
export APPTAINERENV_PIPER_DATA_ROOT="${PIPER_DATA_ROOT}"
export APPTAINERENV_INITIAL_IMAGE_PATH="${INITIAL_IMAGE_PATH}"
export APPTAINERENV_ACTION_NORM_STATS_PATH="${ACTION_NORM_STATS_PATH}"
export APPTAINERENV_VRAM_PRESET="${VRAM_PRESET}"
export APPTAINERENV_SINGLE_GPU_SERIAL_OFFLOAD="${SINGLE_GPU_SERIAL_OFFLOAD}"
export APPTAINERENV_SINGLE_GPU_SERIAL_UNLOAD_ENV="${SINGLE_GPU_SERIAL_UNLOAD_ENV}"
export APPTAINERENV_SINGLE_GPU_SERIAL_LAZY_ACTOR_INIT="${SINGLE_GPU_SERIAL_LAZY_ACTOR_INIT}"
export APPTAINERENV_SMOKE="${SMOKE}"
export APPTAINERENV_HOST_MEMORY_GUARD_GB="${HOST_MEMORY_GUARD_GB}"
export APPTAINERENV_HOST_MEMORY_POLL_SECONDS="${HOST_MEMORY_POLL_SECONDS}"
export APPTAINERENV_DISABLE_TORCH_COMPILE="${DISABLE_TORCH_COMPILE}"
export APPTAINERENV_TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS}"
if [[ -n "${H800_NUM_GPUS:-}" ]]; then
  export H800_NUM_GPUS
  export APPTAINERENV_H800_NUM_GPUS="${H800_NUM_GPUS}"
fi
if [[ -n "${TORCH_COMPILE_DISABLE:-}" ]]; then
  export APPTAINERENV_TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE}"
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
  export APPTAINERENV_HF_TOKEN="${HF_TOKEN}"
fi

echo "Using config: ${CONFIG_NAME}"
echo "Using DreamDojo variant: ${DREAMDOJO_VARIANT}"
echo "Using log dir: ${LOG_DIR}"
echo "Extra Hydra overrides: $*"
echo "Saving full stdout/stderr to: ${LOG_DIR}/run.log"

if [[ "${USE_APPTAINER}" == "auto" ]]; then
  if [[ -e "${APPTAINER_IMAGE}" ]]; then
    USE_APPTAINER=1
  else
    USE_APPTAINER=0
  fi
fi

if [[ "${USE_APPTAINER}" == "0" || "${USE_APPTAINER}" == "false" ]]; then
  echo "Using local Python: ${LOCAL_PYTHON}"
  APPTAINER_PYTHON="${LOCAL_PYTHON}" bash -lc "${INNER_CMD}" bash "$@"
elif [[ -n "${JOB_ID:-}" ]]; then
  echo "Using image: ${APPTAINER_IMAGE}"
  echo "Launching through srun job ${JOB_ID}"
  APPTAINER_CMD=(
    apptainer exec --nv
    -B "${WORKSPACE}:${WORKSPACE}"
    "${APPTAINER_IMAGE}"
    bash -lc "${INNER_CMD}" bash "$@"
  )
  if [[ -d /project ]]; then
    APPTAINER_CMD=(apptainer exec --nv -B /project:/project -B "${WORKSPACE}:${WORKSPACE}" "${APPTAINER_IMAGE}" bash -lc "${INNER_CMD}" bash "$@")
  fi
  srun --jobid="${JOB_ID}" --overlap --nodes=1 --ntasks=1 \
    --gres="${GRES:-gpu:8}" --cpus-per-task="${CPUS_PER_TASK:-16}" \
    bash -lc "module load apptainer && exec \"\$@\"" bash "${APPTAINER_CMD[@]}"
else
  echo "Using image: ${APPTAINER_IMAGE}"
  if command -v module >/dev/null 2>&1; then
    module load apptainer || true
  fi
  APPTAINER_CMD=(
    apptainer exec --nv
    -B "${WORKSPACE}:${WORKSPACE}"
    "${APPTAINER_IMAGE}"
    bash -lc "${INNER_CMD}" bash "$@"
  )
  if [[ -d /project ]]; then
    APPTAINER_CMD=(apptainer exec --nv -B /project:/project -B "${WORKSPACE}:${WORKSPACE}" "${APPTAINER_IMAGE}" bash -lc "${INNER_CMD}" bash "$@")
  fi
  "${APPTAINER_CMD[@]}"
fi
