#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
H800_NUM_GPUS=${H800_NUM_GPUS:-8}
export H800_NUM_GPUS

case "${H800_NUM_GPUS}" in
  2|4|8) ;;
  *)
    echo "H800_NUM_GPUS must be 2, 4, or 8, got: ${H800_NUM_GPUS}" >&2
    exit 2
    ;;
esac

export CONFIG_NAME=dreamdojo_piper_student_grpo_h800_multigpu
export VRAM_PRESET=none
export SINGLE_GPU_SERIAL_OFFLOAD=False
export SINGLE_GPU_SERIAL_UNLOAD_ENV=False
export SINGLE_GPU_SERIAL_LAZY_ACTOR_INIT=False
export HOST_MEMORY_GUARD_GB=${HOST_MEMORY_GUARD_GB:-64}
export GRES=${GRES:-gpu:${H800_NUM_GPUS}}

exec "${SCRIPT_DIR}/run_dreamdojo_piper_grpo.sh" student "$@"
