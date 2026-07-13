#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export CONFIG_NAME=dreamdojo_piper_teacher_grpo_rtx5090_1gpu
export VRAM_PRESET=none
export SINGLE_GPU_SERIAL_OFFLOAD=True
export SINGLE_GPU_SERIAL_UNLOAD_ENV=True
export SINGLE_GPU_SERIAL_LAZY_ACTOR_INIT=True

exec "${SCRIPT_DIR}/run_dreamdojo_piper_grpo.sh" teacher "$@"
