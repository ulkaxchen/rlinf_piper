#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
export MEMORY_MODE=resident
export NUM_CHUNKS=${NUM_CHUNKS:-20}

echo "H800 resident mode: Pi0.5, Reason1, student DiT, and VAE stay on one GPU."
echo "This smoke test reproduces one data-parallel rank; it does not reserve all 8 GPUs."
exec "${SCRIPT_DIR}/run_dreamdojo_piper_rollout.sh" "$@"
