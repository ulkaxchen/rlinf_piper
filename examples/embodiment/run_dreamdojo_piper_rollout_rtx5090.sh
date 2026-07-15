#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
export MEMORY_MODE=alternating
export NUM_CHUNKS=${NUM_CHUNKS:-20}

echo "RTX 5090 mode: Pi0.5 and DreamDojo alternate on one GPU."
exec "${SCRIPT_DIR}/run_dreamdojo_piper_rollout.sh" "$@"
