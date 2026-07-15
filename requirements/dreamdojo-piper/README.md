# DreamDojo Piper runtime

This locked project is the unified runtime for the DreamDojo student world
model, kai0/OpenPI policy, and RLinf scheduler. It deliberately stays separate
from RLinf's root environment: the root project pins Torch 2.6, while the full
Cosmos CUDA 12.8 checkpoints require Torch 2.7.

From the RLinf repository root, create the environment with one locked sync:

```bash
uv sync --project requirements/dreamdojo-piper --frozen
```

The project selects CPython 3.10 because NVIDIA publishes the required
`flash-attn`, `natten`, and Transformer Engine CUDA wheels for CPython 3.10.
The resulting interpreter is:

```text
requirements/dreamdojo-piper/.venv/bin/python
```

The launcher imports DreamDojo and kai0 from the server checkouts configured by
`DREAMDOJO_REPO_PATH` and `KAI0_REPO_PATH`. Before training, it copies kai0's
small Transformers overlay into this locked environment and performs real
imports of the student inference and OpenPI modules. OpenPI's JAX config and
tokenizer helpers run on CPU; the PyTorch VLA and Cosmos student keep all eight
GPUs available for training and world-model inference.

Generate the reset trajectories after syncing:

```bash
bash examples/embodiment/generate_dreamdojo_piper_reset_data.sh
```

Then launch inside an eight-GPU Slurm allocation, or pass an allocation ID:

```bash
JOB_ID=<slurm_job_id> bash examples/embodiment/run_dreamdojo_piper_grpo.sh
```

## Closed-loop rollout video smoke test

The rollout-only entrypoint bypasses Ray and GRPO. It selects one reset episode,
uses its initial three-camera frame and 36-action student warmup prefix, then
alternates Pi0.5 action prediction with four-frame DreamDojo student generation.

On a 32 GiB RTX 5090, Reason1 is loaded only while the student is on CPU. During
steady-state rollout, Pi0.5 and DreamDojo alternate between GPU and CPU, and the
student DiT is also moved to CPU for the VAE decode peak:

```bash
CUDA_VISIBLE_DEVICES=0 \
NUM_CHUNKS=20 \
bash examples/embodiment/run_dreamdojo_piper_rollout_rtx5090.sh
```

On H800, Pi0.5, Reason1, the student DiT, and VAE remain resident on one GPU.
This reproduces the memory behavior of one data-parallel rank; a single-video
smoke test does not need all eight GPUs:

```bash
CUDA_VISIBLE_DEVICES=0 \
NUM_CHUNKS=20 \
bash examples/embodiment/run_dreamdojo_piper_rollout_h800.sh
```

To replace the selected episode's first image and instruction, pass a 1440x640
vertical RGB stack in `cam_high`, `cam_left_wrist`, `cam_right_wrist` order:

```bash
ROLLOUT_INITIAL_IMAGE=/path/to/three_camera_stack.png \
ROLLOUT_INSTRUCTION="insert the mouse battery" \
bash examples/embodiment/run_dreamdojo_piper_rollout_h800.sh
```

The selected reset episode still supplies the required 14-dimensional state and
36 demonstration actions. Outputs are written under `rollout_outputs/`:

- `vla_dreamdojo_rollout.mp4`: initial frame, 12 warmup frames, then four frames
  per VLA/DreamDojo chunk. The default complete episode contains 93 frames:
  one initial frame, 12 warmup frames, and 80 closed-loop predicted frames.
- `actions.npy`: the 12x14 Pi0.5 action chunk generated at every iteration.
- `summary.json`: timings, frame counts, action ranges, and CUDA memory snapshots.
- `run.log`: complete stdout and stderr from the smoke run.
