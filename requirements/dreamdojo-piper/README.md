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
