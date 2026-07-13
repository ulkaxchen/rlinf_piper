RL with DreamDojo Teacher and Student
=====================================

.. figure:: https://raw.githubusercontent.com/NVIDIA/DreamDojo/main/assets/banner.gif
   :align: center
   :width: 90%

   DreamDojo world models (image credit: NVIDIA DreamDojo).

Train the same pi0.5 Piper policy with GRPO against either the frozen DreamDojo
teacher or distilled student. Use the explicit RTX 5090 and H800 launchers to
keep the world-model variant and hardware topology visible in every run.

Overview
--------

Choose the teacher for the original 35-step rollout or the student for
four-step causal streaming.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      pi0.5 · DreamDojo 2B teacher/student

   .. grid-item-card:: Algorithms
      :text-align: center

      GRPO

   .. grid-item-card:: Tasks
      :text-align: center

      Piper battery insertion

   .. grid-item-card:: Hardware
      :text-align: center

      1 × RTX 5090 · 2/4/8 × H800

Tasks
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 14 18 14 27 27

   * - Environment
     - Task
     - World model
     - RTX 5090 configuration
     - H800 configuration
   * - ``dreamdojo_wm``
     - Piper battery insertion
     - Teacher
     - ``dreamdojo_piper_teacher_grpo_rtx5090_1gpu.yaml``
     - ``dreamdojo_piper_teacher_grpo_h800_multigpu.yaml``
   * - ``dreamdojo_student_wm``
     - Piper battery insertion
     - Student
     - ``dreamdojo_piper_student_grpo_rtx5090_1gpu.yaml``
     - ``dreamdojo_piper_student_grpo_h800_multigpu.yaml``

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Field
     - Description
   * - Observation
     - Three Piper RGB views assembled at ``1440 × 640``. Teacher returns 12
       frames per step; student returns four.
   * - Action
     - Teacher consumes 36 and student consumes 12 14-D pi0.5 actions at 30 Hz.
       Both stride-sample by three into the Piper slice of a 384-D DreamDojo
       action vector at 10 Hz.
   * - Reward
     - The internal ResNet scores every generated frame. Each score is repeated
       over its three corresponding 30 Hz actions, producing ``[B, 36]`` teacher
       or ``[B, 12]`` student rewards.
   * - State
     - Teacher keeps its last frame. Student keeps nine conditioning RGB frames,
       eight historical model actions, and the causal generation position.

Installation
------------

Install the existing DreamDojo and OpenPI dependency bundle:

.. code:: bash

   bash requirements/install.sh embodied \
      --model openpi \
      --env dreamdojo \
      --venv .venv-dreamdojo \
      --python 3.10 \
      --no-root

The student environment reuses the ``dreamdojo`` installer because its runtime
dependencies are identical to the teacher environment.

Download the Model
------------------

Place the local artifacts under ``checkpoints/``:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Path
     - Purpose
   * - ``base_policy_30000``
     - Initial pi0.5 policy and Piper normalization statistics.
   * - ``dreamdojo_wm/model_ema_bf16.pt``
     - Frozen DreamDojo teacher checkpoint.
   * - ``dreamdojo_distill_3000``
     - Distilled DreamDojo DCP checkpoint root containing ``model/.metadata``.
   * - ``reward_model/full_weights.pt``
     - Per-frame Piper success reward model.
   * - ``cosmos-predict2.5-2B/robot/action-cond/cr1_empty_string_text_embeddings.pt``
     - Cached text condition; this avoids loading the online 7B text encoder.

Set ``DREAMDOJO_REPO_PATH`` if the DreamDojo checkout is not next to RLinf.
The launcher accepts either a direct DCP root containing ``model/.metadata`` or
a parent directory and resolves its newest valid ``iter_*`` directory.
Pass the checkpoint root, not its ``model/`` child; the loader appends that
directory itself. The distill-3000 student and EMA use the experiment's native
``max_frames=256`` RoPE table.

Prepare Student Reset Trajectories
----------------------------------

The native student has a distinct reset-time warmup: one initial image and
twelve 10 Hz DreamDojo actions produce twelve frames, after which RL begins
from the last nine generated frames. The warmup actions come from the reset
trajectory's demonstration prefix, not from the VLA, so they are excluded from
the GRPO trajectory and loss.

The reset ``.npy`` files must contain at least 36 raw 30 Hz ``abs_action``
entries. Re-export existing five-frame reset files before running the student:

.. code:: bash

   python rlinf/envs/world_model/convert_piper_to_initial_npy.py \
      --dataset-path /home/fenrir/ubunto_data_2/worldmodel/data/piper_insert_mouse_battery_lerobot \
      --out-dir /home/fenrir/ubunto_data_2/worldmodel/data/piper_initial_frames \
      --num-episodes 64 \
      --frames-per-file 36

Teacher runs do not use this warmup and still reset from one image.

Run It
------

Use one explicit launcher per world model and hardware topology.

RTX 5090
~~~~~~~~

Launch either serial single-GPU recipe from the RLinf root:

.. code:: bash

   source .venv-dreamdojo/bin/activate

   USE_APPTAINER=0 \
   LOCAL_PYTHON=$PWD/.venv-dreamdojo/bin/python \
   bash examples/embodiment/run_dreamdojo_piper_teacher_grpo_rtx5090_1gpu.sh

   USE_APPTAINER=0 \
   LOCAL_PYTHON=$PWD/.venv-dreamdojo/bin/python \
   bash examples/embodiment/run_dreamdojo_piper_student_grpo_rtx5090_1gpu.sh

What this does: each launcher pins actor, rollout, and environment to GPU 0,
enables the serial unload/lazy-actor lifecycle, and selects the matching frozen
teacher or student environment. GRPO updates pi0.5, not DreamDojo.

Run a one-step smoke test before a long job:

.. code:: bash

   WANDB_MODE=disabled LOGGER_BACKENDS='[]' \
   USE_APPTAINER=0 LOCAL_PYTHON=$PWD/.venv-dreamdojo/bin/python \
   SMOKE=1 TRAIN_NUM_ENVS=2 TRAIN_MAX_STEPS_PER_ROLLOUT_EPOCH=12 \
   TRAIN_MAX_EPISODE_STEPS=12 TRAIN_INFERENCE_STEPS=1 \
   ACTOR_MICRO_BATCH_SIZE=1 ACTOR_GLOBAL_BATCH_SIZE=2 \
   bash examples/embodiment/run_dreamdojo_piper_student_grpo_rtx5090_1gpu.sh \
      runner.max_steps=1 runner.val_check_interval=-1

What this does: it first creates the dataset-action student warmup, then
collects one two-trajectory GRPO group with one VLA-controlled latent per
trajectory, computes rewards and advantages, updates the actor once, and
synchronizes the new policy weights back to rollout.

H800 Multi-GPU
~~~~~~~~~~~~~~

Launch on one node with 2, 4, or 8 visible H800 GPUs:

.. code:: bash

   H800_NUM_GPUS=8 \
   bash examples/embodiment/run_dreamdojo_piper_teacher_grpo_h800_multigpu.sh

   H800_NUM_GPUS=8 \
   bash examples/embodiment/run_dreamdojo_piper_student_grpo_h800_multigpu.sh

What this does: Ray creates one actor, rollout, and DreamDojo environment process
per visible GPU. The 16 training environments are split evenly across ranks, and
the actor uses ``micro_batch_size=8`` and ``global_batch_size=256`` for the 512
samples collected in each 32-chunk GRPO step.

.. warning::

   This is single-node data parallelism. It replicates the complete DreamDojo
   world model and ``no_shard`` pi0.5 actor on every H800; it does not shard one
   world model across GPUs. Use 2, 4, or 8 GPUs, keep ``pipeline_stage_num=1``,
   and provision at least 512GB of host RAM for an 8-GPU run. The launcher keeps
   64GB free through the host-memory guard by default.

Run one short student update before the full job:

.. code:: bash

   H800_NUM_GPUS=8 SMOKE=1 ACTION_CHUNKS_PER_TRAJECTORY=1 \
   TRAIN_INFERENCE_STEPS=1 ACTOR_MICRO_BATCH_SIZE=1 \
   ACTOR_GLOBAL_BATCH_SIZE=8 \
   bash examples/embodiment/run_dreamdojo_piper_student_grpo_h800_multigpu.sh \
      runner.max_steps=1 runner.val_check_interval=-1

Memory Behavior
---------------

The RTX 5090 presets enable ``runner.single_gpu_serial_offload``. The student
path also applies four bounded-memory operations:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Mechanism
     - Effect
   * - Inference-only network pruning
     - Skips the distillation fake-score network and releases the teacher after checkpoint loading.
   * - Environment unload
     - Deletes the DreamDojo pipeline between environment phases instead of retaining all weights in host RAM.
   * - Decode swap
     - Moves the student DiT to CPU while the full-resolution causal VAE decodes four frames.
   * - Bucket synchronization
     - Streams policy weights in 128MB CPU buckets instead of retaining a complete pinned-memory snapshot.

The launcher writes ``memory.csv`` beside ``run.log``. The RTX 5090 launchers
keep 12GB free by default; the H800 launchers keep 64GB free. The H800 student
preset also retains the DCP pipeline between GRPO steps and avoids moving the
2B DiT to CPU before every latent decode.

Visualization and Results
-------------------------

Monitor ``actor/policy_loss``, ``actor/total_loss``, ``rollout/rewards``,
``env/return``, and ``env/success_once``. See :doc:`Training metrics
</rst_source/reference/metrics>` for logger configuration.
