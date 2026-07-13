使用 DreamDojo Teacher 和 Student 进行强化学习
==============================================

.. figure:: https://raw.githubusercontent.com/NVIDIA/DreamDojo/main/assets/banner.gif
   :align: center
   :width: 90%

   DreamDojo 世界模型（图片来源：NVIDIA DreamDojo）。

使用冻结的 DreamDojo teacher 或蒸馏后的 student 作为环境，通过 GRPO 训练同一
个 Piper pi0.5 策略。使用明确区分 RTX 5090 与 H800 的入口，让每次运行的世界
模型变体和硬件拓扑都能从命令中直接看出。

概览
----

需要原始 35-step rollout 时选择 teacher；需要 4-step 因果流式生成时选择 student。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      pi0.5 · DreamDojo 2B teacher/student

   .. grid-item-card:: 算法
      :text-align: center

      GRPO

   .. grid-item-card:: 任务
      :text-align: center

      Piper 电池插入

   .. grid-item-card:: 硬件
      :text-align: center

      1 × RTX 5090 · 2/4/8 × H800

任务
~~~~

.. list-table::
   :header-rows: 1
   :widths: 14 18 14 27 27

   * - 环境
     - 任务
     - 世界模型
     - RTX 5090 配置
     - H800 配置
   * - ``dreamdojo_wm``
     - Piper 电池插入
     - Teacher
     - ``dreamdojo_piper_teacher_grpo_rtx5090_1gpu.yaml``
     - ``dreamdojo_piper_teacher_grpo_h800_multigpu.yaml``
   * - ``dreamdojo_student_wm``
     - Piper 电池插入
     - Student
     - ``dreamdojo_piper_student_grpo_rtx5090_1gpu.yaml``
     - ``dreamdojo_piper_student_grpo_h800_multigpu.yaml``

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - 字段
     - 说明
   * - 观测
     - 三路 Piper RGB 视角拼接为 ``1440 × 640``。Teacher 每个 step 返回 12 帧，
       student 返回 4 帧。
   * - 动作
     - Teacher 消耗 36 条、student 消耗 12 条 30 Hz 的 14 维 pi0.5 动作。两者
       都按 stride 3 采样，再写入 10 Hz、384 维 DreamDojo action vector 的 Piper slice。
   * - 奖励
     - 内部 ResNet 给每帧打分，每个分数复制给对应的 3 条 30 Hz 动作，teacher
       奖励形状为 ``[B, 36]``，student 为 ``[B, 12]``。
   * - 状态
     - Teacher 保留最后一帧；student 保留最近 9 帧条件图像、8 条历史模型动作
       和因果生成位置。

安装
----

安装现有 DreamDojo 与 OpenPI 依赖：

.. code:: bash

   bash requirements/install.sh embodied \
      --model openpi \
      --env dreamdojo \
      --venv .venv-dreamdojo \
      --python 3.10 \
      --no-root

student 和 teacher 的运行依赖相同，因此新环境直接复用 ``dreamdojo`` 安装入口。

下载模型
--------

把本地文件放在 ``checkpoints/`` 下：

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - 路径
     - 用途
   * - ``base_policy_30000``
     - pi0.5 初始策略和 Piper 动作归一化统计。
   * - ``dreamdojo_wm/model_ema_bf16.pt``
     - 冻结的 DreamDojo teacher checkpoint。
   * - ``dreamdojo_distill_3000``
     - 蒸馏 DreamDojo 的 DCP checkpoint 根目录，其中包含 ``model/.metadata``。
   * - ``reward_model/full_weights.pt``
     - 对 Piper 每帧成功状态打分的 reward model。
   * - ``cosmos-predict2.5-2B/robot/action-cond/cr1_empty_string_text_embeddings.pt``
     - 缓存文本条件，用于禁用在线 7B text encoder。

如果 DreamDojo 不在 RLinf 相邻目录，设置 ``DREAMDOJO_REPO_PATH``。启动脚本既
可以直接使用包含 ``model/.metadata`` 的 DCP 根目录，也可以从父目录中自动选择
最新且有效的 ``iter_*`` 目录。
传入 checkpoint 根目录，而不是其中的 ``model/`` 子目录；loader 会自行追加该
目录。distill-3000 student 和 EMA 使用当前 experiment 原生的
``max_frames=256`` RoPE 表。

准备 Student Reset 轨迹
-----------------------

原生 student 在 reset 时有独立 warmup：一张初始图像加 12 条 10 Hz DreamDojo
action 先生成 12 帧，随后强化学习从最后 9 张生成图像开始。warmup action 来自
reset 轨迹的 demonstration 前缀，不来自 VLA，因此不会写入 GRPO trajectory 或 loss。

reset 使用的 ``.npy`` 文件必须至少含有 36 条原始 30 Hz ``abs_action``。已有的
5 帧 reset 文件需要先重新导出：

.. code:: bash

   python rlinf/envs/world_model/convert_piper_to_initial_npy.py \
      --dataset-path /home/fenrir/ubunto_data_2/worldmodel/data/piper_insert_mouse_battery_lerobot \
      --out-dir /home/fenrir/ubunto_data_2/worldmodel/data/piper_initial_frames \
      --num-episodes 64 \
      --frames-per-file 36

teacher 不使用这段 warmup，仍然从单张图像 reset。

运行
----

每种世界模型与硬件拓扑都使用独立入口。

RTX 5090
~~~~~~~~

在 RLinf 根目录启动任一单卡串行配方：

.. code:: bash

   source .venv-dreamdojo/bin/activate

   USE_APPTAINER=0 \
   LOCAL_PYTHON=$PWD/.venv-dreamdojo/bin/python \
   bash examples/embodiment/run_dreamdojo_piper_teacher_grpo_rtx5090_1gpu.sh

   USE_APPTAINER=0 \
   LOCAL_PYTHON=$PWD/.venv-dreamdojo/bin/python \
   bash examples/embodiment/run_dreamdojo_piper_student_grpo_rtx5090_1gpu.sh

这两个入口都会把 actor、rollout 和 environment 固定在 GPU 0，并打开串行卸载与
lazy actor 生命周期，再选择对应的冻结 teacher 或 student 环境。GRPO 更新的是
pi0.5，而不是 DreamDojo。

长时间训练前先运行一个 step 的 smoke test：

.. code:: bash

   WANDB_MODE=disabled LOGGER_BACKENDS='[]' \
   USE_APPTAINER=0 LOCAL_PYTHON=$PWD/.venv-dreamdojo/bin/python \
   SMOKE=1 TRAIN_NUM_ENVS=2 TRAIN_MAX_STEPS_PER_ROLLOUT_EPOCH=12 \
   TRAIN_MAX_EPISODE_STEPS=12 TRAIN_INFERENCE_STEPS=1 \
   ACTOR_MICRO_BATCH_SIZE=1 ACTOR_GLOBAL_BATCH_SIZE=2 \
   bash examples/embodiment/run_dreamdojo_piper_student_grpo_rtx5090_1gpu.sh \
      runner.max_steps=1 runner.val_check_interval=-1

这条命令会先构造 dataset-action 的 student warmup，随后收集一组包含两条
trajectory 的 GRPO 数据，为每条 trajectory 生成一个由 VLA 控制的 latent，计算
reward 和 advantage，更新一次 actor，并把新权重同步回 rollout。

H800 多卡
~~~~~~~~~

在单个节点的 2、4 或 8 张可见 H800 上启动：

.. code:: bash

   H800_NUM_GPUS=8 \
   bash examples/embodiment/run_dreamdojo_piper_teacher_grpo_h800_multigpu.sh

   H800_NUM_GPUS=8 \
   bash examples/embodiment/run_dreamdojo_piper_student_grpo_h800_multigpu.sh

Ray 会在每张可见 GPU 上各创建一个 actor、rollout 和 DreamDojo environment
进程。16 个训练环境会平均分配到各 rank；每个 32-chunk GRPO step 产生 512 个
样本，actor 使用 ``micro_batch_size=8`` 与 ``global_batch_size=256``。

.. warning::

   这是单节点数据并行：每张 H800 都会复制完整 DreamDojo 世界模型和
   ``no_shard`` pi0.5 actor，并不会把一个世界模型切分到多卡。请使用 2、4 或
   8 张卡，保持 ``pipeline_stage_num=1``；8 卡运行建议至少准备 512GB 主存。
   启动器默认通过主存 guard 保留 64GB 可用空间。

完整训练前先执行一次短 student update：

.. code:: bash

   H800_NUM_GPUS=8 SMOKE=1 ACTION_CHUNKS_PER_TRAJECTORY=1 \
   TRAIN_INFERENCE_STEPS=1 ACTOR_MICRO_BATCH_SIZE=1 \
   ACTOR_GLOBAL_BATCH_SIZE=8 \
   bash examples/embodiment/run_dreamdojo_piper_student_grpo_h800_multigpu.sh \
      runner.max_steps=1 runner.val_check_interval=-1

内存机制
--------

RTX 5090 preset 会启用 ``runner.single_gpu_serial_offload``。student 路径还包含四个
有界内存机制：

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 机制
     - 效果
   * - 仅加载推理网络
     - 跳过蒸馏训练专用的 fake-score 网络，并在 checkpoint 加载后释放 teacher。
   * - 环境卸载
     - 环境阶段结束后删除 DreamDojo pipeline，避免全部权重常驻主存。
   * - Decode 交换
     - 完整分辨率因果 VAE 解码 4 帧时，暂时把 student DiT 移到 CPU。
   * - Bucket 权重同步
     - 用 128MB CPU bucket 流式同步策略权重，不保留完整 pinned-memory 快照。

启动脚本会在 ``run.log`` 旁写入 ``memory.csv``。RTX 5090 入口默认保留 12GB
可用主存，H800 入口默认保留 64GB。H800 student preset 还会在 GRPO step 之间
保留 DCP pipeline，并避免每个 latent decode 前都把 2B DiT 移到 CPU。

可视化与结果
------------

训练时关注 ``actor/policy_loss``、``actor/total_loss``、``rollout/rewards``、
``env/return`` 与 ``env/success_once``。logger 配置参见 :doc:`训练指标
</rst_source/reference/metrics>`。
