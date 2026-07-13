# DreamDojo Piper GRPO 训练与 GPU 生命周期

本文说明当前 RLinf checkout 中 DreamDojo Piper GRPO 的端到端训练流程、
teacher/student 时序差异、关键参数、模型规模，以及 RTX 5090 单卡串行和
H800 单机多卡两种资源模式。运行命令见[第 8 节](#8-运行和监控)。

最重要的边界是：DreamDojo teacher 和 student 都是冻结的世界模型环境。
它们负责根据动作生成未来视频，再由环境内的 ResNet 计算 reward；GRPO 实际
更新的是 pi0.5 的 action expert 和 action projection，不更新 DreamDojo。

相关入口：

- RTX 5090 teacher：[`run_dreamdojo_piper_teacher_grpo_rtx5090_1gpu.sh`](run_dreamdojo_piper_teacher_grpo_rtx5090_1gpu.sh)
- RTX 5090 student：[`run_dreamdojo_piper_student_grpo_rtx5090_1gpu.sh`](run_dreamdojo_piper_student_grpo_rtx5090_1gpu.sh)
- H800 teacher：[`run_dreamdojo_piper_teacher_grpo_h800_multigpu.sh`](run_dreamdojo_piper_teacher_grpo_h800_multigpu.sh)
- H800 student：[`run_dreamdojo_piper_student_grpo_h800_multigpu.sh`](run_dreamdojo_piper_student_grpo_h800_multigpu.sh)
- 兼容调度器：[`run_dreamdojo_piper_grpo.sh`](run_dreamdojo_piper_grpo.sh)
- 训练入口：[`train_embodied_agent.py`](train_embodied_agent.py)
- Runner：[`../../rlinf/runners/embodied_runner.py`](../../rlinf/runners/embodied_runner.py)
- Teacher 环境：[`../../rlinf/envs/world_model/world_model_dreamdojo_env.py`](../../rlinf/envs/world_model/world_model_dreamdojo_env.py)
- Student 环境：[`../../rlinf/envs/world_model/world_model_dreamdojo_student_env.py`](../../rlinf/envs/world_model/world_model_dreamdojo_student_env.py)

## 1. 组件和模型

`train_embodied_agent.py` 使用 Ray 创建以下 worker group：

| 组件 | 作用 |
| --- | --- |
| `ActorGroup` | 加载 FSDP pi0.5，计算 GRPO loss 并更新参数 |
| `RolloutGroup` | 使用当前 pi0.5 根据观察生成 action 和旧策略 logprob |
| `EnvGroup` | 运行 DreamDojo 世界模型、环境状态和内部 reward model |
| `EmbodiedRunner` | 调度 rollout、advantage、训练、保存和 worker 生命周期 |

当前配置为 `algorithm.loss_type: actor`，因此 actor worker 是
`EmbodiedFSDPActor`，不是 SAC、DAGGER 或 pipeline actor。

### 1.1 RTX 5090 单卡核心参数

两个单卡入口都将 actor、rollout 和 env 放到同一个逻辑 GPU 0，并启用
`single_gpu_serial_offload`、`single_gpu_serial_unload_env` 和
`single_gpu_serial_lazy_actor_init`。同一时刻只让 rollout VLA、DreamDojo 或
训练 actor 中的一类大模型使用 GPU。

| 参数 | Teacher | Student |
| --- | --- | --- |
| 入口 | `run_dreamdojo_piper_teacher_grpo_rtx5090_1gpu.sh` | `run_dreamdojo_piper_student_grpo_rtx5090_1gpu.sh` |
| 冻结环境 | `dreamdojo_wm` | `dreamdojo_student_wm` |
| GRPO group | 2 条同初始 episode 的 trajectory | 2 条同初始 episode 的 trajectory |
| Policy action/chunk | `[2, 36, 14]` | `[2, 12, 14]` |
| stride 后的模型 action | `[2, 12, 14]` | `[2, 4, 14]` |
| 每 chunk 生成 | 12 张 RGB 帧 | 1 个 latent，即 4 张 RGB 帧 |
| 默认 inference steps | 35 | 4 |
| 默认 trajectory | 32 chunks，1152 个 30 Hz action | 32 chunks，384 个 30 Hz action |
| 每条 trajectory 的 RL 帧 | 384 | 128，另有 12 张 reset warmup 帧 |
| Reward shape/chunk | `[2, 12]` repeat 3 为 `[2, 36]` | `[2, 4]` repeat 3 为 `[2, 12]` |
| Chunk 间世界模型状态 | pipeline 搬到 CPU，完整 trajectory 后删除 | 保留 9 帧/8 action，pipeline 每 chunk 删除并从 DCP 重建 |
| Actor batch | `micro_batch_size=1`，`global_batch_size=2` | 同 teacher |
| 每个 global step | 64 个 chunk sample，32 次 `optimizer.step()` | 64 个 chunk sample，32 次 `optimizer.step()` |

两种默认配置只对齐了 32 个 policy chunk，没有对齐物理时长。Teacher 每条
trajectory 是 38.4 秒，student 是 12.8 秒。若要让 student 与 teacher 都覆盖
1152 个 30 Hz action，运行 student 时设置
`ACTION_CHUNKS_PER_TRAJECTORY=96`。

### 1.2 模型规模

以下参数量和磁盘大小来自本机当前 checkpoint：

| 模型 | 参数量 | 本地磁盘占用 | 是否训练 |
| --- | ---: | ---: | --- |
| pi0.5 VLA | 3.617B | `base_policy_30000` 约 6.8 GB | 部分训练 |
| PaliGemma 主干 | 2.923B | 包含在 pi0.5 中 | 冻结 |
| Gemma action expert | 0.691B | 包含在 pi0.5 中 | 训练 |
| Action projection | 约 2.2M | 包含在 pi0.5 中 | 训练 |
| DreamDojo teacher DiT | 2.151B | 实际 BF16 权重约 4.3 GB | 冻结推理 |
| DreamDojo student DiT | 2.151B | 完整 DCP 约 34 GB；推理读取的 `model/` 约 17 GB | 冻结推理 |
| Reward ResNet | 约 11.2M | 约 43 MB | 冻结推理 |
| Reason1/Qwen text encoder | 7B | 当前禁用 | 不加载 |
| Cosmos video tokenizer | - | `tokenizer.pth` 约 508 MB | 冻结推理 |

`checkpoints/dreamdojo_wm` 整个目录约 45 GB，是因为同时保存了 FP32、
BF16、EMA、DCP model 和 optimizer。Teacher RL 实际加载的是：

```text
checkpoints/dreamdojo_wm/model_ema_bf16.pt
```

它约 4.3 GB，并不会把整个 45 GB 目录加载到内存。

Student RL 传入的是 DCP 根目录
`checkpoints/dreamdojo_distill_3000`；DreamDojo loader 会自行追加 `model/`，
不要把配置写成 `.../dreamdojo_distill_3000/model`。完整目录约 34 GB，但推理
只读取约 17 GB 的 `model/`，不会加载 `optim_*`、`scheduler_*` 或 `trainer/`。
该 checkpoint 的 `net`/`net_ema` RoPE 序列长度为 256，配置保持当前
self-forcing experiment 的 `max_frames=256`，不再使用旧的128-frame override。

pi0.5 当前设置为：

```yaml
openpi:
  train_expert_only: true
```

因此这是完整的 rollout -> reward -> GRPO 训练流程，但不是全参数 VLA
微调。PaliGemma 视觉语言主干冻结，实际更新约 0.693B action expert 和
projection 参数。

## 2. 启动和 Ray placement

四个用户入口共享同一个兼容调度器，执行路径如下：

```text
run_dreamdojo_piper_{teacher,student}_grpo_{rtx5090_1gpu,h800_multigpu}.sh
  -> run_dreamdojo_piper_grpo.sh
  -> train_embodied_agent.py
  -> EmbodiedRunner.init_workers()
  -> EmbodiedRunner.run()
```

RTX 5090 preset 明确固定到第一个可见的逻辑 GPU 0：

```yaml
cluster:
  num_nodes: 1
  component_placement:
    actor,env,rollout: 0
```

H800 preset 使用当前 Ray 节点上所有可见 GPU：

```yaml
cluster:
  num_nodes: 1
  component_placement:
    actor,env,rollout: all
```

在单张 RTX 5090 上，三个组件是不同 Ray 进程，但共享 GPU 0，后面的 serial
offload 生命周期保证它们不会同时占满显存。在 2、4 或 8 张 H800 上，`all`
会在每张卡各启动一套 actor、rollout 和 DreamDojo env 进程。这是数据并行和
完整模型复制，不是把一个 DreamDojo 模型切分到多张卡。

RTX 5090 专用 YAML 默认打开：

```yaml
runner.single_gpu_serial_offload: true
runner.single_gpu_serial_unload_env: true
runner.single_gpu_serial_lazy_actor_init: true
actor.fsdp_config.save_full_model_weights: false
```

## 3. 一个 global step 是什么

`global_step` 是一次完整 GRPO 更新，不是一个 action，也不是一次
DreamDojo 推理。

Teacher 的一个 global step 默认包含：

```text
2 条 trajectory（group_size=2）
  x 每条 32 个 policy action chunk
  x 每个 chunk 36 条 30 Hz action
  x 每次 DreamDojo 生成使用 35 步 denoise
完成后调用 1 次 actor.run_training()
```

Student 同样收集 2 条 trajectory 和 32 个 chunk，但每个 chunk 只有 12 条
30 Hz action，并用 4 步推理生成 4 帧。

因此每个 global step 都包含 32 次 batched `chunk_step()`。环境在每次
`chunk_step()` 内依次处理两个 env，所以 teacher 和 student 都有 64 次
per-env world-model generation：

```text
32 次 batched chunk_step
2 env x 32 chunk = 64 次 per-env generation
```

32 个有效 chunk 结束后，RolloutWorker 还会执行第 33 次通用 bootstrap policy
forward。该 forward 不再调用 DreamDojo，也不会生成第 33 个训练 sample。
Student 在这 64 次 RL generation 之外，还有 reset 时每个 env 一次的 warmup
generation；两条 trajectory 共生成 24 张 warmup 帧，但它们不进入 reward 或
GRPO loss。

### 3.1 VLA 生成 action

EnvWorker 将当前观察发送给 RolloutWorker。观察包括：

- 三路 Piper 相机图像；
- 当前关节状态；
- task description。

pi0.5 返回：

- action chunk；
- rollout 时的 `prev_logprobs`；
- actor 重新计算 logprob 所需的 `forward_inputs`；
- policy version 等辅助信息。

Teacher 每次输出 `[num_envs, 36, 14]` action，student 每次输出
`[num_envs, 12, 14]` action。

### 3.2 DreamDojo 推演未来

EnvWorker 最终调用：

```python
self.env_list[stage_id].chunk_step(chunk_actions)
```

Teacher 环境执行：

1. 将 36 条 30 Hz action 按 `action_stride=3` 下采样为 12 条 10 Hz action。
2. 对下采样后的 absolute joint action 做 min-max normalization。
3. 按 DreamDojo 数据处理方式转成 grouped delta action。
4. 将 Piper 的 14 维 action 写入 Cosmos 384 维 action 的 `[169:183]`。
5. 使用当前最后一帧作为 context，构造 1 context + 12 future frames。
6. 运行 35 步 rectified-flow inference。
7. 使用 Cosmos VAE/tokenizer 解码 12 张未来 RGB 帧。
8. 保留最后一帧作为下一个 chunk 的观察和条件帧。

日志中的：

```text
Generating samples: 17/35
```

表示当前 DreamDojo 视频生成已完成第 17 个 denoise step。它不是 VLA
训练进度。`Generating Rollout Epochs` 才是外层 trajectory 生成进度。

### 3.3 Reward 计算

顶层配置中的：

```yaml
reward:
  use_reward_model: false
```

只表示不创建独立 `EmbodiedRewardWorker`。DreamDojoEnv 仍会从
`env.train.reward_model` 加载内部 ResNet reward model。

Teacher 的 reward 映射为：

```text
12 张 10 Hz 生成帧
  -> 12 个 reward score
  -> 每个 score repeat 3 次
  -> 对齐 36 条 30 Hz policy action
```

Student 每次生成 4 帧，得到 4 个 reward，再映射到 12 条 policy action。

当前 `use_rel_reward: false`，所以 `_calc_step_reward()` 直接返回原始 ResNet
score。虽然配置中 `algorithm.reward_coef: 5.0` 会传入 env，但当前路径不会把
reward 乘以 5；只有 relative-reward 分支会使用这个系数。

当前 `success_reward_threshold` 用于 `success_once` 等日志指标，不会提前
终止 trajectory。环境在到达 `max_episode_steps` 时才 truncation，因此会
完整 rollout 32 个 chunk。

### 3.4 GRPO advantage

完整 trajectory 被发送给 ActorGroup。`reward_type: chunk_level` 会依次执行：

```text
逐帧 reward -> repeat 到每条 30 Hz action
每个 chunk 内沿 36/12 条 action 求和 -> chunk score
沿 32 个 chunk 累加 -> trajectory score
两个 trajectory score reshape 为一个 GRPO group
```

`group_size=2` 时，GRPO 用两条完整 trajectory 的 score 计算：

```text
advantage = (trajectory_score - group_mean) / (group_std + 1e-6)
```

较高 reward 的 trajectory 获得正 advantage，较低 reward 的 trajectory
获得负 advantage，并将同一个 trajectory advantage 广播到它的全部 32 个
chunk。如果两条 trajectory score 完全相同，advantage 为零（忽略浮点误差），
这个 group 不会产生有效策略梯度。

### 3.5 Actor 更新

Actor 使用 rollout 保存的 `forward_inputs` 重新计算当前策略 logprob：

```text
ratio = exp(new_logprob - old_logprob)
```

然后使用 PPO-style clipped GRPO actor loss。RTX 5090 配置会把
`2 env x 32 chunk = 64` 个 chunk sample 按 global batch 2 切成 32 个
minibatch；每个 minibatch 再拆成两个 micro-batch 1。因此一次
`actor.run_training()` 实际执行 32 次 `optimizer.step()`，LR scheduler 在全部
minibatch 完成后只前进一步。日志中的 actor 指标是这些 minibatch 的聚合值。

每个 minibatch 依次执行，全部 minibatch 完成后再更新一次 LR scheduler：

```text
forward -> policy loss -> backward -> gradient clipping
-> optimizer step

全部 minibatch 完成 -> LR scheduler step
```

WandB 中主要观察：

- `train/actor/policy_loss`
- `train/actor/total_loss`
- `train/actor/policy_loss_abs`
- `train/actor/ratio`
- `train/actor/approx_kl`
- `train/actor/clip_fraction`
- `train/actor/grad_norm`
- `env/reward`
- `env/return`
- `rollout/advantages_*`

终端 Metric Table 会省略最外层的 `train/` 前缀，例如终端里的
`actor/policy_loss` 对应 WandB 的 `train/actor/policy_loss`。

## 4. Teacher 与 student 的时序差异

### 4.1 Teacher

```text
VLA:       36 条 30 Hz action
stride:    每 3 条取 1 条
DreamDojo: 12 条 10 Hz action -> 12 张未来帧
context:   上一个 chunk 的最后一帧
denoise:   默认 35 步
trajectory: 32 chunk = 1152 条 policy action
```

### 4.2 Student

Student reset 时先进行一次不参与 RL loss 的 native warmup：

```text
1 张数据集初始帧
+ 数据集前 36 条 demonstration action
-> 12 条 10 Hz DreamDojo action
-> 生成 12 帧 warmup prefix
```

之后保留：

- 最近 9 张 RGB context frame；
- 最近 8 条 DreamDojo action。

每个 RL chunk：

```text
VLA:       12 条 30 Hz action
stride:    每 3 条取 1 条
DreamDojo: 4 条 10 Hz action
生成:      1 个 latent = 4 张 RGB 帧
denoise:   默认 4 步
trajectory: 32 chunk = 384 条 policy action
```

Student 在每次生成前用最近 9 帧和 8 条历史 action 重建三 latent KV cache，
而不是只取上一张图。生成完成后，旧历史和新 4 帧拼接，再裁剪回固定长度。
Student 默认还启用 `student_unload_on_offload: true`：每个 chunk 完成后会
直接销毁 student pipeline，下一个 chunk 再从 checkpoint 重建。它比 teacher
仅移动到 CPU 更慢，但能进一步降低单机主存常驻量。

当前单卡 worker 生命周期中，teacher 每个 global step 构建一次 pipeline，
reset 和 32 个 chunk 只做 GPU/CPU onload/offload。Student 会构建 34 次：环境
构造时 1 次、reset warmup 前重建 1 次、32 个 chunk 各重建 1 次。Student 每个
chunk 又依次处理两个 env，因此 2B DiT 在 VAE decode 前后的 CPU/GPU 往返共
`32 x 2 = 64` 次。

## 5. GPU/CPU 生命周期

当前串行路径的核心目标是避免以下三部分同时驻留 GPU 或主存：

- rollout VLA；
- DreamDojo 世界模型；
- actor VLA + gradient + Adam optimizer。

### 5.1 初始化阶段

Runner 先初始化 RolloutWorker 和 EnvWorker，但延迟构建 actor model：

```text
Rollout VLA 初始化 -> offload 到 CPU
Teacher 初始化 -> pipeline offload 到 CPU
Student 初始化 -> pipeline 立即删除
Actor Ray 进程存在，但尚未构建 FSDP model/optimizer
```

训练 env 配置为 `auto_reset: false`，因此 worker 初始化时不会执行 reset。
第一次 rollout bootstrap 才会 reset：teacher 从数据集读取初始图像；student
先从 DCP 重建 pipeline，再用 36 条 demonstration action 完成 warmup，随后再次
删除 pipeline。

### 5.2 VLA action 推理阶段

```text
GPU: rollout pi0.5 VLA
CPU: teacher pipeline/reward，或 student reward + rolling state
未构建: actor optimizer
```

RolloutWorker 对每个 chunk 执行：

```python
self.reload_model()
actions, result = self.predict(obs)
self.offload_model()
```

当前 2-step smoke 中，这一阶段采样到约 8-9 GiB GPU 使用量。

### 5.3 DreamDojo 生成阶段

Teacher 生成时，GPU 依次承载 DreamDojo DiT、video tokenizer/VAE 和 reward
model；rollout pi0.5 位于 CPU，actor optimizer 尚未构建。

Student 先用 GPU 上的 2B DiT 生成一个 latent。每个 env 在 VAE decode 前都会
清空 KV cache，并把 DiT 临时移到 CPU，避免 DiT 权重和全分辨率 decode 峰值
重叠；decode 后再把 DiT 移回 GPU，继续处理下一个 env。

DreamDojo `chunk_step()` 开始时调用 `onload()`，生成和 reward 完成后由
EnvWorker 调用 `offload()`。

2.151B teacher BF16 参数本身约 4.3 GB，但运行时还需要：

- 1440 x 640 x 13 帧视频张量；
- attention 激活和临时 buffer；
- rectified-flow latent；
- VAE decode buffer；
- action condition 和 reward 输入。

因此 2B 参数量不等于只使用 4 GB 显存。Teacher 2-step smoke 记录到的总
GPU 峰值是 `30178 MiB`，约 29.5 GiB，发生在 teacher 生成阶段；这个数值
不能直接当作 student 峰值。

### 5.4 Actor 训练阶段

完整 trajectory 生成结束后：

1. DreamDojo pipeline 被彻底删除，而不只是移动到 CPU。
2. Rollout VLA model 被删除。
3. Actor 才开始构建 FSDP VLA 和 optimizer。
4. 如果不是第一步，恢复上一 global step 的 trainable weights 和 optimizer。
5. 执行 GRPO forward/backward/update。

```text
GPU: actor pi0.5 + gradient + Adam state + activation
已删除: DreamDojo pipeline、rollout VLA
```

当前 batch size 2 的 smoke 中，actor 训练阶段采样峰值约 14.4 GiB。

### 5.5 保存和下一步

Actor 更新完成后：

1. 参数和 optimizer offload 到 CPU。
2. 按 tensor 分文件流式写入 checkpoint。
3. 删除 actor model、optimizer 和 trajectory batch。
4. 关闭旧 Ray channel。
5. 重启 actor、rollout、env worker。
6. 创建带 generation 后缀的新 channel。
7. 下一步将更新后的 action-expert 权重恢复到 rollout VLA。

流式保存避免 FSDP 构造完整 CPU state dict，后者曾在 checkpoint 保存时
造成显著主存峰值。

### 5.6 驻留汇总

| 阶段 | GPU 上 | CPU 上 |
| --- | --- | --- |
| 初始化完成 | 少量 CUDA context | Rollout VLA；teacher pipeline 在 CPU，student pipeline 已删除 |
| VLA action | Rollout VLA | Teacher pipeline/reward；student 只保留环境状态和 rolling context |
| Teacher denoise/decode | Teacher DiT、VAE、reward | Rollout VLA |
| Student generate/decode | DiT generation 与 VAE decode 分时驻留、reward | Rollout VLA；decode 时临时存放 DiT |
| GRPO actor 训练 | Actor VLA、gradient、optimizer | 无 DreamDojo pipeline |
| Stream checkpoint | 少量 CUDA context | Actor state，逐项写磁盘 |
| 7B text embedding | 不加载 | 不加载，直接读取 embedding cache |

Rollout VLA 和 actor VLA 使用同一套策略权重，但位于不同 Ray 进程，并承担
不同职责。当前串行模式不会让两个完整 VLA 同时驻留 GPU。

表中的 DreamDojo CPU 常驻描述适用于 teacher。Student 在 chunk 间默认直接
删除 pipeline，因此 VLA action 阶段通常只有 student 的环境状态和 rolling
context 留在 CPU，而没有完整 student DiT 权重。

## 6. Checkpoint 生命周期

串行训练为了在下一 global step 恢复 actor，内部每一步都需要保存状态：

- 未到 `runner.save_interval`：写入 `.serial_actor_state/global_step_N`；
- 到正式保存步：写入 `checkpoints/global_step_N`；
- 前一个临时状态在新状态成功后删除；
- 正式 checkpoint 不自动删除。

当前单个完整 serial checkpoint 约 6.8 GB。若设置：

```yaml
runner.save_interval: 10
```

100 个 global step 会永久保留 step 10、20、...、100，共约 68 GB。

## 7. 最近实现的改动

1. 启动脚本增加 `teacher` 和 `student` 入口。
2. 新增 `DreamDojoStudentEnv` 和独立 student YAML/GRPO YAML。
3. Teacher 与 student 默认都 rollout 32 个 policy chunk。
4. Student 实现 1 帧 -> 12 帧 warmup，以及 9 帧/8 action rolling context。
5. Reward 改为逐生成帧评分，再映射回 30 Hz policy action。
6. 禁用 online 7B Reason1 text encoder，使用 cached text embedding。
7. 增加 rollout VLA 与 DreamDojo 的逐 chunk GPU offload。
8. 增加完整 trajectory 后 DreamDojo pipeline unload。
9. 延迟 actor 初始化，避免 actor optimizer 与世界模型同时驻留主存。
10. 增加 trainable model 和 optimizer 的流式 checkpoint。
11. 增加 Ray WorkerGroup restart 和每步 channel generation，避免复用失效的
    Gloo process group。
12. RTX 5090 launcher 默认关闭 torch compile，并增加主存 guard 和
    `memory.csv`。

## 8. 运行和监控

### 8.1 运行前检查

从 RLinf 根目录运行。默认 launcher 会解析下列本地资源；路径不同时，用右侧
环境变量覆盖。

| 资源 | 默认路径 | 覆盖变量 |
| --- | --- | --- |
| Python | `.venv-dreamdojo/bin/python` | `LOCAL_PYTHON` |
| DreamDojo 仓库 | `../DreamDojo` | `DREAMDOJO_REPO_PATH` |
| OpenPI/kai0 仓库 | `../kai0` | `KAI0_PATH` |
| pi0.5 checkpoint | `checkpoints/base_policy_30000` | `BASE_POLICY_CKPT` |
| Teacher checkpoint | `checkpoints/dreamdojo_wm/model_ema_bf16.pt` | `DREAMDOJO_WM_CKPT` |
| Student DCP 根目录 | `checkpoints/dreamdojo_distill_3000` | `STUDENT_DREAMDOJO_WM_ROOT` |
| Reward model | `checkpoints/reward_model/full_weights.pt` | `REWARD_MODEL_CKPT` |
| Cached text embedding | `checkpoints/cosmos-predict2.5-2B/robot/action-cond/cr1_empty_string_text_embeddings.pt` | `DREAMDOJO_TEXT_EMBED_CACHE` |
| Reset NPY | `../data/piper_initial_frames` | `INITIAL_IMAGE_PATH` |

先确认关键资源存在：

```zsh
cd /home/fenrir/ubunto_data_2/worldmodel/RLinf

test -x .venv-dreamdojo/bin/python
test -d ../DreamDojo
test -d ../kai0
test -d checkpoints/base_policy_30000
test -f checkpoints/dreamdojo_wm/model_ema_bf16.pt
test -f checkpoints/dreamdojo_distill_3000/model/.metadata
test -f checkpoints/reward_model/full_weights.pt
test -f checkpoints/cosmos-predict2.5-2B/robot/action-cond/cr1_empty_string_text_embeddings.pt
```

Student reset 必须从每个 NPY 读取至少 36 条 30 Hz `abs_action`。本机当前
`../data/piper_initial_frames` 中的文件只有 5 帧，不能直接运行 student。保留
原目录并导出一份 36 帧版本：

```zsh
.venv-dreamdojo/bin/python \
  rlinf/envs/world_model/convert_piper_to_initial_npy.py \
  --dataset-path ../data/piper_insert_mouse_battery_lerobot \
  --out-dir ../data/piper_initial_frames_36 \
  --num-episodes 64 \
  --frames-per-file 36
```

后续 student 命令使用
`INITIAL_IMAGE_PATH="$PWD/../data/piper_initial_frames_36"`。缩短 RL trajectory
不会跳过这次 reset warmup。

### 8.2 RTX 5090 冒烟运行

长时间训练前，先分别完成一个 chunk 和一个 actor update。先停止旧 Ray，确保
`CUDA_VISIBLE_DEVICES` 和 Ray 内存配置由新进程继承：

```zsh
source .venv-dreamdojo/bin/activate
.venv-dreamdojo/bin/ray stop --force

CUDA_VISIBLE_DEVICES=0 \
RAY_memory_monitor_refresh_ms=0 \
USE_APPTAINER=0 \
LOCAL_PYTHON="$PWD/.venv-dreamdojo/bin/python" \
SMOKE=1 \
ACTION_CHUNKS_PER_TRAJECTORY=1 \
TRAIN_INFERENCE_STEPS=1 \
TRAIN_NUM_ENVS=2 \
ACTOR_MICRO_BATCH_SIZE=1 \
ACTOR_GLOBAL_BATCH_SIZE=2 \
HOST_MEMORY_GUARD_GB=12 \
bash examples/embodiment/run_dreamdojo_piper_teacher_grpo_rtx5090_1gpu.sh \
  runner.max_steps=1 \
  runner.val_check_interval=-1
```

Teacher smoke 成功退出后，重新执行
`.venv-dreamdojo/bin/ray stop --force`，再运行 student：

```zsh
CUDA_VISIBLE_DEVICES=0 \
RAY_memory_monitor_refresh_ms=0 \
USE_APPTAINER=0 \
LOCAL_PYTHON="$PWD/.venv-dreamdojo/bin/python" \
INITIAL_IMAGE_PATH="$PWD/../data/piper_initial_frames_36" \
SMOKE=1 \
ACTION_CHUNKS_PER_TRAJECTORY=1 \
TRAIN_INFERENCE_STEPS=1 \
TRAIN_NUM_ENVS=2 \
ACTOR_MICRO_BATCH_SIZE=1 \
ACTOR_GLOBAL_BATCH_SIZE=2 \
HOST_MEMORY_GUARD_GB=12 \
bash examples/embodiment/run_dreamdojo_piper_student_grpo_rtx5090_1gpu.sh \
  runner.max_steps=1 \
  runner.val_check_interval=-1
```

这两条 smoke 都收集一个包含两条 trajectory 的 GRPO group，执行 1 次
`actor.run_training()`，并流式保存更新后的 actor 状态。因为
`runner.max_steps=1`，进程会在保存后退出；更新权重只有在存在下一个 global
step 时才会恢复进 rollout。

### 8.3 RTX 5090 正式训练

Teacher 100 global steps、每条 trajectory 32 chunks、每次生成 35 个 inference
steps：

```zsh
.venv-dreamdojo/bin/ray stop --force

CUDA_VISIBLE_DEVICES=0 \
RAY_memory_monitor_refresh_ms=0 \
USE_APPTAINER=0 \
LOCAL_PYTHON="$PWD/.venv-dreamdojo/bin/python" \
ACTION_CHUNKS_PER_TRAJECTORY=32 \
TRAIN_INFERENCE_STEPS=35 \
TRAIN_NUM_ENVS=2 \
EVAL_NUM_ENVS=1 \
HOST_MEMORY_GUARD_GB=12 \
bash examples/embodiment/run_dreamdojo_piper_teacher_grpo_rtx5090_1gpu.sh \
  runner.max_steps=100 \
  runner.val_check_interval=-1 \
  runner.save_interval=10 \
  runner.per_worker_log=False
```

Student 使用 4 个 inference steps；默认 32 chunks 对应 384 个 30 Hz action：

```zsh
.venv-dreamdojo/bin/ray stop --force

CUDA_VISIBLE_DEVICES=0 \
RAY_memory_monitor_refresh_ms=0 \
USE_APPTAINER=0 \
LOCAL_PYTHON="$PWD/.venv-dreamdojo/bin/python" \
INITIAL_IMAGE_PATH="$PWD/../data/piper_initial_frames_36" \
ACTION_CHUNKS_PER_TRAJECTORY=32 \
TRAIN_INFERENCE_STEPS=4 \
TRAIN_NUM_ENVS=2 \
EVAL_NUM_ENVS=1 \
HOST_MEMORY_GUARD_GB=12 \
bash examples/embodiment/run_dreamdojo_piper_student_grpo_rtx5090_1gpu.sh \
  runner.max_steps=100 \
  runner.val_check_interval=-1 \
  runner.save_interval=10 \
  runner.per_worker_log=False
```

若要让 student 与 teacher 都覆盖 1152 个 30 Hz action，将 student 命令中的
`ACTION_CHUNKS_PER_TRAJECTORY=32` 改为 `96`。这会把每个 global step 的
student pipeline 重建次数和运行时间扩大约三倍。

RTX 5090 wrapper 设置 `VRAM_PRESET=none`，因此串行参数来自专用 YAML，而不是
公共 launcher 的旧 `32g` 分支。`RAY_memory_monitor_refresh_ms=0` 也不会由
wrapper 自动设置；上面的命令显式禁用 Ray 内存杀进程机制，同时保留
`HOST_MEMORY_GUARD_GB=12` 作为主存下限保护。

### 8.4 H800 单机多卡

单节点 8 张 H800 的 teacher 和 student 正式入口分别为：

```zsh
H800_NUM_GPUS=8 \
bash examples/embodiment/run_dreamdojo_piper_teacher_grpo_h800_multigpu.sh \
  runner.max_steps=100 \
  runner.val_check_interval=-1 \
  runner.save_interval=10

H800_NUM_GPUS=8 \
INITIAL_IMAGE_PATH="$PWD/../data/piper_initial_frames_36" \
bash examples/embodiment/run_dreamdojo_piper_student_grpo_h800_multigpu.sh \
  runner.max_steps=100 \
  runner.val_check_interval=-1 \
  runner.save_interval=10
```

H800 配置默认支持 2、4 或 8 张可见 GPU。16 个 train env 和 8 个 eval env
按 EnvWorker rank 均分；`micro_batch_size=8`、`global_batch_size=256`，因此
`16 env × 32 chunk = 512` 个样本正好执行两个 global minibatch。Student H800
配置会保留 DCP pipeline，并关闭每个 latent decode 前后的 2B DiT CPU/GPU
交换。8 卡节点建议至少准备 512 GB 主存；启动器默认保留 64 GB 可用主存。

完整 H800 训练前先跑一次单 chunk、单 actor update：

```zsh
H800_NUM_GPUS=8 \
INITIAL_IMAGE_PATH="$PWD/../data/piper_initial_frames_36" \
SMOKE=1 \
ACTION_CHUNKS_PER_TRAJECTORY=1 \
TRAIN_INFERENCE_STEPS=1 \
ACTOR_MICRO_BATCH_SIZE=1 \
ACTOR_GLOBAL_BATCH_SIZE=8 \
bash examples/embodiment/run_dreamdojo_piper_student_grpo_h800_multigpu.sh \
  runner.max_steps=1 \
  runner.val_check_interval=-1
```

### 8.5 WandB

RTX 5090 专用 YAML 默认设置 `runner.logger.logger_backends: []`。由于 wrapper
使用 `VRAM_PRESET=none`，不要依赖 `LOGGER_BACKENDS` 环境变量；公共 launcher
只在旧 `32g` 分支中读取它。先登录并设置项目：

```zsh
.venv-dreamdojo/bin/wandb login
export WANDB_PROJECT=rlinf
export WANDB_ENTITY="your-wandb-entity"  # 替换为实际 entity
```

然后将任一 RTX 5090 正式命令末尾的 `runner.per_worker_log=False` 替换为：

```zsh
  runner.per_worker_log=False \
  'runner.logger.logger_backends=[wandb]' \
  runner.logger.project_name="$WANDB_PROJECT" \
  runner.logger.wandb_entity="$WANDB_ENTITY"
```

建议 WandB 训练使用 `runner.per_worker_log=False`。当它为 `True` 时，
MetricLogger 会额外创建 `ActorGroup`、`RolloutGroup` 和 `EnvGroup` 的 scoped
run，loss 可能散落到多个 WandB run 中。关闭 per-worker metric 后，聚合后的
`train/*`、`env/*`、`rollout/*` 和 `time/*` 会记录在同一个主 run。

每个 global step 只在完整 trajectory 和 actor update 完成后写入一次训练
指标。因此 Step 1 结束前 WandB 没有 loss；Step 1 结束后也只有一个点，至少
完成两个 global step 才会形成可见的 loss 线段。

### 8.6 监控 GPU、主存和阶段

按进程观察显存：

```zsh
watch -n 1 'nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv'
```

查看总 GPU/主存监控：

先把 `<run-directory>` 替换为 launcher 启动时打印的实际日志目录。

```zsh
tail -f logs/<run-directory>/memory.csv
```

根据日志判断当前 GPU 阶段：

```zsh
tail -f logs/<run-directory>/run.log | \
  rg --line-buffered 'Generating samples|Initializing actor|run_training|Saving streamed|Restoring rollout'
```

关键日志含义：

| 日志 | 当前阶段 |
| --- | --- |
| `Generating samples` | DreamDojo 正在执行 denoise |
| `Generating Rollout Epochs` | 正在构造完整 trajectory |
| `Initializing actor` | DreamDojo/rollout 已释放，开始构建训练 actor |
| `actor/run_training` | VLA 正在 forward/backward/update |
| `Saving streamed serial actor state` | 正在流式保存 actor 和 optimizer |
| `Restoring rollout expert weights` | 下一 global step 正在恢复更新后的 VLA |

## 9. 当前验证范围

已经完成的 teacher 单卡串行核心路径端到端验证：

- 两个连续 global step；
- 第二步成功恢复第一步的 VLA trainable weights；
- 第二步成功恢复 optimizer 和 scheduler state；
- trajectory、GRPO loss、backward、保存和 worker/channel restart 全部完成；
- 进程退出码为 0。

当前 teacher/student preset、student env 和 lazy-actor runner 的单元测试为
22 个，运行命令如下：

```zsh
.venv-dreamdojo/bin/python -m pytest -q \
  tests/unit_tests/test_dreamdojo_grpo_presets.py \
  tests/unit_tests/test_dreamdojo_student_env.py \
  tests/unit_tests/test_embodied_runner_lazy_actor.py
```

当前结果为 `22 passed`。RTX 5090 student 仍需先准备 36-frame reset NPY；H800
入口完成了配置和单元测试检查，但本文不把它表述为已完成真实 H800 硬件 E2E。

### 9.1 完整 teacher Step 1 实测

以下数据来自 `20260710-144258-dreamdojo_piper_grpo`，配置为：

```text
global steps:                 100
group size / trajectories:   2
action chunks per trajectory: 32
actions per chunk:           36
episode length:              1152
DreamDojo denoise steps:     35
```

Step 1 完成后的时间统计：

| 指标 | 实测值 |
| --- | ---: |
| Global step 总时间 | 3039.171 s，约 50 分 39 秒 |
| 完整 rollout | 2968.72 s，约 49 分 28 秒 |
| DreamDojo env interaction | 2815.7 s |
| VLA rollout predict 累计 | 5.329 s |
| Actor GRPO training | 14.582 s |
| Advantage 计算 | 0.041 s |

Step 1 的环境和 rollout 指标：

| WandB key | 实测值 |
| --- | ---: |
| `env/episode_len` | 1152 |
| `env/num_trajectories` | 2 |
| `env/return` | 1117.92749 |
| `env/reward` | 0.970423 |
| `rollout/advantages_max` | 0.707113 |
| `rollout/advantages_mean` | 0.00000635 |
| `rollout/advantages_min` | -0.707100 |

Step 1 的 actor 指标：

| WandB key | 实测值 |
| --- | ---: |
| `train/actor/policy_loss` | 0.251525 |
| `train/actor/total_loss` | 0.125763 |
| `train/actor/policy_loss_abs` | 0.970626 |
| `train/actor/ratio` | 1.593341 |
| `train/actor/ratio_abs` | 0.956927 |
| `train/actor/approx_kl` | -0.040538 |
| `train/actor/clip_fraction` | 0.4375 |
| `train/actor/clipped_ratio` | 1.088674 |
| `train/actor/grad_norm` | 1568.01 |
| `train/actor/lr` | 0.000005 |
| `train/actor/entropy_loss` | 0 |

这些值是该次训练 Step 1 的观测结果，会随随机 action、reset episode、reward
和更新后的策略变化，不应当作为固定超参数或正确性阈值。

上述约 29.5 GiB DreamDojo 峰值、8-9 GiB rollout 峰值和 14.4 GiB actor
峰值来自缩短 chunk/denoise 数量的 2-step smoke，并由 2 秒间隔的
`nvidia-smi` 采样得到。完整 32 chunk x 35 denoise 运行会显著增加耗时，
但单次生成的主要张量形状不变；仍应以正式运行生成的 `memory.csv` 作为
最终峰值依据。
