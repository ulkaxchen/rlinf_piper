import sys
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from rlinf.envs import get_env_cls
from rlinf.envs.world_model import convert_piper_to_initial_npy
from rlinf.envs.world_model.world_model_dreamdojo_env import DreamDojoEnv
from rlinf.envs.world_model.world_model_dreamdojo_student_env import (
    DreamDojoStudentEnv,
)


def _make_env_without_models(num_envs: int = 1) -> DreamDojoStudentEnv:
    env = object.__new__(DreamDojoStudentEnv)
    env.num_envs = num_envs
    env.chunk = 12
    env.action_stride = 3
    env.gen_frames = 4
    env.piper_action_dim = 14
    env.model_action_dim = 384
    env.action_slot_start = 169
    env.action_slot_end = 183
    env.policy_action_format = "delta"
    env.student_actions_per_latent = 4
    env.student_cache_latents = 3
    env.student_condition_fps = 4.0
    env.student_decode_dit_offload = False
    env.student_release_text_encoder_after_reset = False
    env.student_sequential_text_encoder = False
    env.student_capture_bootstrap_frames = False
    env.student_context_pixel_frames = 9
    env.student_history_actions = 8
    env.student_bootstrap_enabled = True
    env.student_bootstrap_action_key = "abs_action"
    env.student_bootstrap_policy_actions = 36
    env.student_bootstrap_model_actions = 12
    env.student_bootstrap_complete = False
    env.seed_base = 0
    env.elapsed_steps = 0
    env.device = torch.device("cpu")
    env.gen_height = 2
    env.gen_width = 2
    env.num_inference_steps = 4
    env.current_obs = torch.zeros(num_envs, 2, 2, 3, dtype=torch.uint8)
    env.current_states = torch.zeros(num_envs, 14)
    env._last_action_state = None
    env._last_reset_episode_indices = np.arange(num_envs, dtype=np.int64)
    env._student_condition_frames = torch.zeros(num_envs, 3, 9, 2, 2, dtype=torch.uint8)
    env._student_action_history = torch.zeros(num_envs, 8, 384)
    env.task_descriptions = [f"task {idx}" for idx in range(num_envs)]
    env._student_text_embeddings_gpu = None
    env._student_text_mask_gpu = None
    env._student_text_prompts = ()
    env.last_bootstrap_frames = None
    env._clear_accelerator_cache = lambda: None
    return env


def test_student_backend_selects_student_env_without_changing_teacher():
    student_cfg = {"dreamdojo_backend": "distilled_student"}
    teacher_cfg = {"dreamdojo_backend": "autoreg"}

    assert get_env_cls("dreamdojo_wm", student_cfg) is DreamDojoStudentEnv
    assert get_env_cls("dreamdojo_wm", teacher_cfg) is DreamDojoEnv


def test_student_temporal_contract_matches_checkpoint():
    env = _make_env_without_models()
    env.pipe = SimpleNamespace(
        model=SimpleNamespace(
            net=SimpleNamespace(_num_action_per_latent_frame=4),
            config=SimpleNamespace(cache_frame_size=3),
        )
    )

    env._validate_student_temporal_contract()

    env.gen_frames = 12
    with pytest.raises(ValueError, match="generate one latent"):
        env._validate_student_temporal_contract()


def test_student_encodes_episode_instructions_with_native_reason1():
    env = _make_env_without_models(num_envs=2)
    env.task_descriptions = ["insert mouse battery", "close the drawer"]
    calls = []

    class _TextEncoder:
        def compute_text_embeddings_online(self, data_batch, input_caption_key):
            calls.append((data_batch, input_caption_key))
            return torch.stack(
                [
                    torch.full((3, 4), 1.0),
                    torch.full((3, 4), 2.0),
                ]
            )

    env.pipe = SimpleNamespace(
        model=SimpleNamespace(
            text_encoder=_TextEncoder(),
            input_caption_key="ai_caption",
            tensor_kwargs={"device": torch.device("cpu"), "dtype": torch.bfloat16},
        ),
        t5_text_embeddings_cpu=torch.zeros(1, 3, 4),
    )

    env._encode_student_episode_instructions()

    assert len(calls) == 1
    assert calls[0][1] == "ai_caption"
    assert calls[0][0] == {
        "ai_caption": ["insert mouse battery", "close the drawer"],
        "images": None,
    }
    assert env._student_text_embeddings_gpu.shape == (2, 3, 4)
    assert env._student_text_embeddings_gpu.dtype == torch.bfloat16
    assert env._student_text_mask_gpu.shape == (2, 3)
    assert torch.all(env._student_text_mask_gpu == 1)

    first_batch = {}
    second_batch = {}
    env._inject_student_text_condition(first_batch, 0)
    env._inject_student_text_condition(second_batch, 1)
    assert first_batch["t5_text_embeddings"].shape == (1, 3, 4)
    assert second_batch["t5_text_embeddings"].shape == (1, 3, 4)
    assert torch.all(first_batch["t5_text_embeddings"] == 1)
    assert torch.all(second_batch["t5_text_embeddings"] == 2)
    assert first_batch["t5_text_mask"].shape == (1, 3)


def test_student_rejects_wrong_online_reason1_batch_size():
    env = _make_env_without_models(num_envs=2)
    env.pipe = SimpleNamespace(
        model=SimpleNamespace(
            text_encoder=SimpleNamespace(
                compute_text_embeddings_online=lambda **_: torch.zeros(1, 3, 4)
            ),
            input_caption_key="ai_caption",
            tensor_kwargs={"device": torch.device("cpu"), "dtype": torch.bfloat16},
        ),
        t5_text_embeddings_cpu=torch.zeros(1, 3, 4),
    )

    with pytest.raises(RuntimeError, match="wrong embedding batch size"):
        env._encode_student_episode_instructions()


def test_student_sequential_reason1_moves_pipeline_around_temporary_encoder(
    monkeypatch,
):
    env = _make_env_without_models()
    env.student_sequential_text_encoder = True
    events = []

    class _TemporaryTextEncoder:
        def __init__(self, config, device):
            events.append(("encoder_init", config, device))

        def compute_text_embeddings_online(self, data_batch, input_caption_key):
            events.append(("encode", data_batch, input_caption_key))
            return torch.ones(1, 3, 4)

    text_encoder_module = ModuleType("text_encoder")
    text_encoder_module.TextEncoder = _TemporaryTextEncoder
    monkeypatch.setitem(
        sys.modules,
        "cosmos_predict2._src.predict2.text_encoders.text_encoder",
        text_encoder_module,
    )

    text_encoder_config = SimpleNamespace(compute_online=False)
    env.pipe = SimpleNamespace(
        model=SimpleNamespace(
            text_encoder=None,
            config=SimpleNamespace(text_encoder_config=text_encoder_config),
            input_caption_key="ai_caption",
            tensor_kwargs={"device": torch.device("cpu"), "dtype": torch.bfloat16},
        ),
        t5_text_embeddings_cpu=torch.zeros(1, 3, 4),
    )
    env._move_pipe = lambda device: events.append(("move_student", str(device)))

    env._encode_student_episode_instructions()

    assert events == [
        ("move_student", "cpu"),
        ("encoder_init", text_encoder_config, "cpu"),
        (
            "encode",
            {"ai_caption": ["task 0"], "images": None},
            "ai_caption",
        ),
        ("move_student", "cpu"),
    ]
    assert env.pipe.model.text_encoder is None
    assert env._student_text_embeddings_gpu.shape == (1, 3, 4)
    assert env._student_text_embeddings_gpu.dtype == torch.bfloat16


def test_student_pipe_move_includes_plain_reason1_wrapper():
    env = _make_env_without_models()
    events = []

    class _Movable:
        def __init__(self, name):
            self.name = name

        def to(self, device):
            events.append((self.name, str(device)))
            return self

    text_encoder = SimpleNamespace(model=_Movable("reason1"), device="cuda")
    model = _Movable("student")
    model.text_encoder = text_encoder
    env.pipe = SimpleNamespace(model=model)

    env._move_pipe("cpu")

    assert events == [("student", "cpu"), ("reason1", "cpu")]
    assert env.pipe.model.text_encoder.device == "cpu"


def test_student_bootstrap_uses_matching_online_reason1_embedding():
    env = _make_env_without_models(num_envs=2)
    env._student_text_prompts = tuple(env.task_descriptions)
    env._student_text_embeddings_gpu = torch.stack(
        [torch.full((3, 4), 1.0), torch.full((3, 4), 2.0)]
    )
    env._student_text_mask_gpu = torch.ones(2, 3)
    compatibility_embedding = torch.zeros(1, 3, 4)
    seen_embeddings = []

    def _fake_generate_action_streaming(**_):
        seen_embeddings.append(env.pipe.t5_text_embeddings_cpu.clone())
        return torch.zeros(1, 3, 13, 2, 2)

    env.pipe = SimpleNamespace(
        t5_text_embeddings_cpu=compatibility_embedding,
        generate_action_streaming=_fake_generate_action_streaming,
    )
    initial = torch.zeros(3, 2, 2, dtype=torch.uint8)
    actions = torch.zeros(12, 384)

    env._generate_student_bootstrap_video(0, initial, actions, seed=0)
    assert env.pipe.t5_text_embeddings_cpu is compatibility_embedding
    env._generate_student_bootstrap_video(1, initial, actions, seed=1)
    assert env.pipe.t5_text_embeddings_cpu is compatibility_embedding

    assert len(seen_embeddings) == 2
    assert seen_embeddings[0].shape == (1, 3, 4)
    assert torch.all(seen_embeddings[0] == 1)
    assert torch.all(seen_embeddings[1] == 2)

    def _fail_generate_action_streaming(**_):
        raise RuntimeError("warmup failed")

    env.pipe.generate_action_streaming = _fail_generate_action_streaming
    with pytest.raises(RuntimeError, match="warmup failed"):
        env._generate_student_bootstrap_video(0, initial, actions, seed=2)
    assert env.pipe.t5_text_embeddings_cpu is compatibility_embedding


def test_student_steady_condition_uses_matching_online_reason1_embedding(
    monkeypatch,
):
    action_module = ModuleType("action_conditioner")
    action_module.ActionConditionedCondition = lambda **kwargs: kwargs
    conditioner_module = ModuleType("conditioner")
    conditioner_module.DataType = SimpleNamespace(VIDEO="video")
    monkeypatch.setitem(
        sys.modules,
        ("cosmos_predict2._src.predict2.action.configs.action_conditioned.conditioner"),
        action_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "cosmos_predict2._src.predict2.conditioner",
        conditioner_module,
    )

    env = _make_env_without_models(num_envs=2)
    env._student_text_prompts = tuple(env.task_descriptions)
    env._student_text_embeddings_gpu = torch.stack(
        [torch.full((3, 4), 1.0), torch.full((3, 4), 2.0)]
    )
    env._student_text_mask_gpu = torch.ones(2, 3)
    captured_batches = []

    class _Condition:
        def edit_data_type(self, _):
            return self

        def set_video_condition(self, **_):
            return self

        def to_dict(self):
            return {}

    def _get_data_and_condition(data_batch):
        captured_batches.append(data_batch.copy())
        return None, torch.zeros(1, 1, 4, 1, 1), _Condition(), None

    model = SimpleNamespace(
        tensor_kwargs={"device": torch.device("cpu"), "dtype": torch.bfloat16},
        _normalize_video_databatch_inplace=lambda _: None,
        _augment_image_dim_inplace=lambda _: None,
        get_data_and_condition=_get_data_and_condition,
    )
    env.pipe = SimpleNamespace(
        model=model,
        _prepare_data_batch=lambda **_: {"video": torch.zeros(1, dtype=torch.float32)},
    )
    current_actions = torch.zeros(4, 384)

    env._build_student_condition(0, current_actions)
    env._build_student_condition(1, current_actions)

    assert captured_batches[0]["t5_text_embeddings"].shape == (1, 3, 4)
    assert torch.all(captured_batches[0]["t5_text_embeddings"] == 1)
    assert torch.all(captured_batches[1]["t5_text_embeddings"] == 2)
    assert captured_batches[0]["t5_text_mask"].shape == (1, 3)


def test_student_reset_encodes_new_instructions_before_warmup(monkeypatch):
    env = _make_env_without_models(num_envs=2)
    instruction_batches = iter(
        [
            ["task A", "task B"],
            ["task C", "task D"],
        ]
    )
    events = []

    def _fake_base_reset(self, *args, **kwargs):
        del args, kwargs
        self.task_descriptions = next(instruction_batches)
        events.append(("base", tuple(self.task_descriptions)))
        return {"obs": "base"}, {}

    def _fake_encode(self):
        self._student_text_prompts = tuple(self.task_descriptions)
        self._student_text_embeddings_gpu = torch.zeros(2, 3, 4)
        self._student_text_mask_gpu = torch.ones(2, 3)
        events.append(("encode", self._student_text_prompts))

    def _fake_warmup(self):
        events.append(("warmup", self._student_text_prompts))

    monkeypatch.setattr(DreamDojoEnv, "reset", _fake_base_reset)
    env._encode_student_episode_instructions = MethodType(_fake_encode, env)
    env._bootstrap_student_context = MethodType(_fake_warmup, env)
    env._wrap_obs = lambda: {"obs": "student"}

    env.reset()
    env.reset()

    assert events == [
        ("base", ("task A", "task B")),
        ("encode", ("task A", "task B")),
        ("warmup", ("task A", "task B")),
        ("base", ("task C", "task D")),
        ("encode", ("task C", "task D")),
        ("warmup", ("task C", "task D")),
    ]


def test_student_chunk_generates_four_frames_and_preserves_causal_history():
    env = _make_env_without_models()
    captured_actions = []
    captured_contexts = []

    def _fake_generate(self, env_idx, current_actions, seed):
        del env_idx, seed
        captured_actions.append(current_actions.clone())
        captured_contexts.append(self._student_condition_frames.clone())
        call_idx = len(captured_actions)
        values = torch.arange(
            call_idx * 4,
            call_idx * 4 + 4,
            dtype=torch.float32,
        ).view(1, 1, 4, 1, 1)
        # Map to [-1, 1] while retaining four distinct frame values.
        values = values / 8.0 - 1.0
        return values.expand(1, 3, 4, 2, 2).clone()

    env._generate_one_latent = MethodType(_fake_generate, env)
    first_actions = torch.arange(12, dtype=torch.float32).view(1, 12, 1)
    first_actions = first_actions.expand(-1, -1, 14)
    second_actions = first_actions + 100

    env._infer_next_chunk_frames(first_actions)
    first_generated_tail = env._student_condition_frames[:, :, -4:].clone()
    env._infer_next_chunk_frames(second_actions)

    assert env.last_chunk_frames.shape == (1, 4, 3, 2, 2)
    assert env.current_obs.shape == (1, 2, 2, 3)
    assert env._student_condition_frames.shape == (1, 3, 9, 2, 2)
    assert env._student_action_history.shape == (1, 8, 384)
    assert torch.equal(captured_contexts[1][:, :, -4:], first_generated_tail)
    assert torch.equal(
        captured_actions[0][:, 169:183],
        torch.tensor([0.0, 3.0, 6.0, 9.0]).view(4, 1).expand(-1, 14),
    )
    assert torch.equal(
        captured_actions[1][:, 169:183],
        torch.tensor([100.0, 103.0, 106.0, 109.0]).view(4, 1).expand(-1, 14),
    )
    assert torch.equal(
        env._student_action_history[0],
        torch.cat(captured_actions, dim=0),
    )


def test_student_bootstrap_uses_36_demo_actions_without_exposing_reward():
    env = _make_env_without_models()
    env.student_capture_bootstrap_frames = True
    trajectory = np.array(
        [
            {"abs_action": np.full(14, action_idx, dtype=np.float32)}
            for action_idx in range(36)
        ],
        dtype=object,
    )
    env.dataset = SimpleNamespace(
        _load_trajectory=lambda _: trajectory,
        npy_files=["episode_00000.npy"],
    )
    captured_actions = []

    def _fake_bootstrap(self, env_idx, initial_frame, model_actions, seed):
        del self, initial_frame, seed
        assert env_idx == 0
        captured_actions.append(model_actions.clone())
        frame_values = torch.linspace(-1.0, 1.0, 13).view(1, 1, 13, 1, 1)
        return frame_values.expand(1, 3, 13, 2, 2).clone()

    env._generate_student_bootstrap_video = MethodType(_fake_bootstrap, env)

    env._bootstrap_student_context()

    assert captured_actions[0].shape == (12, 384)
    assert torch.equal(
        captured_actions[0][:, 169:183],
        torch.arange(0, 36, 3, dtype=torch.float32).view(12, 1).expand(-1, 14),
    )
    assert env._student_condition_frames.shape == (1, 3, 9, 2, 2)
    assert env._student_action_history.shape == (1, 8, 384)
    assert torch.equal(
        env._student_action_history[0, :, 169:183],
        torch.arange(12, 36, 3, dtype=torch.float32).view(8, 1).expand(-1, 14),
    )
    assert torch.equal(env.current_states, torch.full((1, 14), 35.0))
    assert env.last_bootstrap_frames.shape == (1, 12, 3, 2, 2)
    assert env.last_bootstrap_frames.device.type == "cpu"
    assert env.last_chunk_frames is None


def test_student_bootstrap_requires_36_demo_actions():
    env = _make_env_without_models()
    env.dataset = SimpleNamespace(
        _load_trajectory=lambda _: np.array(
            [{"abs_action": np.zeros(14, dtype=np.float32)} for _ in range(5)],
            dtype=object,
        ),
        npy_files=["episode_00000.npy"],
    )

    with pytest.raises(ValueError, match="frames-per-file 36"):
        env._load_student_bootstrap_policy_actions()


def test_student_frame_rewards_expand_to_twelve_policy_actions():
    env = _make_env_without_models()
    env.last_chunk_frames = torch.zeros(1, 4, 3, 2, 2, dtype=torch.uint8)

    class _RewardModel:
        def __init__(self):
            self.frame_idx = 0

        def compute_reward(self, obs):
            assert obs["main_images"].shape == (1, 3, 2, 2)
            reward = torch.tensor([float(self.frame_idx)])
            self.frame_idx += 1
            return reward

    env.reward_model = _RewardModel()

    rewards = env._infer_next_chunk_rewards()

    assert rewards.shape == (1, 12)
    assert torch.equal(
        rewards,
        torch.tensor([[0.0] * 3 + [1.0] * 3 + [2.0] * 3 + [3.0] * 3]),
    )


def test_student_checkpoint_validation_accepts_dcp_root(tmp_path):
    import torch.distributed.checkpoint as dcp
    from safetensors.torch import save_file

    checkpoint = tmp_path / "dreamdojo_distill_3000"
    dcp.save(
        {"net_ema": {"weight": torch.ones(1)}},
        checkpoint_id=checkpoint / "model",
    )
    tokenizer = tmp_path / "tokenizer.pth"
    tokenizer.touch()
    embedding = tmp_path / "cr1.pt"
    embedding.touch()
    reason1 = tmp_path / "Cosmos-Reason1-7B"
    reason1.mkdir()
    for name in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.json",
        "preprocessor_config.json",
    ):
        (reason1 / name).write_text("{}", encoding="utf-8")
    (reason1 / "config.json").write_text(
        (
            '{"model_type":"qwen2_5_vl","hidden_size":3584,'
            '"num_hidden_layers":28,"vocab_size":152064}'
        ),
        encoding="utf-8",
    )
    reason1_shard = reason1 / "model-00001-of-00001.safetensors"
    save_file({"model.weight": torch.ones(1)}, reason1_shard)
    (reason1 / "model.safetensors.index.json").write_text(
        (
            '{"metadata":{"total_size":4},'
            '"weight_map":{"model.weight":"model-00001-of-00001.safetensors"}}'
        ),
        encoding="utf-8",
    )

    env = object.__new__(DreamDojoStudentEnv)
    env.cfg = OmegaConf.create(
        {
            "config_file": (
                "cosmos_predict2/_src/predict2/interactive/configs/config_distill.py"
            ),
            "experiment": (
                "cosmos_predict2p5_2B_action_gr00t_pretrain_self_forcing_no_s3"
            ),
            "dreamdojo_ckpt_path": str(checkpoint),
            "cosmos_tokenizer_path": str(tokenizer),
            "cosmos_reason1_path": str(reason1),
            "cr1_embeddings_path": str(embedding),
        }
    )

    env._validate_distilled_student_inputs()

    reason1_shard.unlink()
    with pytest.raises(FileNotFoundError, match="weight shards"):
        env._validate_distilled_student_inputs()
    reason1_shard.write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="truncated"):
        env._validate_distilled_student_inputs()
    save_file({"model.weight": torch.ones(1)}, reason1_shard)

    env.cfg.dreamdojo_ckpt_path = str(checkpoint / "model")
    with pytest.raises(ValueError, match="DCP checkpoint root"):
        env._validate_distilled_student_inputs()

    env.cfg.dreamdojo_ckpt_path = str(checkpoint)
    next((checkpoint / "model").glob("*.distcp")).unlink()
    with pytest.raises(FileNotFoundError, match="DCP is incomplete"):
        env._validate_distilled_student_inputs()


def test_student_checkpoint_opts_add_explicit_full_tokenizer(tmp_path):
    env = object.__new__(DreamDojoStudentEnv)
    tokenizer = tmp_path / "tokenizer.pth"
    reason1 = tmp_path / "Cosmos-Reason1-7B"
    env.cfg = OmegaConf.create(
        {
            "cosmos_tokenizer_path": str(tokenizer),
            "cosmos_reason1_path": str(reason1),
            "student_experiment_opts": ["model.config.fsdp_shard_size=1"],
        }
    )
    assert env._student_checkpoint_experiment_opts() == [
        "model.config.net_fake_score=null",
        f"+model.config.tokenizer.vae_pth={tokenizer}",
        "model.config.fsdp_shard_size=1",
        "model.config.text_encoder_config.compute_online=true",
        f"model.config.text_encoder_config.ckpt_path={reason1}",
        (
            "model.config.text_encoder_config.model_config.tokenizer."
            f"cache_dir={reason1}"
        ),
    ]


def test_piper_reset_export_contains_36_absolute_actions(tmp_path, monkeypatch):
    import json

    import pandas as pd

    dataset_path = tmp_path / "lerobot"
    parquet = dataset_path / "data" / "chunk-000" / "episode_000000.parquet"
    parquet.parent.mkdir(parents=True)
    parquet.touch()
    out_dir = tmp_path / "reset_36"
    episode_stats_path = dataset_path / "meta" / "episodes_stats.jsonl"
    episode_stats_path.parent.mkdir(parents=True)
    episode_stats_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "episode_index": 0,
                        "stats": {
                            "action": {
                                "min": [0.0] * 14,
                                "max": [35.0] * 14,
                            }
                        },
                    }
                ),
                json.dumps(
                    {
                        "episode_index": 1,
                        "stats": {
                            "action": {
                                "min": [-2.0] * 14,
                                "max": [50.0] * 14,
                            }
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    dataframe = pd.DataFrame(
        {
            "task_index": [0] * 36,
            "observation.state": [np.arange(14, dtype=np.float32)] * 36,
            "action": [
                np.full(14, frame_idx, dtype=np.float32) for frame_idx in range(36)
            ],
        }
    )
    monkeypatch.setattr(pd, "read_parquet", lambda _: dataframe)
    monkeypatch.setattr(
        convert_piper_to_initial_npy,
        "_find_video_path",
        lambda *_: tmp_path / "unused.mp4",
    )
    monkeypatch.setattr(
        convert_piper_to_initial_npy,
        "_read_video_frames",
        lambda path, num_frames, size: [
            np.zeros((*size, 3), dtype=np.uint8) for _ in range(num_frames)
        ],
    )
    args = SimpleNamespace(
        dataset_path=str(dataset_path),
        out_dir=str(out_dir),
        camera_keys="cam_high,cam_left_wrist,cam_right_wrist",
        height=6,
        width=2,
        num_episodes=1,
        frames_per_file=36,
        instruction="insert the battery",
        piper_action_dim=14,
    )

    assert convert_piper_to_initial_npy._export_direct(args) == 1
    trajectory = np.load(out_dir / "episode_00000.npy", allow_pickle=True)

    assert len(trajectory) == 36
    assert trajectory[0]["image"].shape == (6, 2, 3)
    assert "image" not in trajectory[-1]
    assert trajectory[0]["abs_action"].shape == (14,)
    assert all(frame["instruction"] == "insert the battery" for frame in trajectory)
    assert np.array_equal(
        trajectory[-1]["abs_action"], np.full(14, 35, dtype=np.float32)
    )
    action_stats = json.loads((out_dir / "action_stats.json").read_text())
    assert action_stats["action"]["min"] == [-2.0] * 14
    assert action_stats["action"]["max"] == [50.0] * 14


def test_default_grpo_config_is_8gpu_resident_student(monkeypatch):
    config_dir = Path(__file__).parents[2] / "examples" / "embodiment" / "config"
    monkeypatch.setenv("EMBODIED_PATH", str(config_dir.parent))
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(config_name="dreamdojo_piper_grpo")

    assert cfg.env.train.dreamdojo_backend == "distilled_student"
    assert cfg.env.eval.dreamdojo_backend == "distilled_student"
    assert cfg.env.train.chunk == cfg.actor.model.num_action_chunks == 12
    assert cfg.env.train.num_inference_steps == 4
    assert cfg.env.train.max_steps_per_rollout_epoch == 240
    assert cfg.env.train.max_steps_per_rollout_epoch // cfg.env.train.chunk == 20
    assert cfg.env.train.enable_offload is False
    assert cfg.env.eval.enable_offload is False
    assert cfg.rollout.enable_offload is False
    assert cfg.actor.enable_offload is False
    assert cfg.env.train.student_decode_dit_offload is False
    assert cfg.env.train.student_release_text_encoder_after_reset is False
    assert cfg.env.train.student_sequential_text_encoder is False
    assert cfg.env.train.student_capture_bootstrap_frames is False
    assert cfg.env.train.initial_image_path.endswith("piper_initial_frames_36")
    assert cfg.env.train.cr1_embeddings_path.endswith(
        "cosmos-predict2.5-2B/robot/action-cond/cr1_empty_string_text_embeddings.pt"
    )
    assert cfg.env.train.cosmos_reason1_path.endswith("Cosmos-Reason1-7B")
    assert cfg.env.eval.cosmos_reason1_path == cfg.env.train.cosmos_reason1_path
    assert cfg.env.train.cosmos_tokenizer_path.endswith(
        "cosmos-predict2.5-2B/tokenizer.pth"
    )

    rollout_samples = (
        cfg.env.train.total_num_envs
        * cfg.env.train.max_steps_per_rollout_epoch
        // cfg.actor.model.num_action_chunks
    )
    assert rollout_samples == 320
    assert rollout_samples % cfg.actor.global_batch_size == 0
    assert cfg.actor.global_batch_size % (cfg.actor.micro_batch_size * 8) == 0

    teacher_env = OmegaConf.load(config_dir / "env" / "dreamdojo_piper.yaml")
    student_env = OmegaConf.load(config_dir / "env" / "dreamdojo_piper_student.yaml")
    assert OmegaConf.to_container(student_env.reward_model) == OmegaConf.to_container(
        teacher_env.reward_model
    )
