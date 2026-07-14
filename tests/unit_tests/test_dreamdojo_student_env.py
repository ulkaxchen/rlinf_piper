from pathlib import Path
from types import MethodType, SimpleNamespace

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


def _make_env_without_models() -> DreamDojoStudentEnv:
    env = object.__new__(DreamDojoStudentEnv)
    env.num_envs = 1
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
    env.current_obs = torch.zeros(1, 2, 2, 3, dtype=torch.uint8)
    env.current_states = torch.zeros(1, 14)
    env._last_action_state = None
    env._last_reset_episode_indices = np.array([0], dtype=np.int64)
    env._student_condition_frames = torch.zeros(1, 3, 9, 2, 2, dtype=torch.uint8)
    env._student_action_history = torch.zeros(1, 8, 384)
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

    def _fake_bootstrap(self, initial_frame, model_actions, seed):
        del self, initial_frame, seed
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
    checkpoint = tmp_path / "dreamdojo_distill_3000"
    (checkpoint / "model").mkdir(parents=True)
    (checkpoint / "model" / ".metadata").touch()
    embedding = tmp_path / "cr1.pt"
    embedding.touch()

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
            "cr1_embeddings_path": str(embedding),
        }
    )

    env._validate_distilled_student_inputs()

    env.cfg.dreamdojo_ckpt_path = str(checkpoint / "model")
    with pytest.raises(ValueError, match="DCP checkpoint root"):
        env._validate_distilled_student_inputs()


def test_piper_reset_export_contains_36_absolute_actions(tmp_path, monkeypatch):
    import pandas as pd

    dataset_path = tmp_path / "lerobot"
    parquet = dataset_path / "data" / "chunk-000" / "episode_000000.parquet"
    parquet.parent.mkdir(parents=True)
    parquet.touch()
    out_dir = tmp_path / "reset_36"

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
    assert trajectory[0]["abs_action"].shape == (14,)
    assert np.array_equal(
        trajectory[-1]["abs_action"], np.full(14, 35, dtype=np.float32)
    )


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
    assert cfg.env.train.initial_image_path.endswith("piper_initial_frames_36")
    assert cfg.env.train.cr1_embeddings_path.endswith(
        "cosmos-predict2.5-2B/robot/action-cond/cr1_empty_string_text_embeddings.pt"
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
