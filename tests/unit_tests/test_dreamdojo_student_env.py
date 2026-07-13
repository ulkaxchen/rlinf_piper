from types import MethodType, SimpleNamespace
import weakref

import numpy as np
import pytest
import torch

from rlinf.envs import get_env_cls
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


def test_student_env_is_registered():
    assert get_env_cls("dreamdojo_student_wm") is DreamDojoStudentEnv


def test_base_env_unload_rebuilds_and_rehydrates_cpu_state():
    env = object.__new__(DreamDojoEnv)
    cleanup_calls = []
    rebuilt_pipes = []

    class _Pipe:
        def cleanup(self):
            cleanup_calls.append(True)

    env.pipe = _Pipe()
    original_pipe_ref = weakref.ref(env.pipe)
    env._is_offloaded = False
    env.reward_model = None
    env.current_obs = torch.ones(1)
    env.current_states = torch.ones(1)
    env.last_chunk_frames = torch.ones(1)
    env.prev_step_reward = torch.zeros(1)
    env.reset_state_ids = torch.zeros(1, dtype=torch.long)
    env.record_metrics = False
    env.device = torch.device("cpu")
    env._clear_accelerator_cache = lambda: None

    def _build_pipeline():
        pipe = _Pipe()
        rebuilt_pipes.append(pipe)
        return pipe

    env._build_pipeline = _build_pipeline

    env.unload()

    assert env.pipe is None
    assert original_pipe_ref() is None
    assert env._is_offloaded
    assert cleanup_calls == [True]

    env.onload()

    assert env.pipe is rebuilt_pipes[0]
    assert not env._is_offloaded
    assert env.current_obs.device.type == "cpu"


def test_student_env_temporal_contract():
    env = _make_env_without_models()
    env.pipe = SimpleNamespace(
        model=SimpleNamespace(
            net=SimpleNamespace(_num_action_per_latent_frame=4),
            config=SimpleNamespace(cache_frame_size=3),
        )
    )

    env._validate_student_temporal_contract()


def test_student_chunk_advances_four_frames_and_preserves_history():
    env = _make_env_without_models()
    captured_actions = []

    def _fake_generate(self, env_idx, current_actions, seed):
        del self, env_idx, seed
        captured_actions.append(current_actions.clone())
        frame_values = torch.tensor([-1.0, -0.5, 0.0, 1.0]).view(1, 1, 4, 1, 1)
        return frame_values.expand(1, 3, 4, 2, 2).clone()

    env._generate_one_latent = MethodType(_fake_generate, env)
    actions = torch.arange(12, dtype=torch.float32).view(1, 12, 1).expand(-1, -1, 14)

    env._infer_next_chunk_frames(actions)

    assert env.last_chunk_frames.shape == (1, 4, 3, 2, 2)
    assert env.current_obs.shape == (1, 2, 2, 3)
    assert torch.all(env.current_obs == 255)
    assert env._student_condition_frames.shape == (1, 3, 9, 2, 2)
    assert env._student_action_history.shape == (1, 8, 384)
    assert captured_actions[0].shape == (4, 384)
    assert torch.equal(
        captured_actions[0][
            :,
            169:183,
        ],
        torch.tensor([0.0, 3.0, 6.0, 9.0]).view(4, 1).expand(-1, 14),
    )
    assert torch.equal(env._student_action_history[0, -4:], captured_actions[0])


def test_student_bootstrap_uses_demo_prefix_and_populates_native_cache():
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


def test_student_bootstrap_passes_hwc_initial_image_to_streaming_api():
    env = _make_env_without_models()
    captured = {}

    class _Pipe:
        def generate_action_streaming(self, **kwargs):
            captured.update(kwargs)
            return torch.zeros(1, 3, 13, 2, 2)

    env.pipe = _Pipe()
    video = env._generate_student_bootstrap_video(
        torch.zeros(3, 2, 2, dtype=torch.uint8),
        torch.zeros(12, 384),
        seed=7,
    )

    assert captured["video_path"].shape == (1, 2, 2, 3)
    assert captured["actions_np"].shape == (12, 384)
    assert captured["num_steps"] == 4
    assert video.shape == (1, 3, 13, 2, 2)


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


def test_student_frame_rewards_expand_to_policy_actions():
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
