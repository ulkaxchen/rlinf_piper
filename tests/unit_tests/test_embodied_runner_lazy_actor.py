from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.utils.serial_checkpoint import (
    load_trainable_model_state,
    save_trainable_model_state,
    serial_model_state_exists,
)


class _FakeHandle:
    def wait(self):
        return None


class _FakeGroup:
    def __init__(self):
        self.init_calls = 0
        self.restart_calls = 0

    def init_worker(self):
        self.init_calls += 1
        return _FakeHandle()

    def restart(self):
        self.restart_calls += 1


class _FakeChannel:
    def __init__(self, name: str):
        self.name = name
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


def _make_lazy_runner(max_steps: int) -> EmbodiedRunner:
    runner = object.__new__(EmbodiedRunner)
    runner.cfg = OmegaConf.create(
        {
            "runner": {"resume_dir": None},
            "actor": {"fsdp_config": {"save_full_model_weights": False}},
        }
    )
    runner.single_gpu_serial_offload = True
    runner.single_gpu_serial_unload_env = True
    runner.single_gpu_serial_lazy_actor_init = True
    runner.max_steps = max_steps
    runner._profile_all_steps = False
    runner._profile_steps = None
    runner.rollout = _FakeGroup()
    runner.env = _FakeGroup()
    runner.actor = _FakeGroup()
    runner.reward = None
    runner.logger = SimpleNamespace(info=lambda _: None)
    runner._actor_initialized = False
    return runner


def test_lazy_actor_init_defers_actor_model_setup():
    runner = _make_lazy_runner(max_steps=1)

    runner.init_workers()

    assert runner.rollout.init_calls == 1
    assert runner.env.init_calls == 1
    assert runner.actor.init_calls == 0
    assert not runner._actor_initialized


def test_lazy_actor_init_supports_multiple_grpo_updates():
    runner = _make_lazy_runner(max_steps=2)

    runner.init_workers()

    assert runner.rollout.init_calls == 1
    assert runner.env.init_calls == 1
    assert runner.actor.init_calls == 0


def test_lazy_actor_init_rejects_full_weight_checkpoint_copy():
    runner = _make_lazy_runner(max_steps=2)
    runner.cfg.actor.fsdp_config.save_full_model_weights = True

    with pytest.raises(ValueError, match="save_full_model_weights=False"):
        runner.init_workers()


def test_serial_runtime_restart_replaces_workers_and_channels(monkeypatch):
    runner = _make_lazy_runner(max_steps=2)
    runner.reward = _FakeGroup()
    old_channels = [
        _FakeChannel("Env"),
        _FakeChannel("Rollout"),
        _FakeChannel("Actor"),
        _FakeChannel("Reward"),
    ]
    (
        runner.env_channel,
        runner.rollout_channel,
        runner.actor_channel,
        runner.reward_channel,
    ) = old_channels
    runner._serial_channel_generation = 0
    created_names = []

    def create_channel(name):
        created_names.append(name)
        return _FakeChannel(name)

    monkeypatch.setattr("rlinf.runners.embodied_runner.Channel.create", create_channel)

    runner._restart_serial_runtime()

    assert all(channel.close_calls == 1 for channel in old_channels)
    assert runner.actor.restart_calls == 1
    assert runner.rollout.restart_calls == 1
    assert runner.env.restart_calls == 1
    assert runner.reward.restart_calls == 1
    assert runner.actor.init_calls == 0
    assert runner.rollout.init_calls == 1
    assert runner.env.init_calls == 1
    assert runner.reward.init_calls == 1
    assert created_names == [
        "EnvSerial1",
        "RolloutSerial1",
        "ActorSerial1",
        "RewardSerial1",
    ]


def test_streamed_model_checkpoint_only_restores_trainable_parameters(tmp_path):
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    model[0].requires_grad_(False)
    frozen_before = model[0].weight.detach().clone()
    trainable_before = model[1].weight.detach().clone()

    save_trainable_model_state(model, str(tmp_path))
    assert serial_model_state_exists(str(tmp_path))

    with torch.no_grad():
        model[0].weight.add_(1)
        model[1].weight.add_(1)
    load_trainable_model_state(model, str(tmp_path))

    assert not torch.equal(model[0].weight, frozen_before)
    assert torch.equal(model[1].weight, trainable_before)
