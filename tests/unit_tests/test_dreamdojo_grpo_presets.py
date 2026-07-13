# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

CONFIG_DIR = Path(__file__).parents[2] / "examples" / "embodiment" / "config"
REPO_ROOT = Path(__file__).parents[2]
LAUNCHER = REPO_ROOT / "examples" / "embodiment" / "run_dreamdojo_piper_grpo.sh"


def _compose_preset(config_name: str, monkeypatch: pytest.MonkeyPatch) -> DictConfig:
    """Compose a DreamDojo preset without starting Ray or probing a GPU."""
    monkeypatch.setenv("EMBODIED_PATH", str(CONFIG_DIR.parent))
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        return compose(config_name=config_name)


def _dry_run_student_launcher(
    tmp_path: Path,
    student_root: Path,
    explicit_ckpt: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Resolve a student checkpoint without importing models or starting Ray."""
    env = os.environ.copy()
    if explicit_ckpt is None:
        env.pop("DREAMDOJO_WM_CKPT", None)
    else:
        env["DREAMDOJO_WM_CKPT"] = str(explicit_ckpt)
    env.update(
        {
            "STUDENT_DREAMDOJO_WM_ROOT": str(student_root),
            "USE_APPTAINER": "0",
            "LOCAL_PYTHON": "/bin/true",
            "HOST_MEMORY_GUARD_GB": "0",
            "LOG_DIR": str(tmp_path / "logs"),
            "ACTION_NORM_SOURCE": str(tmp_path / "missing-source.json"),
            "ACTION_NORM_STATS_PATH": str(tmp_path / "missing-stats.json"),
        }
    )
    return subprocess.run(
        ["bash", str(LAUNCHER), "student"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize("variant", ["teacher", "student"])
def test_rtx5090_presets_are_serial_single_gpu(
    variant: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check the explicit RTX 5090 presets retain the serial lifecycle."""
    cfg = _compose_preset(f"dreamdojo_piper_{variant}_grpo_rtx5090_1gpu", monkeypatch)

    assert str(cfg.cluster.component_placement["actor,env,rollout"]) == "0"
    assert cfg.env.train.total_num_envs == 2
    assert cfg.env.eval.total_num_envs == 1
    assert cfg.actor.micro_batch_size == 1
    assert cfg.actor.global_batch_size == 2
    assert cfg.runner.single_gpu_serial_offload is True
    assert cfg.runner.single_gpu_serial_unload_env is True
    assert cfg.runner.single_gpu_serial_lazy_actor_init is True
    assert cfg.algorithm.adv_type == "grpo"
    if variant == "student":
        assert cfg.env.train.dreamdojo_ckpt_path.endswith("dreamdojo_distill_3000")
        assert not cfg.env.train.get("student_experiment_opts", [])


def test_student_launcher_accepts_direct_dcp_root(tmp_path: Path) -> None:
    """Use a checkpoint root that directly contains model metadata."""
    student_root = tmp_path / "dreamdojo_distill_3000"
    (student_root / "model").mkdir(parents=True)
    (student_root / "model" / ".metadata").touch()

    result = _dry_run_student_launcher(tmp_path, student_root)

    assert result.returncode == 0, result.stderr
    assert f"Resolved DreamDojo WM: {student_root}" in result.stdout


def test_student_launcher_selects_latest_valid_iter_dcp(tmp_path: Path) -> None:
    """Select the newest valid iteration and ignore incomplete directories."""
    student_root = tmp_path / "student-checkpoints"
    for iteration in ("iter_2", "iter_10"):
        model_dir = student_root / iteration / "model"
        model_dir.mkdir(parents=True)
        (model_dir / ".metadata").touch()
    (student_root / "iter_99").mkdir(parents=True)

    result = _dry_run_student_launcher(tmp_path, student_root)

    assert result.returncode == 0, result.stderr
    assert f"Resolved DreamDojo WM: {student_root / 'iter_10'}" in result.stdout


def test_student_launcher_rejects_missing_dcp(tmp_path: Path) -> None:
    """Fail before training when no model metadata can be resolved."""
    student_root = tmp_path / "missing-student-checkpoint"

    result = _dry_run_student_launcher(tmp_path, student_root)

    assert result.returncode == 2
    assert "No valid student DCP checkpoint found" in result.stderr


def test_student_launcher_prefers_explicit_dcp(tmp_path: Path) -> None:
    """Use an explicit checkpoint even when the configured root is missing."""
    student_root = tmp_path / "missing-student-root"
    explicit_ckpt = tmp_path / "explicit-student-checkpoint"
    (explicit_ckpt / "model").mkdir(parents=True)
    (explicit_ckpt / "model" / ".metadata").touch()

    result = _dry_run_student_launcher(tmp_path, student_root, explicit_ckpt)

    assert result.returncode == 0, result.stderr
    assert f"Resolved DreamDojo WM: {explicit_ckpt}" in result.stdout


def test_student_launcher_rejects_invalid_explicit_dcp(tmp_path: Path) -> None:
    """Do not silently fall back when an explicit checkpoint is invalid."""
    student_root = tmp_path / "valid-student-root"
    (student_root / "model").mkdir(parents=True)
    (student_root / "model" / ".metadata").touch()
    explicit_ckpt = tmp_path / "invalid-explicit-checkpoint"

    result = _dry_run_student_launcher(tmp_path, student_root, explicit_ckpt)

    assert result.returncode == 2
    assert "Invalid student DCP checkpoint" in result.stderr


@pytest.mark.parametrize(
    ("variant", "env_type", "action_chunks"),
    [
        ("teacher", "dreamdojo_wm", 36),
        ("student", "dreamdojo_student_wm", 12),
    ],
)
def test_h800_presets_support_two_four_or_eight_gpus(
    variant: str,
    env_type: str,
    action_chunks: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check H800 data, actor batches, and GRPO groups for each topology."""
    cfg = _compose_preset(f"dreamdojo_piper_{variant}_grpo_h800_multigpu", monkeypatch)

    assert cfg.cluster.component_placement["actor,env,rollout"] == "all"
    assert cfg.env.train.env_type == env_type
    assert cfg.actor.model.num_action_chunks == action_chunks
    assert cfg.runner.single_gpu_serial_offload is False
    assert cfg.runner.single_gpu_serial_unload_env is False
    assert cfg.runner.single_gpu_serial_lazy_actor_init is False
    assert cfg.actor.fsdp_config.sharding_strategy == "no_shard"

    rollout_samples = (
        cfg.env.train.total_num_envs
        * cfg.env.train.max_steps_per_rollout_epoch
        // cfg.actor.model.num_action_chunks
    )
    assert rollout_samples == 512
    assert rollout_samples % cfg.actor.global_batch_size == 0

    for world_size in (2, 4, 8):
        assert cfg.env.train.total_num_envs % world_size == 0
        assert (
            cfg.env.train.total_num_envs // world_size % cfg.algorithm.group_size == 0
        )
        assert cfg.env.eval.total_num_envs % world_size == 0
        assert (
            cfg.actor.global_batch_size % (cfg.actor.micro_batch_size * world_size) == 0
        )

    if variant == "student":
        assert cfg.env.train.dreamdojo_ckpt_path.endswith("dreamdojo_distill_3000")
        assert not cfg.env.train.get("student_experiment_opts", [])
        assert cfg.env.train.student_unload_on_offload is False
        assert cfg.env.train.student_decode_dit_offload is False
