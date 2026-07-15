import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import pytest
import torch


def _load_rollout_module():
    repo_root = Path(__file__).parents[2]
    examples_dir = repo_root / "examples" / "embodiment"
    sys.path.insert(0, str(examples_dir))
    try:
        module_path = examples_dir / "rollout_dreamdojo_piper_student.py"
        spec = importlib.util.spec_from_file_location(
            "dreamdojo_rollout_test_module", module_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(examples_dir))


ROLLOUT = _load_rollout_module()


def _write_reset_episode(path: Path) -> np.ndarray:
    trajectory = np.empty(36, dtype=object)
    for index in range(36):
        trajectory[index] = {
            "image": np.zeros((1440, 640, 3), dtype=np.uint8) if index == 0 else None,
            "instruction": "original instruction",
            "observation.state": np.arange(14, dtype=np.float32),
            "abs_action": np.full(14, index, dtype=np.float32),
        }
    np.save(path, trajectory, allow_pickle=True)
    return trajectory


def test_rollout_reset_override_preserves_student_warmup_prefix(tmp_path):
    reset_dir = tmp_path / "reset"
    reset_dir.mkdir()
    original = _write_reset_episode(reset_dir / "episode_00000.npy")
    override_image = np.full((1440, 640, 3), 127, dtype=np.uint8)
    image_path = tmp_path / "initial.png"
    imageio.imwrite(image_path, override_image)

    selected_dir, episode_index, selected_path = ROLLOUT._prepare_reset_source(
        reset_dir,
        0,
        image_path,
        "insert the mouse battery",
        tmp_path / "output",
    )

    copied = np.load(selected_path, allow_pickle=True)
    assert selected_dir == tmp_path / "output" / "reset_input"
    assert episode_index == 0
    assert np.array_equal(copied[0]["image"], override_image)
    assert all(item["instruction"] == "insert the mouse battery" for item in copied)
    assert np.array_equal(copied[-1]["abs_action"], original[-1]["abs_action"])
    assert np.array_equal(
        copied[0]["observation.state"], original[0]["observation.state"]
    )


def test_rollout_rejects_single_camera_initial_image(tmp_path):
    reset_dir = tmp_path / "reset"
    reset_dir.mkdir()
    _write_reset_episode(reset_dir / "episode_00000.npy")
    image_path = tmp_path / "single_camera.png"
    imageio.imwrite(image_path, np.zeros((480, 640, 3), dtype=np.uint8))

    with pytest.raises(ValueError, match="vertical stack"):
        ROLLOUT._prepare_reset_source(
            reset_dir,
            0,
            image_path,
            None,
            tmp_path / "output",
        )


def test_rollout_video_starts_with_input_then_bootstrap_frames():
    initial = np.zeros((2, 2, 3), dtype=np.uint8)
    bootstrap = torch.arange(2 * 3 * 2 * 2, dtype=torch.uint8).reshape(1, 2, 3, 2, 2)
    env = SimpleNamespace(last_bootstrap_frames=bootstrap)

    frames = ROLLOUT._bootstrap_video_frames(env, initial)

    assert len(frames) == 3
    assert np.array_equal(frames[0], initial)
    assert np.array_equal(frames[1], bootstrap[0, 0].permute(1, 2, 0).numpy())


@pytest.mark.parametrize(
    ("memory_mode", "expected_offload"),
    [("alternating", True), ("resident", False)],
)
def test_rollout_memory_mode_controls_all_student_offload_flags(
    tmp_path, monkeypatch, memory_mode, expected_offload
):
    repo_root = Path(__file__).parents[2]
    config_dir = repo_root / "examples" / "embodiment" / "config"
    monkeypatch.setenv("EMBODIED_PATH", str(config_dir.parent))
    args = SimpleNamespace(
        config_dir=str(config_dir),
        config_name="dreamdojo_piper_grpo",
        memory_mode=memory_mode,
        num_chunks=1,
        num_inference_steps=4,
        dreamdojo_repo=str(tmp_path / "DreamDojo"),
        student_checkpoint=str(tmp_path / "student"),
        cosmos_tokenizer=str(tmp_path / "tokenizer.pth"),
        cosmos_reason1=str(tmp_path / "Cosmos-Reason1-7B"),
        cr1_embeddings=str(tmp_path / "cr1.pt"),
        action_stats=str(tmp_path / "action_stats.json"),
        vla_checkpoint=str(tmp_path / "vla"),
    )

    env_cfg, model_cfg = ROLLOUT._compose_configs(args, tmp_path / "reset")

    assert env_cfg.enable_offload is expected_offload
    assert env_cfg.student_decode_dit_offload is expected_offload
    assert env_cfg.student_release_text_encoder_after_reset is expected_offload
    assert env_cfg.student_sequential_text_encoder is expected_offload
    assert env_cfg.student_capture_bootstrap_frames is True
    assert env_cfg.reward_model.type is None
    assert model_cfg.load_to_device is False
