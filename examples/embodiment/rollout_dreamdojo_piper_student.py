#!/usr/bin/env python
"""Closed-loop Pi0.5 + DreamDojo student rollout video smoke test.

This entrypoint intentionally bypasses Ray and GRPO. It loads one reset episode,
performs the student's native one-frame/36-action warmup, alternates policy action
generation with four-frame world-model generation, and writes a single MP4.

Two memory modes are supported:

* ``resident`` keeps Pi0.5, Cosmos-Reason1, the student DiT, and VAE on one GPU.
  This matches one data-parallel rank of the H800 GRPO preset.
* ``alternating`` encodes the instruction once, releases Reason1, and moves Pi0.5
  and DreamDojo between CPU and GPU for every chunk. It also offloads the student
  DiT during VAE decode, matching the constrained RTX 5090 execution timeline.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import time
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
from dreamdojo_venv_compat import ensure_scheduler_worker
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, open_dict

Worker = ensure_scheduler_worker()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Roll out Pi0.5 and the DreamDojo distilled student to MP4."
    )
    parser.add_argument(
        "--config-dir",
        default=str(Path(__file__).resolve().parent / "config"),
    )
    parser.add_argument("--config-name", default="dreamdojo_piper_grpo")
    parser.add_argument(
        "--memory-mode", choices=("resident", "alternating"), required=True
    )
    parser.add_argument("--vla-checkpoint", required=True)
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--dreamdojo-repo", required=True)
    parser.add_argument("--cosmos-tokenizer", required=True)
    parser.add_argument("--cosmos-reason1", required=True)
    parser.add_argument("--cr1-embeddings", required=True)
    parser.add_argument("--reset-data", required=True)
    parser.add_argument("--action-stats", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument(
        "--initial-image",
        default=None,
        help=(
            "Optional 1440x640 RGB three-camera stack. It replaces only the first "
            "image; the selected reset episode still supplies the required state "
            "and 36-action student warmup prefix."
        ),
    )
    parser.add_argument(
        "--instruction",
        default=None,
        help="Optional instruction override for the selected reset episode.",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=20,
        help=(
            "Number of closed-loop VLA/DreamDojo chunks. The default 20 covers "
            "one complete 240-action GRPO episode (12 policy actions per chunk)."
        ),
    )
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda-device", type=int, default=0)
    return parser.parse_args()


def _read_rgb_image(path: Path) -> np.ndarray:
    image = np.asarray(imageio.imread(path))
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        raise ValueError(f"Expected an RGB/RGBA image, got {image.shape} from {path}.")
    image = image[..., :3]
    if np.issubdtype(image.dtype, np.floating):
        scale = 255.0 if image.max(initial=0) <= 1.0 else 1.0
        image = image * scale
    return np.ascontiguousarray(np.clip(image, 0, 255).astype(np.uint8))


def _prepare_reset_source(
    reset_data: Path,
    episode_index: int,
    initial_image: Path | None,
    instruction: str | None,
    output_dir: Path,
) -> tuple[Path, int, Path]:
    npy_files = sorted(path for path in reset_data.glob("*.npy") if path.is_file())
    if not npy_files:
        raise FileNotFoundError(f"No reset .npy trajectories found in {reset_data}.")
    if not 0 <= episode_index < len(npy_files):
        raise IndexError(
            f"episode-index {episode_index} is outside [0, {len(npy_files) - 1}]."
        )
    source_path = npy_files[episode_index]
    if initial_image is None and instruction is None:
        return reset_data, episode_index, source_path

    trajectory = np.load(source_path, allow_pickle=True)
    if len(trajectory) < 36:
        raise ValueError(
            f"Student rollout needs 36 warmup actions, but {source_path} has "
            f"only {len(trajectory)} frames."
        )
    copied = np.empty(len(trajectory), dtype=object)
    for index, item in enumerate(trajectory):
        frame = dict(item)
        if instruction is not None:
            frame["instruction"] = instruction
            frame.pop("task", None)
        copied[index] = frame

    if initial_image is not None:
        image = _read_rgb_image(initial_image)
        if image.shape[:2] != (1440, 640):
            raise ValueError(
                "--initial-image must be a 1440x640 vertical stack of "
                "cam_high, cam_left_wrist, and cam_right_wrist; got "
                f"{image.shape[:2]} from {initial_image}."
            )
        copied[0]["image"] = image

    override_dir = output_dir / "reset_input"
    if override_dir.exists():
        shutil.rmtree(override_dir)
    override_dir.mkdir(parents=True)
    override_path = override_dir / "episode_00000.npy"
    np.save(override_path, copied, allow_pickle=True)
    return override_dir, 0, override_path


def _compose_configs(args: argparse.Namespace, reset_data: Path):
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base="1.1", config_dir=str(Path(args.config_dir).resolve())
    ):
        cfg = compose(config_name=args.config_name)

    env_cfg = cfg.env.train
    with open_dict(env_cfg):
        env_cfg.total_num_envs = 1
        env_cfg.group_size = 1
        env_cfg.use_fixed_reset_state_ids = False
        env_cfg.auto_reset = False
        env_cfg.ignore_terminations = True
        env_cfg.max_episode_steps = args.num_chunks * int(env_cfg.chunk)
        env_cfg.max_steps_per_rollout_epoch = env_cfg.max_episode_steps
        env_cfg.num_inference_steps = args.num_inference_steps
        env_cfg.enable_offload = args.memory_mode == "alternating"
        env_cfg.student_decode_dit_offload = args.memory_mode == "alternating"
        env_cfg.student_release_text_encoder_after_reset = (
            args.memory_mode == "alternating"
        )
        env_cfg.student_sequential_text_encoder = args.memory_mode == "alternating"
        env_cfg.student_capture_bootstrap_frames = True
        env_cfg.dreamdojo_repo_path = str(Path(args.dreamdojo_repo).resolve())
        env_cfg.dreamdojo_ckpt_path = args.student_checkpoint
        env_cfg.cosmos_tokenizer_path = str(Path(args.cosmos_tokenizer).resolve())
        env_cfg.cosmos_reason1_path = str(Path(args.cosmos_reason1).resolve())
        env_cfg.cr1_embeddings_path = str(Path(args.cr1_embeddings).resolve())
        env_cfg.initial_image_path = str(reset_data.resolve())
        env_cfg.action_norm_stats_path = str(Path(args.action_stats).resolve())
        env_cfg.video_cfg.save_video = False
        env_cfg.reward_model.type = None

    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.actor.model, resolve=True))
    with open_dict(model_cfg):
        model_cfg.model_path = str(Path(args.vla_checkpoint).resolve())
        model_cfg.load_to_device = False
        model_cfg.add_value_head = False
        model_cfg.openpi.add_value_head = False
    return env_cfg, model_cfg


def _recursive_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _recursive_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_recursive_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_recursive_cpu(item) for item in value)
    return value


def _clear_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _gpu_memory_mib() -> dict[str, float]:
    return {
        "allocated_mib": torch.cuda.memory_allocated() / 2**20,
        "reserved_mib": torch.cuda.memory_reserved() / 2**20,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
    }


def _native_initial_frame(env, episode_index: int) -> np.ndarray:
    trajectory = env.dataset._load_trajectory(episode_index)
    image = np.asarray(trajectory[0]["image"])
    image = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()
    if tuple(image.shape[1:]) != (env.gen_height, env.gen_width):
        image = torch.nn.functional.interpolate(
            image.unsqueeze(0),
            size=(env.gen_height, env.gen_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return image.clamp(0, 255).to(torch.uint8).permute(1, 2, 0).contiguous().numpy()


def _bootstrap_video_frames(env, initial_frame: np.ndarray) -> list[np.ndarray]:
    frames = [initial_frame]
    bootstrap = env.last_bootstrap_frames
    if bootstrap is None:
        return frames
    # [N, T, C, H, W] -> T x [H, W, C]
    frames.extend(
        np.ascontiguousarray(frame.transpose(1, 2, 0)) for frame in bootstrap[0].numpy()
    )
    return frames


def _write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        raise ValueError("Cannot write an empty rollout video.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps, macro_block_size=16) as writer:
        for frame in frames:
            writer.append_data(np.ascontiguousarray(frame.astype(np.uint8)))


def main() -> None:
    args = _parse_args()
    if args.num_chunks <= 0:
        raise ValueError("--num-chunks must be positive.")
    if not torch.cuda.is_available():
        raise RuntimeError("This DreamDojo rollout requires a CUDA GPU.")

    torch.cuda.set_device(args.cuda_device)
    device = torch.device("cuda", args.cuda_device)
    Worker.torch_device_type = "cuda"
    Worker.torch_platform = torch.cuda
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats(device)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reset_data, episode_index, reset_source = _prepare_reset_source(
        Path(args.reset_data).resolve(),
        args.episode_index,
        Path(args.initial_image).resolve() if args.initial_image else None,
        args.instruction,
        output_dir,
    )
    env_cfg, model_cfg = _compose_configs(args, reset_data)

    from rlinf.envs import get_env_cls
    from rlinf.envs.action_utils import prepare_actions_for_piper
    from rlinf.models import get_model

    env_cls = get_env_cls(env_cfg.env_type, env_cfg)
    env = env_cls(
        env_cfg,
        num_envs=1,
        seed_offset=args.seed,
        total_num_processes=1,
        worker_info=None,
        record_metrics=False,
    )

    reset_started = time.perf_counter()
    obs, _ = env.reset(episode_indices=torch.tensor([episode_index]))
    reset_seconds = time.perf_counter() - reset_started
    initial_frame = _native_initial_frame(env, episode_index)
    frames = _bootstrap_video_frames(env, initial_frame)
    phase_memory = [{"phase": "student_reset", **_gpu_memory_mib()}]

    if args.memory_mode == "alternating":
        policy_obs = _recursive_cpu(obs)
        env.offload()
        del obs
        _clear_cuda()
    else:
        policy_obs = obs
        del obs

    # Construct on CPU so alternating mode never overlaps policy construction
    # with the active student. Resident mode moves it once and keeps it there.
    policy = get_model(model_cfg).eval()
    if args.memory_mode == "resident":
        policy = policy.to(device)
        _clear_cuda()

    all_actions = []
    chunk_summaries = []
    for chunk_index in range(args.num_chunks):
        chunk_started = time.perf_counter()
        if args.memory_mode == "alternating":
            policy = policy.to(device)
            _clear_cuda()

        policy_started = time.perf_counter()
        with torch.no_grad():
            actions, result = policy.predict_action_batch(
                env_obs=policy_obs,
                mode="eval",
                compute_values=False,
            )
        policy_seconds = time.perf_counter() - policy_started
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        actions = prepare_actions_for_piper(actions)
        if actions.shape != (1, int(env_cfg.chunk), 14):
            raise RuntimeError(
                "Pi0.5 returned the wrong action shape: "
                f"{actions.shape}, expected (1, {env_cfg.chunk}, 14)."
            )
        actions_cpu = torch.as_tensor(actions, dtype=torch.float32).cpu()
        all_actions.append(actions_cpu[0].numpy())
        phase_memory.append(
            {"phase": f"chunk_{chunk_index:03d}_vla", **_gpu_memory_mib()}
        )
        del result, actions

        if args.memory_mode == "alternating":
            policy = policy.to("cpu")
            del policy_obs
            _clear_cuda()
            env.onload()

        student_started = time.perf_counter()
        obs_list, _, _, _, _ = env.chunk_step(actions_cpu)
        student_seconds = time.perf_counter() - student_started
        next_obs = obs_list[-1]
        chunk_frames = env.last_chunk_frames[0].detach().cpu().numpy()
        frames.extend(
            np.ascontiguousarray(frame.transpose(1, 2, 0)) for frame in chunk_frames
        )
        phase_memory.append(
            {"phase": f"chunk_{chunk_index:03d}_student", **_gpu_memory_mib()}
        )

        if args.memory_mode == "alternating":
            policy_obs = _recursive_cpu(next_obs)
            env.offload()
            del next_obs, obs_list
            _clear_cuda()
        else:
            policy_obs = next_obs

        chunk_summary = {
            "chunk": chunk_index,
            "policy_seconds": policy_seconds,
            "student_seconds": student_seconds,
            "total_seconds": time.perf_counter() - chunk_started,
            "action_min": float(actions_cpu.min()),
            "action_max": float(actions_cpu.max()),
            "action_mean": float(actions_cpu.mean()),
            "generated_frames": int(chunk_frames.shape[0]),
        }
        chunk_summaries.append(chunk_summary)
        print(json.dumps(chunk_summary, sort_keys=True), flush=True)

    video_path = output_dir / "vla_dreamdojo_rollout.mp4"
    actions_path = output_dir / "actions.npy"
    summary_path = output_dir / "summary.json"
    _write_video(video_path, frames, args.fps)
    np.save(actions_path, np.stack(all_actions, axis=0))

    summary = {
        "memory_mode": args.memory_mode,
        "device": torch.cuda.get_device_name(args.cuda_device),
        "episode_index": args.episode_index,
        "reset_source": str(reset_source),
        "initial_image_override": args.initial_image,
        "instruction_override": args.instruction,
        "reset_seconds": reset_seconds,
        "num_chunks": args.num_chunks,
        "policy_actions_per_chunk": int(env_cfg.chunk),
        "student_frames_per_chunk": int(env.student_actions_per_latent),
        "bootstrap_frames": int(env.student_bootstrap_model_actions),
        "video_frames": len(frames),
        "fps": args.fps,
        "video": str(video_path),
        "actions": str(actions_path),
        "phase_memory": phase_memory,
        "chunks": chunk_summaries,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
