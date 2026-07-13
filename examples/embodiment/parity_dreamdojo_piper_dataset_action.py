#!/usr/bin/env python
"""Parity check DreamDojo piper actions through the RLinf env wrapper.

This bypasses OpenPI entirely. It loads DreamDojo's dataset action sequence,
takes the piper slice [169:183], configures DreamDojoEnv as a 12-frame delta
action world model, and runs chunk_step through the RLinf input path.
"""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from dreamdojo_venv_compat import ensure_scheduler_worker

Worker = ensure_scheduler_worker()
from rlinf.envs.world_model.world_model_dreamdojo_env import DreamDojoEnv  # noqa: E402


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-dir",
        default="/project/peilab/srk/wmpo_workspace/RLinf/examples/embodiment/config",
    )
    parser.add_argument("--config-name", default="dreamdojo_piper_teacher_grpo")
    parser.add_argument(
        "--dataset-path",
        default="/project/peilab/srk/wmpo_workspace/piper_data/insert-mouse-battery/piper_insert_mouse_battery_lerobot",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--num-chunks", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=12)
    parser.add_argument("--num-inference-steps", type=int, default=35)
    parser.add_argument("--height", type=int, default=1440)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--video-key", default="video.cam_vertical")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _compose_env_cfg(args):
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    overrides = [
        "env.train.total_num_envs=1",
        "env.train.group_size=1",
        f"env.train.chunk={args.chunk_size}",
        "env.train.action_stride=1",
        "env.train.policy_action_format=delta",
        "env.train.action_norm_mode=none",
        f"env.train.num_inference_steps={args.num_inference_steps}",
        f"env.train.max_episode_steps={args.num_chunks * args.chunk_size}",
        f"env.train.max_steps_per_rollout_epoch={args.num_chunks * args.chunk_size}",
        "env.train.auto_reset=False",
        "env.train.ignore_terminations=True",
        "env.train.use_fixed_reset_state_ids=False",
        "env.train.enable_offload=False",
        "env.train.video_cfg.save_video=False",
        "env.train.reward_model.type=null",
    ]
    with initialize_config_dir(version_base="1.1", config_dir=args.config_dir):
        cfg = compose(config_name=args.config_name, overrides=overrides)
    return cfg.env.train


def _write_video(path: Path, frames: np.ndarray, fps: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps, macro_block_size=16) as writer:
        for frame in frames:
            writer.append_data(frame)


def main():
    args = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    Worker.torch_device_type = "cuda" if torch.cuda.is_available() else "cpu"
    Worker.torch_platform = torch.cuda if torch.cuda.is_available() else torch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Import after PYTHONPATH has been set by the launcher.
    from groot_dreams.dataloader import MultiVideoActionDataset

    num_frames = 1 + args.chunk_size * args.num_chunks
    dataset = MultiVideoActionDataset(
        dataset_path=args.dataset_path,
        num_frames=num_frames,
        data_split="train",
        restrict_len=args.index + 1,
        height=args.height,
        width=args.width,
        video_key=args.video_key,
        fps=args.fps,
    )
    data = dataset[args.index]
    full_action = data["action"][: num_frames - 1].float()
    piper_action = full_action[:, 169:183].contiguous()
    gt_video = data["video"].permute(1, 2, 3, 0).cpu().numpy()

    env_cfg = _compose_env_cfg(args)
    env = DreamDojoEnv(
        env_cfg,
        num_envs=1,
        seed_offset=args.seed,
        total_num_processes=1,
        worker_info=None,
        record_metrics=True,
    )
    env.reset(episode_indices=torch.tensor([args.index]))

    pred_frames = [env.current_obs[0].detach().cpu().numpy()]
    for chunk_idx in range(args.num_chunks):
        start = chunk_idx * args.chunk_size
        end = start + args.chunk_size
        chunk_action = piper_action[start:end].unsqueeze(0)
        env.chunk_step(chunk_action)
        frames = env.last_chunk_frames[0].detach().cpu().numpy()
        frames = np.transpose(frames, (0, 2, 3, 1))
        pred_frames.extend(list(frames))
        print(
            json.dumps(
                {
                    "chunk": chunk_idx,
                    "action_shape": list(chunk_action.shape),
                    "action_min": float(chunk_action.min()),
                    "action_max": float(chunk_action.max()),
                    "action_mean": float(chunk_action.mean()),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    pred_video = np.stack(pred_frames, axis=0).astype(np.uint8)
    gt_video = gt_video[: pred_video.shape[0]].astype(np.uint8)
    merged_video = np.concatenate([gt_video, pred_video], axis=2)

    _write_video(out_dir / "pred.mp4", pred_video, args.fps)
    _write_video(out_dir / "gt.mp4", gt_video, args.fps)
    _write_video(out_dir / "merged_gt_left_pred_right.mp4", merged_video, args.fps)
    np.savez_compressed(
        out_dir / "frames.npz", pred=pred_video, gt=gt_video, merged=merged_video
    )

    summary = {
        "index": args.index,
        "num_chunks": args.num_chunks,
        "chunk_size": args.chunk_size,
        "num_frames": int(pred_video.shape[0]),
        "action_source": "DreamDojo data['action'][:,169:183]",
        "env_overrides": {
            "chunk": args.chunk_size,
            "action_stride": 1,
            "policy_action_format": "delta",
            "action_norm_mode": "none",
            "num_inference_steps": args.num_inference_steps,
        },
        "pred": str(out_dir / "pred.mp4"),
        "gt": str(out_dir / "gt.mp4"),
        "merged": str(out_dir / "merged_gt_left_pred_right.mp4"),
        "frames_npz": str(out_dir / "frames.npz"),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
