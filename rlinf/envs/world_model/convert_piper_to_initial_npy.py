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

"""Export Piper LeRobot episodes into per-episode reset trajectories.

DreamDojo environments reuse RLinf's :class:`NpyTrajectoryDatasetWrapper`,
which expects one object-array ``.npy`` file per episode. The distilled student
additionally needs the first 36 absolute 30 Hz actions for its native reset
warmup. The default direct loader therefore reads the LeRobot parquet and three
camera videos and writes ``image``, ``abs_action``, ``init_ee_pose``, and task
metadata. The older DreamDojo loader remains available for teacher-only reset
exports with ``--loader dreamdojo``.

Example::

    python rlinf/envs/world_model/convert_piper_to_initial_npy.py \
        --dataset-path /path/to/piper_insert_mouse_battery_lerobot \
        --out-dir /path/to/piper_initial_frames_36 \
        --num-episodes 64 \
        --frames-per-file 36

Set the output directory as ``initial_image_path`` in the selected Piper
DreamDojo environment config.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def _episode_id(path: Path) -> int:
    """Extract the numeric episode id from an episode file path."""
    return int(path.stem.removeprefix("episode_"))


def _write_episode(path: Path, frames: list[dict]) -> None:
    """Write one trajectory in the object-array format used by RLinf."""
    arr = np.empty(len(frames), dtype=object)
    for index, frame in enumerate(frames):
        arr[index] = frame
    np.save(path, arr, allow_pickle=True)


def _load_task_map(dataset_path: Path) -> dict[int, str]:
    """Load LeRobot task-index to instruction mappings when present."""
    tasks_path = dataset_path / "meta" / "tasks.jsonl"
    task_map = {}
    if not tasks_path.exists():
        return task_map
    with tasks_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            item = json.loads(line)
            task_map[int(item["task_index"])] = str(item["task"])
    return task_map


def _normalize_camera_keys(camera_keys: str) -> list[str]:
    """Normalize a comma-separated list to full LeRobot camera keys."""
    cameras = []
    for key in camera_keys.split(","):
        key = key.strip()
        if not key:
            continue
        if not key.startswith("observation.images."):
            key = f"observation.images.{key}"
        cameras.append(key)
    if not cameras:
        raise ValueError("--camera-keys must contain at least one camera")
    return cameras


def _find_video_path(dataset_path: Path, camera_key: str, episode_idx: int) -> Path:
    """Resolve one LeRobot episode video for a camera."""
    pattern = f"videos/chunk-*/{camera_key}/episode_{episode_idx:06d}.mp4"
    matches = sorted(dataset_path.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No video file found for camera={camera_key!r}, "
            f"episode={episode_idx:06d}, pattern={pattern!r}"
        )
    return matches[0]


def _read_video_frames(
    path: Path,
    num_frames: int,
    size: tuple[int, int],
) -> list[np.ndarray]:
    """Read and resize the requested RGB prefix from an episode video."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    frames = []
    try:
        for _ in range(num_frames):
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame.shape[:2] != size:
                frame = cv2.resize(
                    frame,
                    (size[1], size[0]),
                    interpolation=cv2.INTER_AREA,
                )
            frames.append(np.ascontiguousarray(frame.astype(np.uint8)))
    finally:
        capture.release()

    if not frames:
        raise RuntimeError(f"No frames read from video: {path}")
    return frames


def _export_direct(args: argparse.Namespace) -> int:
    """Export raw images, absolute actions, and states directly from LeRobot."""
    import pandas as pd

    dataset_path = Path(args.dataset_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cameras = _normalize_camera_keys(args.camera_keys)
    if args.height % len(cameras) != 0:
        raise ValueError(
            f"--height {args.height} must be divisible by number of cameras "
            f"{len(cameras)}"
        )
    per_view_size = (args.height // len(cameras), args.width)
    task_map = _load_task_map(dataset_path)

    parquet_files = sorted(
        dataset_path.glob("data/chunk-*/episode_*.parquet"), key=_episode_id
    )
    if not parquet_files:
        raise FileNotFoundError(f"No LeRobot parquet files found under {dataset_path}")

    num_episodes = min(args.num_episodes, len(parquet_files))
    print(f"Found {len(parquet_files)} episodes; exporting {num_episodes} episodes.")
    print(f"Using cameras: {', '.join(cameras)}")

    for parquet_path in parquet_files[:num_episodes]:
        episode_idx = _episode_id(parquet_path)
        dataframe = pd.read_parquet(parquet_path)
        num_frames = min(args.frames_per_file, len(dataframe))
        if num_frames <= 0:
            raise ValueError(f"Empty parquet episode: {parquet_path}")

        task_index = (
            int(dataframe["task_index"].iloc[0]) if "task_index" in dataframe else None
        )
        instruction = task_map.get(task_index, args.instruction)

        camera_frames = [
            _read_video_frames(
                _find_video_path(dataset_path, camera_key, episode_idx),
                num_frames,
                per_view_size,
            )
            for camera_key in cameras
        ]
        num_frames = min(num_frames, *(len(frames) for frames in camera_frames))
        if num_frames < args.frames_per_file:
            raise ValueError(
                f"Episode {episode_idx} has only {num_frames} aligned frames; "
                f"requested {args.frames_per_file}."
            )

        frames = []
        for frame_idx in range(num_frames):
            image = np.concatenate(
                [frames_for_camera[frame_idx] for frames_for_camera in camera_frames],
                axis=0,
            )
            state = (
                np.asarray(
                    dataframe["observation.state"].iloc[frame_idx], dtype=np.float32
                )
                if "observation.state" in dataframe
                else np.zeros(args.piper_action_dim, dtype=np.float32)
            )
            action = (
                np.asarray(dataframe["action"].iloc[frame_idx], dtype=np.float32)
                if "action" in dataframe
                else np.zeros(args.piper_action_dim, dtype=np.float32)
            )
            if action.size < args.piper_action_dim:
                raise ValueError(
                    f"Episode {episode_idx} frame {frame_idx} has action dim "
                    f"{action.size}; need {args.piper_action_dim}."
                )
            frames.append(
                {
                    "image": image,
                    "delta_action": np.zeros(args.piper_action_dim, dtype=np.float32),
                    "abs_action": action[: args.piper_action_dim],
                    "init_ee_pose": state[: args.piper_action_dim],
                    "instruction": instruction,
                }
            )

        out_path = out_dir / f"episode_{episode_idx:05d}.npy"
        _write_episode(out_path, frames)

    return num_episodes


def _export_dreamdojo(args: argparse.Namespace) -> int:
    """Export teacher-compatible reset frames through DreamDojo's loader."""
    from groot_dreams.dataloader import MultiVideoActionDataset

    if args.frames_per_file > args.num_frames:
        raise ValueError(
            "--loader dreamdojo cannot export more frames than --num-frames. "
            "Use the default direct loader for distilled-student reset data."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = MultiVideoActionDataset(
        dataset_path=args.dataset_path,
        num_frames=args.num_frames,
        data_split=args.data_split,
        restrict_len=args.num_episodes,
        height=args.height,
        width=args.width,
        video_key=args.video_key,
        fps=args.fps,
    )

    num_episodes = min(args.num_episodes, len(dataset))
    print(f"Dataset has {len(dataset)} samples; exporting {num_episodes} episodes.")

    for idx in range(num_episodes):
        data = dataset[idx]
        video = data["video"].detach().cpu().numpy()  # [C, T, H, W]
        num_frames = min(args.frames_per_file, video.shape[1])

        instruction = data.get("ai_caption") or args.instruction
        if isinstance(instruction, (list, tuple)):
            instruction = instruction[0] if instruction else args.instruction
        instruction = str(instruction) if instruction else args.instruction

        frames = []
        for frame_idx in range(num_frames):
            image = np.ascontiguousarray(video[:, frame_idx].transpose(1, 2, 0)).astype(
                np.uint8
            )
            frames.append(
                {
                    "image": image,
                    "delta_action": np.zeros(args.piper_action_dim, dtype=np.float32),
                    "instruction": instruction,
                }
            )

        _write_episode(out_dir / f"episode_{idx:05d}.npy", frames)

    return num_episodes


def main() -> None:
    """Parse CLI arguments and export the selected reset dataset."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-episodes", type=int, default=64)
    parser.add_argument(
        "--frames-per-file",
        type=int,
        default=36,
        help=(
            "Frames/actions retained per reset trajectory. The distilled student "
            "needs 36 raw 30Hz actions for its native 1-frame -> 12-frame warmup."
        ),
    )
    parser.add_argument("--loader", choices=["direct", "dreamdojo"], default="direct")
    parser.add_argument("--num-frames", type=int, default=13)
    parser.add_argument("--height", type=int, default=1440)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument(
        "--camera-keys",
        default=(
            "observation.images.cam_high,"
            "observation.images.cam_left_wrist,"
            "observation.images.cam_right_wrist"
        ),
        help="Comma-separated LeRobot camera keys for direct loading.",
    )
    parser.add_argument("--video-key", default="video.cam_vertical")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--data-split", default="train")
    parser.add_argument("--instruction", default="insert the battery")
    parser.add_argument("--piper-action-dim", type=int, default=14)
    args = parser.parse_args()

    if args.loader == "dreamdojo":
        num_episodes = _export_dreamdojo(args)
    else:
        num_episodes = _export_direct(args)
    print(f"Wrote {num_episodes} npy files to {Path(args.out_dir)}")


if __name__ == "__main__":
    main()
