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

"""Export piper LeRobot episodes into per-episode .npy initial-frame files.

DreamDojoEnv reuses RLinf's :class:`NpyTrajectoryDatasetWrapper` for resets,
which expects a directory of ``*.npy`` files, one per episode, each holding an
object array of per-frame dicts with keys ``image`` (HWC uint8), ``delta_action``
and ``instruction``. By default this script reads LeRobot mp4/parquet files
directly and writes those npy files. The older DreamDojo
``MultiVideoActionDataset`` path is still available with ``--loader dreamdojo``.

Run inside the DreamDojo venv (so ``groot_dreams`` imports), e.g.::

    cd /mnt/afs-h200/yuyangcheng/data/Shirk6_DreamDojo_piper
    source .venv/bin/activate
    PYTHONPATH=. python /path/to/RLinf/rlinf/envs/world_model/convert_piper_to_initial_npy.py \
        --dataset-path datasets/piper_insert_mouse_battery_lerobot \
        --out-dir /mnt/afs-h200/yuyangcheng/data/piper_initial_frames \
        --num-episodes 64

The ``--out-dir`` value is what you set as ``env.train.initial_image_path`` in
``examples/embodiment/config/env/dreamdojo_piper_teacher.yaml``.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def _episode_id(path: Path) -> int:
    return int(path.stem.removeprefix("episode_"))


def _write_episode(path: Path, frames: list[dict]) -> None:
    arr = np.empty(len(frames), dtype=object)
    for i, frame in enumerate(frames):
        arr[i] = frame
    np.save(path, arr, allow_pickle=True)


def _load_task_map(dataset_path: Path) -> dict[int, str]:
    tasks_path = dataset_path / "meta" / "tasks.jsonl"
    task_map = {}
    if not tasks_path.exists():
        return task_map
    with tasks_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            task_map[int(item["task_index"])] = str(item["task"])
    return task_map


def _normalize_camera_keys(camera_keys: str) -> list[str]:
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
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    frames = []
    try:
        for _ in range(num_frames):
            ok, frame = cap.read()
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
        cap.release()

    if not frames:
        raise RuntimeError(f"No frames read from video: {path}")
    return frames


def _export_direct(args: argparse.Namespace) -> int:
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

    n = min(args.num_episodes, len(parquet_files))
    print(f"Found {len(parquet_files)} episodes; exporting {n} episodes.")
    print(f"Using cameras: {', '.join(cameras)}")

    for parquet_path in parquet_files[:n]:
        episode_idx = _episode_id(parquet_path)
        df = pd.read_parquet(parquet_path)
        k = min(args.frames_per_file, len(df))
        if k <= 0:
            raise ValueError(f"Empty parquet episode: {parquet_path}")

        task_index = int(df["task_index"].iloc[0]) if "task_index" in df else None
        instruction = task_map.get(task_index, args.instruction)

        camera_frames = [
            _read_video_frames(
                _find_video_path(dataset_path, camera_key, episode_idx),
                k,
                per_view_size,
            )
            for camera_key in cameras
        ]
        k = min(k, *(len(frames) for frames in camera_frames))

        frames = []
        for frame_idx in range(k):
            img = np.concatenate(
                [frames_for_camera[frame_idx] for frames_for_camera in camera_frames],
                axis=0,
            )
            state = (
                np.asarray(df["observation.state"].iloc[frame_idx], dtype=np.float32)
                if "observation.state" in df
                else np.zeros(args.piper_action_dim, dtype=np.float32)
            )
            action = (
                np.asarray(df["action"].iloc[frame_idx], dtype=np.float32)
                if "action" in df
                else np.zeros(args.piper_action_dim, dtype=np.float32)
            )
            frames.append(
                {
                    "image": img,
                    "delta_action": np.zeros(args.piper_action_dim, dtype=np.float32),
                    "abs_action": action[: args.piper_action_dim],
                    "init_ee_pose": state[: args.piper_action_dim],
                    "instruction": instruction,
                }
            )

        out_path = out_dir / f"episode_{episode_idx:05d}.npy"
        _write_episode(out_path, frames)

    return n


def _export_dreamdojo(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from groot_dreams.dataloader import MultiVideoActionDataset

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

    n = min(args.num_episodes, len(dataset))
    print(f"Dataset has {len(dataset)} samples; exporting {n} episodes.")

    for idx in range(n):
        data = dataset[idx]
        video = data["video"]  # (C, T, H, W) uint8
        video = video.detach().cpu().numpy()
        c, t, h, w = video.shape
        k = min(args.frames_per_file, t)

        instruction = data.get("ai_caption") or args.instruction
        if isinstance(instruction, (list, tuple)):
            instruction = instruction[0] if instruction else args.instruction
        instruction = str(instruction) if instruction else args.instruction

        frames = []
        for f in range(k):
            img = np.ascontiguousarray(video[:, f].transpose(1, 2, 0)).astype(
                np.uint8
            )  # (H, W, 3)
            frames.append(
                {
                    "image": img,
                    "delta_action": np.zeros(args.piper_action_dim, dtype=np.float32),
                    "instruction": instruction,
                }
            )

        out_path = out_dir / f"episode_{idx:05d}.npy"
        _write_episode(out_path, frames)

    return n


def main() -> None:
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
        n = _export_dreamdojo(args)
    else:
        n = _export_direct(args)
    print(f"Wrote {n} npy files to {Path(args.out_dir)}")


if __name__ == "__main__":
    main()
