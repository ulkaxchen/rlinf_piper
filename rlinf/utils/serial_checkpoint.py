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

"""Streaming checkpoint helpers for memory-constrained serial training."""

from __future__ import annotations

import gc
import os
from collections.abc import Iterable
from pathlib import Path

import torch


def trim_host_allocator() -> None:
    """Return freed glibc heap pages to the OS when available."""
    gc.collect()
    try:
        import ctypes

        malloc_trim = getattr(ctypes.CDLL(None), "malloc_trim", None)
        if malloc_trim is not None:
            malloc_trim(0)
    except (AttributeError, OSError):
        pass


def serial_model_state_exists(checkpoint_path: str) -> bool:
    """Return whether a serial checkpoint contains its model manifest."""
    return os.path.isfile(os.path.join(checkpoint_path, "serial_model", "manifest.pt"))


def save_trainable_model_state(model: torch.nn.Module, checkpoint_path: str) -> None:
    """Stream trainable parameters to separate files without gathering a state dict."""
    save_named_tensors(
        (
            (name, parameter.detach())
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ),
        checkpoint_path,
    )


def save_named_tensors(
    named_tensors: Iterable[tuple[str, torch.Tensor]], checkpoint_path: str
) -> None:
    """Stream named tensors to the serial model checkpoint directory."""
    model_dir = Path(checkpoint_path) / "serial_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for index, (name, parameter) in enumerate(named_tensors):
        filename = f"parameter_{index:05d}.pt"
        tensor = parameter.detach()
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
        torch.save(tensor, model_dir / filename)
        manifest.append(
            {
                "name": name,
                "filename": filename,
                "shape": tuple(parameter.shape),
                "dtype": str(parameter.dtype),
            }
        )
        del tensor

    if not manifest:
        raise RuntimeError("Serial checkpoint found no trainable model parameters.")
    torch.save({"parameters": manifest}, model_dir / "manifest.pt")


def load_trainable_model_state(model: torch.nn.Module, checkpoint_path: str) -> None:
    """Load trainable parameters one at a time into an unwrapped model."""
    model_dir = Path(checkpoint_path) / "serial_model"
    manifest_path = model_dir / "manifest.pt"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing serial model manifest: {manifest_path}")

    manifest = torch.load(manifest_path, map_location="cpu", weights_only=True)
    named_parameters = dict(model.named_parameters())
    for item in manifest["parameters"]:
        name = item["name"]
        if name not in named_parameters:
            raise KeyError(f"Serial checkpoint parameter not found in model: {name}")
        target = named_parameters[name]
        tensor = torch.load(
            model_dir / item["filename"],
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        if tuple(target.shape) != tuple(tensor.shape):
            raise ValueError(
                f"Serial checkpoint shape mismatch for {name}: "
                f"{tuple(tensor.shape)} != {tuple(target.shape)}"
            )
        target.data.copy_(tensor.to(dtype=target.dtype, device=target.device))
        del tensor

    del named_parameters, manifest
    trim_host_allocator()
