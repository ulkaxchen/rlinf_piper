"""Small compatibility shim for DreamDojo-only rollout scripts.

DreamDojo's `.venv` has the Cosmos stack that reproduces the reference piper
videos, but it does not carry RLinf's Ray scheduler dependencies. The world
model sample/parity scripts only need the scheduler module for the global
Worker torch device attributes, so provide a minimal fallback when importing
the real scheduler fails.
"""

from __future__ import annotations

import sys
import types

import torch


def ensure_scheduler_worker():
    try:
        from rlinf.scheduler import Worker

        return Worker
    except ModuleNotFoundError as exc:
        if exc.name != "ray":
            raise

    scheduler_stub = types.ModuleType("rlinf.scheduler")

    class Worker:
        torch_device_type = "cuda" if torch.cuda.is_available() else "cpu"
        torch_platform = torch.cuda if torch.cuda.is_available() else torch

    class WorkerInfo:
        pass

    scheduler_stub.Worker = Worker
    scheduler_stub.WorkerInfo = WorkerInfo
    sys.modules["rlinf.scheduler"] = scheduler_stub
    return Worker
