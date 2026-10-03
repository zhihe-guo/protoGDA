"""Versioned, atomic checkpoints shared by all distributable models."""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf


CHECKPOINT_SCHEMA_VERSION = 1


def json_fingerprint(value: Any) -> str:
    """Stable SHA-256 fingerprint for JSON-compatible experiment metadata."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def resolved_config(cfg) -> dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def atomic_torch_save(payload: dict[str, Any], path: str | Path) -> None:
    """Write a checkpoint completely before making it visible at *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def build_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    plateau_scheduler: Any,
    epoch: int,
    trainer_state: dict[str, Any],
    cfg,
    run_manifest: dict[str, Any] | None = None,
    extra_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = resolved_config(cfg)
    manifest = run_manifest or {}
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "plateau_scheduler_state_dict": (
            plateau_scheduler.state_dict() if plateau_scheduler is not None else None
        ),
        "trainer_state": trainer_state,
        "resolved_config": config,
        "config_fingerprint": json_fingerprint(config),
        "run_manifest": manifest,
        "run_manifest_fingerprint": json_fingerprint(manifest),
        "rng_state": capture_rng_state(),
        "extra_state": extra_state or {},
    }


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device,
    expected_config_fingerprint: str | None = None,
    expected_manifest_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Load and validate a checkpoint before it is applied to a model."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint is not a mapping: {path}")
    version = state.get("schema_version")
    if version != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported or legacy checkpoint schema in {path}: {version!r}; "
            f"expected {CHECKPOINT_SCHEMA_VERSION}."
        )
    required = {
        "epoch", "model_state_dict", "optimizer_state_dict", "trainer_state",
        "resolved_config", "config_fingerprint", "run_manifest",
        "run_manifest_fingerprint", "rng_state",
    }
    missing = sorted(required - set(state))
    if missing:
        raise ValueError(f"Checkpoint {path} misses required fields: {missing}")
    if expected_config_fingerprint and state["config_fingerprint"] != expected_config_fingerprint:
        raise ValueError(f"Checkpoint config fingerprint does not match current run: {path}")
    if expected_manifest_fingerprint and state["run_manifest_fingerprint"] != expected_manifest_fingerprint:
        raise ValueError(f"Checkpoint split/data manifest does not match current run: {path}")
    return state


def restore_training_state(
    checkpoint: dict[str, Any],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    plateau_scheduler: Any,
) -> dict[str, Any]:
    """Restore an epoch-boundary checkpoint and return serialized trainer state."""
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if plateau_scheduler is not None and checkpoint.get("plateau_scheduler_state_dict") is not None:
        plateau_scheduler.load_state_dict(checkpoint["plateau_scheduler_state_dict"])
    restore_rng_state(checkpoint["rng_state"])
    return checkpoint["trainer_state"]
