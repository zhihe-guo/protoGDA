"""Configuration loading utilities."""

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def load_config(path: str | Path = "config/default.yaml") -> DictConfig:
    """Load YAML config and resolve interpolations."""
    cfg = OmegaConf.load(path)
    return cfg


def resolve_device(device: str) -> str:
    """Resolve 'auto' to cuda/cpu."""
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def config_to_dict(cfg: DictConfig) -> dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
