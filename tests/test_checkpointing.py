"""Unit tests for the versioned checkpoint contract."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.training.checkpointing import (
    atomic_torch_save,
    build_checkpoint,
    json_fingerprint,
    load_checkpoint,
    resolved_config,
    restore_training_state,
)


class CheckpointingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = OmegaConf.create({"training": {"seed": 42, "epochs": 2}})
        self.manifest = {"protocol": "v3", "fold": 1, "parts": {"train": {"n_rows": 3}}}
        self.model = torch.nn.Linear(2, 1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1)

    def test_round_trip_restores_model_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            payload = build_checkpoint(
                model=self.model, optimizer=self.optimizer, scheduler=self.scheduler,
                plateau_scheduler=None, epoch=1,
                trainer_state={"history": [{"epoch": 1}], "best_epoch": 1},
                cfg=self.cfg, run_manifest=self.manifest,
            )
            atomic_torch_save(payload, path)
            loaded = load_checkpoint(
                path, map_location="cpu",
                expected_config_fingerprint=json_fingerprint(resolved_config(self.cfg)),
                expected_manifest_fingerprint=json_fingerprint(self.manifest),
            )
            target = torch.nn.Linear(2, 1)
            target_optimizer = torch.optim.Adam(target.parameters(), lr=1e-3)
            target_scheduler = torch.optim.lr_scheduler.StepLR(target_optimizer, step_size=1)
            state = restore_training_state(
                loaded, model=target, optimizer=target_optimizer,
                scheduler=target_scheduler, plateau_scheduler=None,
            )
            self.assertEqual(state["best_epoch"], 1)
            for original, restored in zip(self.model.parameters(), target.parameters()):
                self.assertTrue(torch.equal(original, restored))

    def test_rejects_incompatible_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            atomic_torch_save(build_checkpoint(
                model=self.model, optimizer=self.optimizer, scheduler=None,
                plateau_scheduler=None, epoch=1, trainer_state={"history": []},
                cfg=self.cfg, run_manifest=self.manifest,
            ), path)
            with self.assertRaises(ValueError):
                load_checkpoint(
                    path, map_location="cpu",
                    expected_manifest_fingerprint=json_fingerprint({"protocol": "v3", "fold": 2}),
                )


if __name__ == "__main__":
    unittest.main()
