"""Training loop for GraphDRP baseline.

Self-contained; imports metrics locally. No gradient clipping.
No label_scaler (GraphDRP uses raw log10(IC50) directly).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ReduceLROnPlateau, SequentialLR
from tqdm import tqdm

from metrics import compute_metrics, format_metrics

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.training.checkpointing import (  # noqa: E402
    atomic_torch_save,
    build_checkpoint,
    json_fingerprint,
    load_checkpoint,
    resolved_config,
    restore_training_state,
)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        cfg: DictConfig,
        train_loader,
        valid_loader,
        device: str,
        run_manifest: dict | None = None,
    ):
        self.model = model.to(device)
        self.cfg = cfg
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.run_manifest = run_manifest or {}
        self.config_fingerprint = json_fingerprint(resolved_config(cfg))
        self.manifest_fingerprint = json_fingerprint(self.run_manifest)

        self.criterion = torch.nn.MSELoss()
        self.optimizer = AdamW(
            model.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )
        self.scheduler, self.plateau_scheduler = self._build_scheduler()
        self.best_val_rmse = float("inf")
        self.checkpoint_metric = cfg.training.get("checkpoint_metric", "rmse")
        if self.checkpoint_metric not in {"rmse", "pearson_r"}:
            raise ValueError("training.checkpoint_metric must be 'rmse' or 'pearson_r'.")
        self.best_val_score = float("inf") if self.checkpoint_metric == "rmse" else float("-inf")
        self.best_epoch = 0
        self.no_improve_count = 0
        self.checkpoint_dir = Path(cfg.training.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.gradnorm_monitor = cfg.training.get("gradnorm_monitor", False)

    def _build_scheduler(self):
        warmup = self.cfg.training.warmup_epochs
        scheduler_type = self.cfg.training.get("scheduler", "plateau")

        warmup_sched = None
        if warmup > 0:
            warmup_sched = LinearLR(
                self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup,
            )

        if scheduler_type == "cosine":
            total = self.cfg.training.epochs
            main = CosineAnnealingLR(
                self.optimizer, T_max=max(total - warmup, 1), eta_min=1e-6,
            )
            if warmup_sched is not None:
                return SequentialLR(self.optimizer, schedulers=[warmup_sched, main], milestones=[warmup]), None
            return main, None

        if scheduler_type == "plateau":
            lr_patience = self.cfg.training.get("lr_reduce_patience", 5)
            lr_factor = self.cfg.training.get("lr_reduce_factor", 0.5)
            lr_min = self.cfg.training.get("lr_min", 1e-6)
            plateau = ReduceLROnPlateau(
                self.optimizer, mode="min", factor=lr_factor,
                patience=lr_patience, min_lr=lr_min,
            )
            if warmup_sched is not None:
                return warmup_sched, plateau
            return None, plateau

        if warmup_sched is not None:
            return warmup_sched, None
        return None, None

    def _to_device(self, batch: dict) -> dict:
        return {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

    def _current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    @torch.no_grad()
    def evaluate(self, loader) -> dict[str, float]:
        self.model.eval()
        preds, labels = [], []
        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(loader, desc="Evaluating", leave=False):
            batch = self._to_device(batch)
            y = batch["labels"]
            out = self.model(batch)
            pred = out["prediction"].view(-1)
            total_loss += self.criterion(pred, y).item()
            n_batches += 1
            preds.append(pred.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

        y_true = np.concatenate(labels)
        y_pred = np.concatenate(preds)
        metrics = compute_metrics(y_true, y_pred)
        metrics["loss"] = total_loss / max(n_batches, 1)
        return metrics

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        preds, labels = [], []

        sum_gn_cell = sum_gn_drug = 0.0
        sum_rms_cell = sum_rms_drug = 0.0
        log_every = self.cfg.training.log_every

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for step, batch in enumerate(pbar):
            batch = self._to_device(batch)
            y = batch["labels"]

            self.optimizer.zero_grad()
            out = self.model(batch)
            pred = out["prediction"].view(-1)
            loss = self.criterion(pred, y)
            loss.backward()

            postfix = {"loss": f"{loss.item():.4f}"}
            if self.gradnorm_monitor and hasattr(self.model, "get_branch_grad_norms"):
                norms = self.model.get_branch_grad_norms()
                gn_cell = norms.get("cell", 0.0)
                gn_drug = norms.get("drug_gnn", norms.get("drug", 0.0))
                rms_cell = norms.get("cell_rms", 0.0)
                rms_drug = norms.get("drug_rms", norms.get("drug_gnn_rms", 0.0))
                sum_gn_cell += gn_cell
                sum_gn_drug += gn_drug
                sum_rms_cell += rms_cell
                sum_rms_drug += rms_drug
                postfix.update({
                    "gn_c": f"{gn_cell:.3f}", "gn_d": f"{gn_drug:.3f}",
                    "rms_c": f"{rms_cell:.3f}", "rms_d": f"{rms_drug:.3f}",
                })

            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            preds.append(pred.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

            if (step + 1) % log_every == 0:
                pbar.set_postfix(postfix)

        y_true = np.concatenate(labels)
        y_pred = np.concatenate(preds)
        metrics = compute_metrics(y_true, y_pred)
        metrics["loss"] = total_loss / max(n_batches, 1)

        if n_batches > 0:
            metrics["gn_cell"] = sum_gn_cell / n_batches
            metrics["gn_drug"] = sum_gn_drug / n_batches
            metrics["rms_cell"] = sum_rms_cell / n_batches
            metrics["rms_drug"] = sum_rms_drug / n_batches

        return metrics

    def _trainer_state(self, history: list[dict]) -> dict:
        return {
            "best_val_rmse": self.best_val_rmse,
            "best_val_score": self.best_val_score,
            "best_epoch": self.best_epoch,
            "no_improve_count": self.no_improve_count,
            "checkpoint_metric": self.checkpoint_metric,
            "history": history,
        }

    def save_checkpoint(self, epoch: int, history: list[dict] | None = None, is_best: bool = False):
        state = build_checkpoint(
            model=self.model, optimizer=self.optimizer, scheduler=self.scheduler,
            plateau_scheduler=self.plateau_scheduler, epoch=epoch,
            trainer_state=self._trainer_state(history or []), cfg=self.cfg,
            run_manifest=self.run_manifest,
        )
        atomic_torch_save(state, self.checkpoint_dir / "last.pt")
        if is_best:
            atomic_torch_save(state, self.checkpoint_dir / "best.pt")

    def resume(self, path: str | Path) -> list[dict]:
        state = load_checkpoint(
            path, map_location=self.device,
            expected_config_fingerprint=self.config_fingerprint,
            expected_manifest_fingerprint=self.manifest_fingerprint,
        )
        trainer_state = restore_training_state(
            state, model=self.model, optimizer=self.optimizer,
            scheduler=self.scheduler, plateau_scheduler=self.plateau_scheduler,
        )
        if trainer_state.get("checkpoint_metric") != self.checkpoint_metric:
            raise ValueError("Checkpoint selection metric does not match current configuration.")
        self.best_val_rmse = float(trainer_state["best_val_rmse"])
        self.best_val_score = float(trainer_state["best_val_score"])
        self.best_epoch = int(trainer_state["best_epoch"])
        self.no_improve_count = int(trainer_state["no_improve_count"])
        history = trainer_state.get("history", [])
        if not isinstance(history, list) or len(history) != int(state["epoch"]):
            raise ValueError("Checkpoint history is missing or does not match its epoch.")
        return history

    def train(self, resume_path: str | Path | None = None) -> dict:
        history = self.resume(resume_path) if resume_path is not None else []
        warmup_epochs = self.cfg.training.warmup_epochs
        early_stop_patience = self.cfg.training.early_stopping_patience

        for epoch in range(len(history) + 1, self.cfg.training.epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.evaluate(self.valid_loader)
            val_score = val_metrics[self.checkpoint_metric]
            if not math.isfinite(float(val_score)):
                raise FloatingPointError(f"Non-finite validation {self.checkpoint_metric} at epoch {epoch}.")

            if self.scheduler is not None and (
                self.cfg.training.get("scheduler", "plateau") == "cosine"
                or epoch <= warmup_epochs
            ):
                self.scheduler.step()
            elif self.plateau_scheduler is not None:
                self.plateau_scheduler.step(val_metrics["rmse"])

            record = {
                "epoch": epoch, "train": train_metrics, "valid": val_metrics,
                "lr": self._current_lr(),
            }
            history.append(record)

            print(
                f"Epoch {epoch}/{self.cfg.training.epochs} | "
                f"Train {format_metrics(train_metrics)} loss={train_metrics['loss']:.4f} | "
                f"Valid {format_metrics(val_metrics)} loss={val_metrics['loss']:.4f} | "
                f"LR={self._current_lr():.2e}"
            )
            if "gn_cell" in train_metrics:
                gn_c = train_metrics.get("gn_cell", 0)
                gn_d = train_metrics.get("gn_drug", 0)
                rms_c = train_metrics.get("rms_cell", 0)
                rms_d = train_metrics.get("rms_drug", 0)
                rms_r = rms_c / (rms_d + 1e-8)
                print(f"  [GRAD] avg/step | L2: cell={gn_c:.3f} drug={gn_d:.3f} | "
                      f"RMS: cell={rms_c:.4f} drug={rms_d:.4f} ratio={rms_r:.2f}")

            is_best = (
                val_score < self.best_val_score
                if self.checkpoint_metric == "rmse"
                else val_score > self.best_val_score
            )
            if is_best:
                self.best_val_score = val_score
                self.best_val_rmse = val_metrics["rmse"]
                self.best_epoch = epoch
                self.no_improve_count = 0
            else:
                self.no_improve_count += 1

            self.save_checkpoint(epoch, history, is_best=is_best)

            if self.no_improve_count >= early_stop_patience:
                print(f"Early stopping at epoch {epoch} "
                      f"(best was epoch {self.best_epoch}, RMSE={self.best_val_rmse:.4f}).")
                break

        history_path = self.checkpoint_dir / "history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        return {"history": history, "best_val_rmse": self.best_val_rmse}
