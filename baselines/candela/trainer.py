"""Training loop for CANDELA baseline.

Self-contained; imports metrics locally. No gradient clipping.
No label_scaler (CANDELA uses raw log(IC50) directly).
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
        self.best_epoch = 0
        self.no_improve_count = 0
        self.checkpoint_dir = Path(cfg.training.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.gradnorm_monitor = cfg.training.get("gradnorm_monitor", False)
        self.checkpoint_metric = cfg.training.get("checkpoint_metric", "rmse")
        if self.checkpoint_metric not in {"rmse", "pearson_r"}:
            raise ValueError(
                "training.checkpoint_metric must be 'rmse' or 'pearson_r', "
                f"got {self.checkpoint_metric!r}."
            )
        self.best_val_score = (
            float("inf") if self.checkpoint_metric == "rmse" else float("-inf")
        )

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
        alphas, betas, gammas = [], [], []
        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(loader, desc="Evaluating", leave=False):
            batch = self._to_device(batch)
            y = batch["labels"]
            y = y.view(-1, 1)  # ensure (N, 1) for MSE with pred (N, 1)
            out = self.model(batch)
            pred = out["prediction"]
            total_loss += self.criterion(pred, y).item()
            n_batches += 1
            preds.append(pred.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

            if "alpha" in out:
                alphas.append(out["alpha"].detach().cpu().numpy())
                betas.append(out["beta"].detach().cpu().numpy())
                gammas.append(out["gamma"].detach().cpu().numpy())

        y_true = np.concatenate(labels)
        y_pred = np.concatenate(preds)
        metrics = compute_metrics(y_true, y_pred)
        metrics["loss"] = total_loss / max(n_batches, 1)

        if alphas:
            alpha_arr = np.concatenate(alphas)
            beta_arr = np.concatenate(betas)
            gamma_arr = np.concatenate(gammas)
            metrics["alpha_mean"] = float(alpha_arr.mean())
            metrics["alpha_std"] = float(alpha_arr.std())
            metrics["beta_mean"] = float(beta_arr.mean())
            metrics["beta_std"] = float(beta_arr.std())
            metrics["gamma_mean"] = float(gamma_arr.mean())
            metrics["gamma_std"] = float(gamma_arr.std())
            # Ratio of interaction to total prediction
            abs_total = float(np.abs(alpha_arr + beta_arr + gamma_arr).mean())
            abs_gamma = float(np.abs(gamma_arr).mean())
            metrics["gamma_ratio"] = abs_gamma / (abs_total + 1e-8)

        return metrics

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        preds, labels = [], []
        log_every = self.cfg.training.log_every

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for step, batch in enumerate(pbar):
            batch = self._to_device(batch)
            y = batch["labels"]
            y = y.view(-1, 1)  # ensure (N, 1) for MSE with pred (N, 1)

            self.optimizer.zero_grad()
            out = self.model(batch)
            pred = out["prediction"]
            loss = self.criterion(pred, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            preds.append(pred.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

            if (step + 1) % log_every == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        y_true = np.concatenate(labels)
        y_pred = np.concatenate(preds)
        metrics = compute_metrics(y_true, y_pred)
        metrics["loss"] = total_loss / max(n_batches, 1)
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
        history_path = self.checkpoint_dir / "history.json"

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

            # Incremental save: write history after every epoch
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

            extra = ""
            if "gamma_ratio" in val_metrics:
                extra = (f" | gamma_ratio={val_metrics['gamma_ratio']:.3f}"
                         f" alpha={val_metrics.get('alpha_mean', 0):.4f}"
                         f" beta={val_metrics.get('beta_mean', 0):.4f}")
            print(
                f"Epoch {epoch}/{self.cfg.training.epochs} | "
                f"Train {format_metrics(train_metrics)} loss={train_metrics['loss']:.4f} | "
                f"Valid {format_metrics(val_metrics)} loss={val_metrics['loss']:.4f} | "
                f"LR={self._current_lr():.2e}"
                f"{extra}"
            )

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
                      f"(best was epoch {self.best_epoch}, "
                      f"{self.checkpoint_metric}={self.best_val_score:.4f}).")
                break

        # Final save of history
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        # Save summary for quick inspection
        best_record = history[self.best_epoch - 1]
        summary = {
            "model": self.cfg.training.checkpoint_dir,
            "best_epoch": self.best_epoch,
            "best_val": best_record["valid"],
            "train_at_best": best_record["train"],
            "total_epochs": len(history),
        }
        with open(self.checkpoint_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return {"history": history, "best_val_rmse": self.best_val_rmse}
