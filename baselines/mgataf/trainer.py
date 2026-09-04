"""Training loop for MGATAF baseline.

Self-contained; imports metrics locally instead of from src/.
No gradient clipping. Gradient norms are monitored read-only.
Label scaler is preserved for inverse-transform during eval.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ReduceLROnPlateau, SequentialLR
from tqdm import tqdm

from metrics import compute_metrics, format_metrics


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        cfg: DictConfig,
        train_loader,
        valid_loader,
        device: str,
        label_scaler: dict | None = None,
    ):
        self.model = model.to(device)
        self.cfg = cfg
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.label_scaler = label_scaler

        self.criterion = torch.nn.MSELoss()
        self.optimizer = Adam(
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
        # Normalize the requested metric onto the keys this trainer's metrics
        # module actually reports: MGATAF's local metrics use "pearson", while
        # the unified cv_runner protocol asks for "pearson_r".
        self.checkpoint_metric = cfg.training.get("checkpoint_metric", "rmse")
        if self.checkpoint_metric not in {"rmse", "pearson_r"}:
            raise ValueError(
                "training.checkpoint_metric must be 'rmse' or 'pearson_r', "
                f"got {self.checkpoint_metric!r}."
            )
        self._metric_key = "rmse" if self.checkpoint_metric == "rmse" else "pearson"
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

    def _inverse_scale(self, arr: np.ndarray) -> np.ndarray:
        if self.label_scaler is None:
            return arr
        return arr * self.label_scaler["span"] + self.label_scaler["min"]

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

        if self.label_scaler is not None:
            raw = compute_metrics(self._inverse_scale(y_true), self._inverse_scale(y_pred))
            for k, v in raw.items():
                metrics[f"{k}_raw"] = v
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

        if self.label_scaler is not None:
            raw = compute_metrics(self._inverse_scale(y_true), self._inverse_scale(y_pred))
            metrics["rmse_raw"] = raw["rmse"]
            metrics["pearson_raw"] = raw["pearson"]
        return metrics

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_rmse": self.best_val_rmse,
            "best_val_score": self.best_val_score,
            "checkpoint_metric": self.checkpoint_metric,
        }
        torch.save(state, self.checkpoint_dir / "last.pt")
        if is_best:
            torch.save(state, self.checkpoint_dir / "best.pt")

    def save_history(self, history: list[dict]) -> None:
        """Atomically persist completed epochs so interrupted runs remain usable."""
        history_path = self.checkpoint_dir / "history.json"
        tmp_path = history_path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        tmp_path.replace(history_path)

    def train(self) -> dict:
        history: list[dict] = []
        warmup_epochs = self.cfg.training.warmup_epochs
        early_stop_patience = self.cfg.training.early_stopping_patience

        for epoch in range(1, self.cfg.training.epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.evaluate(self.valid_loader)

            # A cosine schedule is a SequentialLR (warmup + cosine) and must
            # advance every epoch. Plateau warmup, by contrast, advances only
            # during its warmup window before ReduceLROnPlateau takes over.
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

            val_score = val_metrics[self._metric_key]
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

            self.save_checkpoint(epoch, is_best=is_best)
            self.save_history(history)

            if self.no_improve_count >= early_stop_patience:
                print(f"Early stopping at epoch {epoch} "
                      f"(best was epoch {self.best_epoch}, "
                      f"{self.checkpoint_metric}={self.best_val_score:.4f}).")
                break

        return {"history": history, "best_val_rmse": self.best_val_rmse}
