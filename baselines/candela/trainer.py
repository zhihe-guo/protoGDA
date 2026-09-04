"""Training loop for CANDELA baseline.

Self-contained; imports metrics locally. No gradient clipping.
No label_scaler (CANDELA uses raw log(IC50) directly).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from torch.optim import AdamW
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
    ):
        self.model = model.to(device)
        self.cfg = cfg
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device

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

    def train(self) -> dict:
        history: list[dict] = []
        warmup_epochs = self.cfg.training.warmup_epochs
        early_stop_patience = self.cfg.training.early_stopping_patience
        history_path = self.checkpoint_dir / "history.json"

        for epoch in range(1, self.cfg.training.epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.evaluate(self.valid_loader)

            if epoch <= warmup_epochs and self.scheduler is not None:
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

            val_score = val_metrics[self.checkpoint_metric]
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
