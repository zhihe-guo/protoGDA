"""Reptile meta-learning trainer for drug cold-start.

DrugTaskSampler  – samples per-drug tasks (support/query split) from train split
MetaTrainer       – Reptile outer-loop with no-Adam meta-update + evaluate_with_adapt
"""

from __future__ import annotations

import math
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from src.eval.metrics import compute_metrics, format_metrics


# ---------------------------------------------------------------------------
#  DrugTaskSampler
# ---------------------------------------------------------------------------

class DrugTaskSampler:
    """Samples per-drug few-shot tasks from training data.

    Each task = one drug with its cell-line pairs randomly split into
    support (support_frac) and query (1-support_frac).
    """

    def __init__(
        self,
        train_df,
        drug_id_to_idx: dict[str, int],
        cell_id_to_idx: dict[str, int],
        support_frac: float = 0.7,
        min_samples: int = 4,
    ):
        self.drug_to_samples: dict[int, list[tuple[int, float]]] = {}
        for _, row in train_df.iterrows():
            did = drug_id_to_idx[row["Drug_ID"]]
            cid = cell_id_to_idx[row["Cell_Line_ID"]]
            label = float(row["Y"])
            self.drug_to_samples.setdefault(did, []).append((cid, label))

        self.drug_list = sorted(
            did for did, items in self.drug_to_samples.items()
            if len(items) >= min_samples
        )
        if not self.drug_list:
            raise ValueError(f"No drug has >= {min_samples} pairs in train - "
                             f"cannot build tasks.")
        self.support_frac = support_frac

    def sample_tasks(self, n_tasks: int) -> list[dict[str, Any]]:
        chosen = np.random.choice(self.drug_list, size=n_tasks, replace=True)
        tasks = []
        for drug_idx in chosen:
            items = list(self.drug_to_samples[drug_idx])
            np.random.shuffle(items)
            n_support = max(1, int(len(items) * self.support_frac))
            support_items = items[:n_support]
            query_items = items[n_support:]

            tasks.append({
                "support": _items_to_batch(drug_idx, support_items),
                "query":   _items_to_batch(drug_idx, query_items),
            })
        return tasks


def _items_to_batch(drug_idx: int, items: list[tuple[int, float]]) -> dict:
    if not items:
        return {
            "drug_ids": torch.tensor([], dtype=torch.long),
            "cell_ids": torch.tensor([], dtype=torch.long),
            "labels":   torch.tensor([], dtype=torch.float),
        }
    cids, ys = zip(*items)
    return {
        "drug_ids": torch.full((len(cids),), drug_idx, dtype=torch.long),
        "cell_ids": torch.tensor(cids, dtype=torch.long),
        "labels":   torch.tensor(ys,  dtype=torch.float),
    }


# ---------------------------------------------------------------------------
#  MetaTrainer  (Reptile)
# ---------------------------------------------------------------------------

class MetaTrainer:
    """Reptile meta-trainer.

    Patches applied:
      1. state_dict.clone() for safe parameter snapshots
      2. No AdamW outer optimizer - naive p.data.add_(meta_lr * avg_diff)
      3. evaluate_with_adapt inner-loops on 30% support before predicting 70% query
         (no top-level @torch.no_grad - inner loop needs backward).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        cfg: DictConfig,
        train_df,
        drug_id_to_idx: dict[str, int],
        cell_id_to_idx: dict[str, int],
        valid_loader,
        test_loader,
        device: str,
    ):
        self.model = model.to(device)
        self.cfg = cfg
        self.train_df = train_df
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.device = device

        mcfg = cfg.meta
        self.inner_lr    = mcfg.inner_lr
        self.inner_steps = mcfg.inner_steps
        self.meta_lr     = mcfg.meta_lr
        self.n_tasks     = mcfg.n_tasks
        self.support_frac = mcfg.support_frac
        self.meta_steps_per_epoch = mcfg.meta_steps_per_epoch
        self.wdecay      = mcfg.get("meta_weight_decay", 0.0)
        self.adapt_eval  = mcfg.get("adapt_eval", True)

        self.task_sampler = DrugTaskSampler(
            train_df, drug_id_to_idx, cell_id_to_idx,
            support_frac=self.support_frac,
        )

        self.criterion = torch.nn.MSELoss()

        self.checkpoint_dir = Path(cfg.training.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_rmse = float("inf")
        self.best_epoch = 0
        self.no_improve_count = 0

        self._meta_lr_initial = self.meta_lr
        self._epochs = cfg.training.epochs

    # ------------------------------------------------------------------
    #  Parameter snapshot helpers  (patch 1)
    # ------------------------------------------------------------------

    def _clone_state(self) -> dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.model.state_dict().items()}

    def _diff_state(self, before: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: self.model.state_dict()[k] - before[k] for k in before}

    # ------------------------------------------------------------------
    #  Inner loop  (k-step SGD on support)
    # ------------------------------------------------------------------

    def _inner_loop(self, batch: dict[str, torch.Tensor], lr: float | None = None):
        if lr is None:
            lr = self.inner_lr

        bb = {k: v.to(self.device) for k, v in batch.items()
              if isinstance(v, torch.Tensor)}

        for _ in range(self.inner_steps):
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.zero_()
            out = self.model(bb)
            loss = self.criterion(out["prediction"], bb["labels"])
            loss.backward()
            with torch.no_grad():
                for p in self.model.parameters():
                    if p.grad is not None:
                        p.data.sub_(lr * p.grad)

        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.zero_()

    # ------------------------------------------------------------------
    #  Reptile meta-update  (patch 2)
    # ------------------------------------------------------------------

    def _meta_step(self, tasks: list[dict[str, Any]]):
        diffs: list[dict[str, torch.Tensor]] = []

        for task in tasks:
            if task["support"]["drug_ids"].numel() == 0:
                continue
            before = self._clone_state()
            self._inner_loop(task["support"])
            d = self._diff_state(before)
            diffs.append(d)
            self.model.load_state_dict(before)

        if not diffs:
            return

        with torch.no_grad():
            for p_name, p in self.model.named_parameters():
                avg = sum(d[p_name] for d in diffs) / len(diffs)
                p.data.add_(avg, alpha=self.meta_lr)
                if self.wdecay > 0:
                    p.data.mul_(1.0 - self.meta_lr * self.wdecay)

    # ------------------------------------------------------------------
    #  Cosine LR schedule
    # ------------------------------------------------------------------

    def _cosine_lr(self, epoch: int) -> float:
        return self._meta_lr_initial * 0.5 * (
            1.0 + math.cos(math.pi * epoch / max(self._epochs, 1))
        )

    # ------------------------------------------------------------------
    #  Forward helpers
    # ------------------------------------------------------------------

    def _forward_batch(self, batch: dict, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        bb = {k: v.to(device) for k, v in batch.items()
              if isinstance(v, torch.Tensor)}
        out = self.model(bb)
        return out["prediction"], bb["labels"]

    # ------------------------------------------------------------------
    #  Evaluation with adaptation  (patch 3 - no top-level @torch.no_grad)
    # ------------------------------------------------------------------

    def evaluate_with_adapt(self, loader) -> dict[str, float]:
        self.model.eval()
        preds_all, labels_all = [], []
        total_loss = 0.0
        n_q = 0

        pbar = tqdm(loader, desc="Evaluating (adapt)", leave=False)
        for batch in pbar:
            support, query = self._split_batch(batch)

            if support["drug_ids"].numel() == 0 or query["drug_ids"].numel() == 0:
                with torch.no_grad():
                    pred, y = self._forward_batch(batch, self.device)
                    loss = self.criterion(pred, y)
                total_loss += loss.item()
                n_q += y.size(0)
                preds_all.append(pred.cpu().numpy())
                labels_all.append(y.cpu().numpy())
                continue

            before = self._clone_state()
            self._inner_loop(support)          # needs grad -> no no_grad here

            with torch.no_grad():
                pred, y = self._forward_batch(query, self.device)
                loss = self.criterion(pred, y)
            total_loss += loss.item()
            n_q += y.size(0)
            preds_all.append(pred.cpu().numpy())
            labels_all.append(y.cpu().numpy())

            self.model.load_state_dict(before)

        if not preds_all:
            return {"rmse": float("inf"), "mae": float("inf"),
                    "pearson": 0.0, "spearman": 0.0, "loss": float("inf")}

        metrics = compute_metrics(
            np.concatenate(labels_all),
            np.concatenate(preds_all),
        )
        metrics["loss"] = total_loss / max(n_q, 1)
        return metrics

    def _split_batch(self, batch: dict) -> tuple[dict, dict]:
        dids = batch["drug_ids"]
        cids = batch["cell_ids"]
        ys   = batch["labels"]
        n = dids.size(0)

        if n < 2:
            return batch, batch

        perm = torch.randperm(n)
        n_sup = max(1, int(n * self.support_frac))
        sup_idx = perm[:n_sup]
        qry_idx = perm[n_sup:]

        def _gather(indices):
            return {
                "drug_ids": dids[indices],
                "cell_ids": cids[indices],
                "labels":   ys[indices],
            }
        return _gather(sup_idx), _gather(qry_idx)

    # ------------------------------------------------------------------
    #  Simple evaluate (no adaptation)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, loader) -> dict[str, float]:
        self.model.eval()
        preds, labels = [], []
        total_loss, n = 0.0, 0
        for batch in tqdm(loader, desc="Evaluating (no-adapt)", leave=False):
            pred, y = self._forward_batch(batch, self.device)
            loss = self.criterion(pred, y)
            total_loss += loss.item()
            n += y.size(0)
            preds.append(pred.cpu().numpy())
            labels.append(y.cpu().numpy())
        metrics = compute_metrics(
            np.concatenate(labels),
            np.concatenate(preds),
        )
        metrics["loss"] = total_loss / max(n, 1)
        return metrics

    # ------------------------------------------------------------------
    #  Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "best_val_rmse": self.best_val_rmse,
            "config": dict(self.cfg),
        }
        torch.save(state, self.checkpoint_dir / "last.pt")
        if is_best:
            torch.save(state, self.checkpoint_dir / "best.pt")

    # ------------------------------------------------------------------
    #  Main training loop
    # ------------------------------------------------------------------

    def train(self) -> dict:
        history: list[dict] = []
        early_stop = self.cfg.training.early_stopping_patience

        for epoch in range(1, self._epochs + 1):
            self.meta_lr = self._cosine_lr(epoch)

            # Phase 1: Meta-training
            self.model.train()
            for _ in tqdm(range(self.meta_steps_per_epoch),
                          desc=f"Meta steps epoch {epoch}", leave=False):
                tasks = self.task_sampler.sample_tasks(self.n_tasks)
                self._meta_step(tasks)

            # Phase 2: Validation
            if self.adapt_eval:
                val_metrics = self.evaluate_with_adapt(self.valid_loader)
            else:
                val_metrics = self.evaluate(self.valid_loader)

            record = {
                "epoch": epoch,
                "meta_lr": self.meta_lr,
                "valid": val_metrics,
            }
            history.append(record)

            print(
                f"Epoch {epoch}/{self._epochs} | "
                f"Valid {format_metrics(val_metrics)} loss={val_metrics['loss']:.4f} | "
                f"meta_lr={self.meta_lr:.2e}"
            )

            is_best = val_metrics["rmse"] < self.best_val_rmse
            if is_best:
                self.best_val_rmse = val_metrics["rmse"]
                self.best_epoch = epoch
                self.no_improve_count = 0
            else:
                self.no_improve_count += 1

            self.save_checkpoint(epoch, is_best=is_best)

            if self.no_improve_count >= early_stop:
                print(f"Early stopping at epoch {epoch} "
                      f"(best was epoch {self.best_epoch}, "
                      f"RMSE={self.best_val_rmse:.4f}).")
                break

        history_path = self.checkpoint_dir / "meta_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        return {"history": history, "best_val_rmse": self.best_val_rmse}
