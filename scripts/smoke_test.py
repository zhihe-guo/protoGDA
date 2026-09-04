"""Quick smoke test: 1-epoch training to verify data + model + loop integrity."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_device
from src.data.dataset import get_dataloaders
from src.models.model import CellDrugModel
from src.training.trainer import Trainer

config_path = ROOT / "config" / "default.yaml"

cfg = load_config(config_path)
cfg.training.epochs = 1
cfg.training.warmup_epochs = 0
cfg.training.early_stopping_patience = 50
device = resolve_device(cfg.training.device)
print(f"Device: {device}  Config: {config_path}")

train_loader, valid_loader, test_loader, registries, meta = get_dataloaders(cfg)
print(f"Cell dim: {meta['cell_dim']}")

model = CellDrugModel.from_config(
    cell_dim=meta["cell_dim"],
    cfg=cfg,
    drug_graphs=registries["drug_graphs"],
    cell_features_table=registries["cell_table"],
    drug_morgan_table=registries.get("drug_morgan_table"),
    drug_chemberta_table=registries.get("drug_chemberta_table"),
)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Params: {n_params:,}")

trainer = Trainer(
    model=model, cfg=cfg, train_loader=train_loader,
    valid_loader=valid_loader, device=device,
)
result = trainer.train()
print(f"Best val RMSE: {result['best_val_rmse']:.4f}")
