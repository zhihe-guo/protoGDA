"""GraphDRP baseline model — late-fusion GNN + cell encoder.

Faithful reimplementation of Nguyen et al. (2022) architecture.
Uses 78-dim atom features matching the original paper's RDKit specification.

Architecture:
  Drug tower:   SMILES → graph → GNN×5(Conv→BN→ReLU→Dropout) → add_pool →
                Linear(32→128) → ReLU → Dropout(0.2)
  Cell tower:   cell features → MLP (gene_expression) or 1D-CNN (mutations)
  Fusion:       concat(256) → 1024 → ReLU → Drop → 128 → ReLU → Drop → 1

Zero src/ dependency.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATv2Conv, GCNConv, GINConv, global_add_pool

ATOM_DIM = 78  # Paper-specified 78-dim atom features


# ---------------------------------------------------------------------------
# Drug tower: GNN encoder
# ---------------------------------------------------------------------------

class _DrugGNN(nn.Module):
    """GNN stack: Conv → BN → ReLU → Dropout, no residual, add-pool at end."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        gnn_type: str,
        dropout: float,
    ):
        super().__init__()
        self.gnn_type = gnn_type.lower()
        self.dropout_rate = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            conv_in = in_dim if i == 0 else hidden_dim
            if self.gnn_type == "gcn":
                conv = GCNConv(conv_in, hidden_dim)
            elif self.gnn_type == "gat":
                conv = GATv2Conv(conv_in, hidden_dim, heads=1, dropout=dropout)
            elif self.gnn_type == "gin":
                mlp = nn.Sequential(
                    nn.Linear(conv_in, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                conv = GINConv(mlp)
            else:
                raise ValueError(f"Unknown gnn_type: {gnn_type}")
            self.convs.append(conv)
            self.bns.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, batch):
        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        return global_add_pool(x, batch)


# ---------------------------------------------------------------------------
# Cell tower
# ---------------------------------------------------------------------------

class _CellMLP(nn.Module):
    """MLP for gene-expression vectors with BatchNorm."""

    def __init__(self, in_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class _CellCNN(nn.Module):
    """1D-CNN for binary mutation vectors."""

    def __init__(self, in_dim: int, output_dim: int):
        super().__init__()
        n_filters = 32
        self.conv1 = nn.Conv1d(1, n_filters, kernel_size=8)
        self.pool1 = nn.MaxPool1d(3)
        self.conv2 = nn.Conv1d(n_filters, n_filters * 2, kernel_size=8)
        self.pool2 = nn.MaxPool1d(3)
        self.conv3 = nn.Conv1d(n_filters * 2, n_filters * 4, kernel_size=8)
        self.pool3 = nn.MaxPool1d(3)

        dummy = torch.zeros(1, 1, in_dim)
        h = self.pool1(F.relu(self.conv1(dummy)))
        h = self.pool2(F.relu(self.conv2(h)))
        h = self.pool3(F.relu(self.conv3(h)))
        flat_dim = h.numel()

        self.fc = nn.Linear(flat_dim, output_dim)

    def forward(self, x):
        h = x.unsqueeze(1)
        h = F.relu(self.conv1(h))
        h = self.pool1(h)
        h = F.relu(self.conv2(h))
        h = self.pool2(h)
        h = F.relu(self.conv3(h))
        h = self.pool3(h)
        h = h.view(h.size(0), -1)
        h = F.relu(self.fc(h))
        return h


# ---------------------------------------------------------------------------
# Main GraphDRP model
# ---------------------------------------------------------------------------

class GraphDRP(nn.Module):
    """GraphDRP: drug molecular graph → GNN, cell features → encoder, late fusion."""

    def __init__(
        self,
        cell_dim: int,
        drug_graphs: list[Data],
        cell_table: torch.Tensor,
        gnn_type: str = "gin",
        num_gnn_layers: int = 5,
        hidden_dim: int = 32,
        drug_output_dim: int = 128,
        cell_mode: str = "gene_expression",
        cell_output_dim: int = 128,
        dropout: float = 0.5,
        gnn_dropout: float = 0.2,
    ):
        super().__init__()

        self.drug_gnn = _DrugGNN(
            in_dim=ATOM_DIM,
            hidden_dim=hidden_dim,
            num_layers=num_gnn_layers,
            gnn_type=gnn_type,
            dropout=gnn_dropout,
        )
        self.drug_fc = nn.Linear(hidden_dim, drug_output_dim)
        self.drug_dropout = nn.Dropout(gnn_dropout)

        self.cell_mode = cell_mode
        if cell_mode == "gene_expression":
            self.cell_encoder = _CellMLP(cell_dim, cell_output_dim)
        elif cell_mode == "mutations":
            self.cell_encoder = _CellCNN(cell_dim, cell_output_dim)
        else:
            raise ValueError(f"Unknown cell_mode: {cell_mode}")

        combined_dim = drug_output_dim + cell_output_dim
        self.fc1 = nn.Linear(combined_dim, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.out = nn.Linear(128, 1)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("cell_table", cell_table)
        self.drug_graphs = drug_graphs

    @classmethod
    def from_config(
        cls, cell_dim: int, cfg: DictConfig,
        drug_graphs: list[Data], cell_table: torch.Tensor,
    ) -> "GraphDRP":
        m = cfg.model
        return cls(
            cell_dim=cell_dim,
            drug_graphs=drug_graphs,
            cell_table=cell_table,
            gnn_type=m.gnn_type,
            num_gnn_layers=m.num_gnn_layers,
            hidden_dim=m.hidden_dim,
            drug_output_dim=m.drug_output_dim,
            cell_mode=m.cell_mode,
            cell_output_dim=m.cell_output_dim,
            dropout=m.dropout,
            gnn_dropout=m.get("gnn_dropout", 0.2),
        )

    def get_branch_params(self) -> dict[str, list[nn.Parameter]]:
        drug_params = list(self.drug_gnn.parameters()) + [self.drug_fc.weight, self.drug_fc.bias]
        cell_params = list(self.cell_encoder.parameters())
        shared_params = list(self.fc1.parameters()) + list(self.fc2.parameters()) + list(self.out.parameters())
        return {"cell": cell_params, "drug": drug_params, "shared": shared_params}

    def get_branch_grad_norms(self) -> dict[str, float]:
        norms: dict[str, float] = {}
        branch_params = self.get_branch_params()
        key_map = {"cell": "cell", "drug": "drug_gnn", "shared": "shared"}
        for branch_name, params in branch_params.items():
            target = key_map.get(branch_name, branch_name)
            if not params:
                norms[target] = 0.0
                continue
            total = 0.0
            rms_sq = 0.0
            count = 0
            for p in params:
                if p.grad is not None:
                    total += p.grad.norm(2).item() ** 2
                    rms_sq += (p.grad ** 2).sum().item()
                    count += p.grad.numel()
            norms[target] = float(total ** 0.5) if total > 0 else 0.0
            if count > 0:
                norms[f"{target}_rms"] = float((rms_sq / count) ** 0.5)
            else:
                norms[f"{target}_rms"] = 0.0
        if "drug_gnn_rms" in norms:
            norms["drug_rms"] = norms["drug_gnn_rms"]
        return norms

    def forward(self, batch):
        drug_ids = batch["drug_ids"]
        cell_ids = batch["cell_ids"]
        device = drug_ids.device

        graph_list = [self.drug_graphs[int(i)] for i in drug_ids]
        batched_graph = Batch.from_data_list(graph_list).to(device)
        drug_feat = self.drug_gnn(batched_graph.x, batched_graph.edge_index, batched_graph.batch)
        drug_feat = F.relu(self.drug_fc(drug_feat))
        drug_feat = self.drug_dropout(drug_feat)

        cidx = cell_ids.to(self.cell_table.device)
        cell_input = self.cell_table[cidx].to(device)
        cell_feat = self.cell_encoder(cell_input)

        xc = torch.cat([drug_feat, cell_feat], dim=-1)
        xc = F.relu(self.fc1(xc))
        xc = self.dropout(xc)
        xc = F.relu(self.fc2(xc))
        xc = self.dropout(xc)
        pred = self.out(xc).view(-1, 1)

        return {"prediction": pred}
