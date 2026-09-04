"""CANDELA baseline model — modular GNN with score decomposition.

Faithful reimplementation of Campana et al. (2024) NAR Genomics & Bioinformatics.
Architecture: Drug Encoder (3x GATv2Conv) + Drug Module (alpha_i) +
Expression Encoder + Expression Module (beta_j) +
Interaction Module (gamma_ij via cross-attention pooling).
Final: y_hat = alpha_i + beta_j + gamma_ij.

Zero src/ dependency.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.aggr import AttentionalAggregation

ATOM_DIM = 78  # 78-dim atom features, consistent with GraphDRP baseline


# ---------------------------------------------------------------------------
# Drug Encoder: 3 x GATv2Conv (8 heads) + LeakyReLU -> 128-dim node embeddings
# ---------------------------------------------------------------------------

class DrugEncoder(nn.Module):
    """3-layer GATv2Conv stack with 8 attention heads per layer."""

    def __init__(
        self,
        in_dim: int,
        node_embed_dim: int = 128,
        heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.heads = heads
        self.node_embed_dim = node_embed_dim

        self.conv1 = GATv2Conv(in_dim, node_embed_dim, heads=heads, dropout=dropout, concat=False)
        self.conv2 = GATv2Conv(node_embed_dim, node_embed_dim, heads=heads, dropout=dropout, concat=False)
        self.conv3 = GATv2Conv(node_embed_dim, node_embed_dim, heads=heads, dropout=dropout, concat=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.conv3(x, edge_index)
        x = F.leaky_relu(x, negative_slope=0.2)
        return x  # (total_nodes, node_embed_dim)


# ---------------------------------------------------------------------------
# Drug Module: Self-Attention Pooling -> drug emb -> score MLP -> alpha_i
# ---------------------------------------------------------------------------

class DrugModule(nn.Module):
    """Self-attention pooling + MLP to produce drug cytotoxicity score alpha_i."""

    def __init__(
        self,
        node_embed_dim: int = 128,
        pool_hidden: int = 1024,
        score_hidden: int = 1152,
    ):
        super().__init__()
        # Self-attention pooling gate network
        gate_nn = nn.Sequential(
            nn.Linear(node_embed_dim, pool_hidden),
            nn.ReLU(),
            nn.Linear(pool_hidden, 1),
        )
        self.pool = AttentionalAggregation(gate_nn=gate_nn)

        # Drug score MLP: embed -> hidden -> sigmoid -> scalar
        self.score_net = nn.Sequential(
            nn.Linear(node_embed_dim, score_hidden),
            nn.Sigmoid(),
            nn.Linear(score_hidden, 1),
        )

    def forward(self, node_emb: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        drug_emb = self.pool(node_emb, batch)  # (batch_size, node_embed_dim)
        alpha = self.score_net(drug_emb)  # (batch_size, 1)
        return alpha, drug_emb


# ---------------------------------------------------------------------------
# Expression Encoder: 2-layer MLP (in_dim -> 1024 -> 128)
# ---------------------------------------------------------------------------

class ExpressionEncoder(nn.Module):
    """2-layer MLP for gene expression data, producing 128-dim cell embedding."""

    def __init__(self, in_dim: int, hidden: int = 1024, bottleneck: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, bottleneck),
            nn.ReLU(),
        )

    def forward(self, expr: torch.Tensor) -> torch.Tensor:
        return self.net(expr)  # (batch_size, bottleneck)


# ---------------------------------------------------------------------------
# Expression Module: MLP to produce cell survivability score beta_j
# ---------------------------------------------------------------------------

class ExpressionModule(nn.Module):
    """MLP producing cell-line survivability score beta_j."""

    def __init__(self, bottleneck: int = 128, score_hidden: int = 660):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Linear(bottleneck, score_hidden),
            nn.ReLU(),
            nn.Linear(score_hidden, 1),
        )

    def forward(self, cell_emb: torch.Tensor) -> torch.Tensor:
        return self.score_net(cell_emb)  # (batch_size, 1)


# ---------------------------------------------------------------------------
# Interaction Module: Cross-attention pooling + MLP -> gamma_ij
# ---------------------------------------------------------------------------

class InteractionModule(nn.Module):
    """Cross-attention pooling: cell embedding attends to drug node embeddings
    to produce interaction residual gamma_ij. Uses batched scatter ops."""

    def __init__(
        self,
        node_embed_dim: int = 128,
        cell_embed_dim: int = 128,
        hidden: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        # Project cell embedding to query space
        self.cell_proj = nn.Linear(cell_embed_dim, node_embed_dim)

        # Cross-attention: attention scores from cell query against drug nodes
        self.attn_score = nn.Linear(node_embed_dim * 2, 1)

        # Interaction MLP: joint emb -> hidden -> sigmoid -> scalar
        self.interaction_net = nn.Sequential(
            nn.Linear(node_embed_dim * 2 + cell_embed_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        node_emb: torch.Tensor,
        batch: torch.Tensor,
        cell_emb: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Batched cross-attention with scatter_softmax.

        Args:
            node_emb: (total_nodes, node_embed_dim)
            batch: (total_nodes,) batch index per node
            cell_emb: (batch_size, cell_embed_dim)
            batch_size: number of drug-cell pairs
        Returns:
            gamma: (batch_size, 1)
        """
        device = node_emb.device

        # Expand cell embeddings to all nodes: (total_nodes, cell_embed_dim)
        cell_emb_expanded = cell_emb[batch]

        # Cell query: (total_nodes, node_embed_dim)
        cell_query = self.cell_proj(cell_emb_expanded)

        # Attention scores: cat(node, cell_query) -> score
        attn_in = torch.cat([node_emb, cell_query], dim=-1)
        raw_scores = self.attn_score(attn_in).squeeze(-1)  # (total_nodes,)

        # Scatter softmax: softmax within each graph
        scores_max = scatter_max(raw_scores, batch, dim=0)[batch]
        scores_exp = torch.exp(raw_scores - scores_max)
        scores_sum = scatter_sum(scores_exp, batch, dim=0)[batch] + 0.0
        attn_weights = scores_exp / (scores_sum + 1e-8)

        # Weighted sum of node features: (batch_size, node_embed_dim)
        weighted = attn_weights.unsqueeze(-1) * node_emb
        attended_drug = scatter_sum(weighted, batch, dim=0)  # (batch_size, node_embed_dim)

        # Joint representation: cat(cell_emb, attended_drug, cell_emb)
        joint = torch.cat([cell_emb, attended_drug, cell_emb], dim=-1)
        gamma = self.interaction_net(joint)
        return gamma


def _scatter_op(op_name: str, src: torch.Tensor, index: torch.Tensor,
                dim: int = 0, dim_size: int | None = None) -> torch.Tensor:
    """Scatter helper: sum or max. Works for 1D and 2D inputs."""
    if dim_size is None:
        dim_size = int(index.max().item()) + 1
    if src.dim() == 1:
        # 1D case: both src and index are 1D
        out = src.new_zeros(dim_size)
        if op_name == "sum":
            out.scatter_add_(0, index, src)
        elif op_name == "max":
            out.scatter_reduce_(0, index, src, reduce="amax", include_self=False)
    else:
        shape = list(src.shape)
        shape[dim] = dim_size
        out = src.new_zeros(shape)
        idx_expanded = index.unsqueeze(-1).expand_as(src)
        if op_name == "sum":
            out.scatter_add_(dim, idx_expanded, src)
        elif op_name == "max":
            out.scatter_reduce_(dim, idx_expanded, src, reduce="amax", include_self=False)
    return out


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = 0) -> torch.Tensor:
    return _scatter_op("sum", src, index, dim)


def scatter_max(src: torch.Tensor, index: torch.Tensor, dim: int = 0) -> torch.Tensor:
    return _scatter_op("max", src, index, dim)


# ---------------------------------------------------------------------------
# Main CANDELA model
# ---------------------------------------------------------------------------

class CANDELA(nn.Module):
    """CANDELA: Cancer Drug Sensitivity Estimation modular Graph Neural Network.

    y_hat = alpha_i + beta_j + gamma_ij
      alpha_i: drug cytotoxicity (drug module)
      beta_j:  cell-line survivability (expression module)
      gamma_ij: specific drug-cell interaction (interaction module)
    """

    def __init__(
        self,
        cell_dim: int,
        drug_graphs: list[Data],
        cell_table: torch.Tensor,
        node_embed_dim: int = 128,
        gat_heads: int = 8,
        drug_pool_hidden: int = 1024,
        drug_score_hidden: int = 1152,
        expr_hidden: int = 1024,
        expr_bottleneck: int = 128,
        expr_score_hidden: int = 660,
        interaction_hidden: int = 1212,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.drug_encoder = DrugEncoder(
            in_dim=ATOM_DIM,
            node_embed_dim=node_embed_dim,
            heads=gat_heads,
            dropout=dropout,
        )
        self.drug_module = DrugModule(
            node_embed_dim=node_embed_dim,
            pool_hidden=drug_pool_hidden,
            score_hidden=drug_score_hidden,
        )
        self.expr_encoder = ExpressionEncoder(
            in_dim=cell_dim,
            hidden=expr_hidden,
            bottleneck=expr_bottleneck,
        )
        self.expr_module = ExpressionModule(
            bottleneck=expr_bottleneck,
            score_hidden=expr_score_hidden,
        )
        self.interaction_module = InteractionModule(
            node_embed_dim=node_embed_dim,
            cell_embed_dim=expr_bottleneck,
            hidden=interaction_hidden,
            dropout=dropout,
        )

        self.register_buffer("cell_table", cell_table)
        self.drug_graphs = drug_graphs
        self.node_embed_dim = node_embed_dim

    @classmethod
    def from_config(
        cls, cell_dim: int, cfg: DictConfig,
        drug_graphs: list[Data], cell_table: torch.Tensor,
    ) -> "CANDELA":
        m = cfg.model
        return cls(
            cell_dim=cell_dim,
            drug_graphs=drug_graphs,
            cell_table=cell_table,
            node_embed_dim=m.get("node_embed_dim", 128),
            gat_heads=m.get("gat_heads", 8),
            drug_pool_hidden=m.get("drug_pool_hidden", 1024),
            drug_score_hidden=m.get("drug_score_hidden", 1152),
            expr_hidden=m.get("expr_hidden", 1024),
            expr_bottleneck=m.get("expr_bottleneck", 128),
            expr_score_hidden=m.get("expr_score_hidden", 660),
            interaction_hidden=m.get("interaction_hidden", 1212),
            dropout=m.get("dropout", 0.3),
        )

    def forward(self, batch: dict) -> dict:
        drug_ids = batch["drug_ids"]
        cell_ids = batch["cell_ids"]
        device = drug_ids.device

        # --- Drug branch ---
        graph_list = [self.drug_graphs[int(i)] for i in drug_ids]
        batched_graph = Batch.from_data_list(graph_list).to(device)

        node_emb = self.drug_encoder(batched_graph.x, batched_graph.edge_index)
        alpha, drug_emb = self.drug_module(node_emb, batched_graph.batch)

        # --- Cell branch ---
        cidx = cell_ids.to(self.cell_table.device)
        cell_input = self.cell_table[cidx].to(device)
        cell_emb = self.expr_encoder(cell_input)  # (batch_size, bottleneck)
        beta = self.expr_module(cell_emb)

        # --- Interaction branch ---
        batch_size = len(drug_ids)
        gamma = self.interaction_module(node_emb, batched_graph.batch, cell_emb, batch_size)

        # --- Final prediction ---
        pred = alpha + beta + gamma  # (batch_size, 1)

        return {
            "prediction": pred,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
        }
