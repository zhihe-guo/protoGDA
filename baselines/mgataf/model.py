"""MGATAF: Multi-channel Graph Attention Network with Adaptive Fusion.

Faithful reimplementation of Saeed et al. (2025),
BMC Bioinformatics 26:19. https://doi.org/10.1186/s12859-024-05987-0

Three modules (per paper):
  1. GNN-based node representation  (Eq.1)
     Multi-layer GCN with ReLU, retains node embeddings from every layer.
     Xavier init, L2 weight decay 5e-4.

  2. Multi-channel graph attention  (Eq.2-7)
     Virtual super-node c_s; GAT per layer (8 heads first, 1 rest);
     GRU updates c_s hidden state; layer attention weights graph embedding.
     LeakyReLU negative_slope=0.2.

  3. Adaptive fusion  (Eq.8-19)
     Cell encoder: Conv1d → Flatten → FC + ReLU  (Eq.8-9)
     Fingerprint encoder:  Conv1d → Flatten → FC + ReLU  (Eq.10-11)
     Concatenation (Eq.12), sigmoid control gates (Eq.13-14),
     feature-dimension attention (Eq.15-17), single FC → IC50 (Eq.18-19).

Training (per paper, now faithfully followed):
  Adam, lr=5e-4, dropout=0.3, early stopping patience=50, max 300 epochs,
  weight_decay=5e-4, IC50 normalized to [0,1].

This implementation uses 78-dim atom features faithfully matching the original
paper Table 1 (atom type 44 + degree 11 + implicit valence 7 + formal charge 1
+ radical electrons 1 + hybridization 5 + aromatic 1 + hydrogens 5 + ring 1
+ chirality 2).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATv2Conv, GCNConv, global_mean_pool

# Paper-specified 78-dim atom features (RDKit standard)
ATOM_DIM = 78


# ---------------------------------------------------------------------------
# Module 1: GNN-based node representation
# ---------------------------------------------------------------------------

class GNNNodeRepr(nn.Module):
    """Multi-layer GCN with ReLU. Returns node embeddings from every layer.

    Per paper: Z^(0) = X (raw atom features); Z^(l) = ReLU(tilde_A Z^(l-1) W).
    All layers' outputs are retained for the multi-channel attention module.
    Xavier init on weights.
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.dropout = dropout
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, edge_index):
        outs = [x]
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            outs.append(h)
        return outs


# ---------------------------------------------------------------------------
# Module 2: Multi-channel graph attention
# ---------------------------------------------------------------------------

class MultiChannelGAT(nn.Module):
    """Multi-channel GAT with virtual super-node and GRU."""

    def __init__(
        self,
        num_channels: int,
        in_dims: list[int],
        hidden_dim: int,
        heads_first: int = 8,
        heads_rest: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim

        self.gats = nn.ModuleList()
        for i in range(num_channels):
            heads = heads_first if i == 0 else heads_rest
            out_per_head = hidden_dim // heads
            self.gats.append(
                GATv2Conv(
                    in_dims[i],
                    out_per_head,
                    heads=heads,
                    negative_slope=0.2,
                    dropout=dropout,
                    concat=True,
                )
            )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.layer_attn = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, layer_outs: list[torch.Tensor], edge_index, batch) -> torch.Tensor:
        device = layer_outs[0].device
        B = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        h_super = torch.zeros(B, self.hidden_dim, device=device)

        layer_embs = []
        for i, z in enumerate(layer_outs):
            z_new = self.gats[i](z, edge_index)
            z_new = F.elu(z_new)
            g = global_mean_pool(z_new, batch)
            h_super = self.gru(g, h_super)
            layer_embs.append(h_super)

        stacked = torch.stack(layer_embs, dim=1)
        scores = self.layer_attn(torch.tanh(stacked)).squeeze(-1)
        weights = F.softmax(scores, dim=-1)
        graph_emb = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        return graph_emb


# ---------------------------------------------------------------------------
# Module 3a: Cancer cell line embedding (conv + FC, per paper)
# ---------------------------------------------------------------------------

class CellLineEncoder(nn.Module):
    """Conv + FC encoder for cancer cell line features.

    Per paper Eq.8-9: single Conv1d layer followed by Flatten then a
    fully connected layer with ReLU.  The FC operates on the flattened
    feature map, preserving positional information of each genomic locus.

    LayerNorm is added after flatten to stabilize the large FC input
    (32 × in_dim can be > 20K dims for genomic features) in a batch-independent way.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.conv = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.ln = nn.LayerNorm(32 * in_dim)
        self.fc = nn.Linear(32 * in_dim, out_dim)
        self.dropout = dropout

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv(x))
        x = x.flatten(1)
        x = self.ln(x)
        x = F.relu(self.fc(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# ---------------------------------------------------------------------------
# Module 3b: Fingerprint encoding (conv + FC, per paper)
# ---------------------------------------------------------------------------

class FingerprintEncoder(nn.Module):
    """Conv + FC encoder for drug fingerprints, mirroring the CCL encoder.

    LayerNorm is added after flatten to stabilize the large FC input
    (32 × in_dim = 32,768 for 1024-bit fingerprints) in a batch-independent way.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.conv = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.ln = nn.LayerNorm(32 * in_dim)
        self.fc = nn.Linear(32 * in_dim, out_dim)
        self.dropout = dropout

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv(x))
        x = x.flatten(1)
        x = self.ln(x)
        x = F.relu(self.fc(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# ---------------------------------------------------------------------------
# Module 3c: Adaptive fusion (gating + GAT attention + FC)
# ---------------------------------------------------------------------------

class GATAttention(nn.Module):
    """GAT-based self-attention over the modalities (Eq. 15-17)."""

    def __init__(self, in_dim: int, out_dim: int, negative_slope: float = 0.2):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Parameter(torch.zeros(2 * out_dim, 1))
        self.negative_slope = negative_slope
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, num_nodes, in_dim)
        B, N, D = h.shape
        Wh = self.W(h)  # (B, N, out_dim)

        # Compute attention scores e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
        Wh_i = Wh.unsqueeze(2).repeat(1, 1, N, 1)  # (B, N, N, out_dim)
        Wh_j = Wh.unsqueeze(1).repeat(1, N, 1, 1)  # (B, N, N, out_dim)
        Wh_concat = torch.cat([Wh_i, Wh_j], dim=-1)  # (B, N, N, 2 * out_dim)

        e = torch.matmul(Wh_concat, self.a).squeeze(-1)  # (B, N, N)
        e = F.leaky_relu(e, negative_slope=self.negative_slope)

        alpha = F.softmax(e, dim=-1)  # (B, N, N)

        # h_i' = \sigma(\sum \alpha_ij Wh_j)
        h_prime = torch.matmul(alpha, Wh)  # (B, N, out_dim)
        h_prime = F.relu(h_prime)
        return h_prime


class AdaptiveFusion(nn.Module):
    """Adaptive fusion module with gating and GAT attention.

    Per paper: sigmoid control gates (Eq. 13-14) → GAT attention (Eq. 15-17) → Flatten (Eq. 18) → single FC layer (Eq. 19).
    """

    def __init__(self, drug_dim: int, cell_dim: int, fp_dim: int, out_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.proj_drug = nn.Linear(drug_dim, out_dim)
        self.proj_cell = nn.Linear(cell_dim, out_dim)
        self.proj_fp = nn.Linear(fp_dim, out_dim)

        # Gating mechanism (Eq. 13-14)
        self.gate_layer = nn.Linear(3 * out_dim, 3)

        # GAT attention (Eq. 15-17)
        self.gat_attn = GATAttention(in_dim=out_dim, out_dim=out_dim)

        # Final fully connected layer (Eq. 18-19)
        self.fc = nn.Linear(3 * out_dim, 1)
        self.dropout = dropout

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, h_drug: torch.Tensor, h_cell: torch.Tensor, h_fp: torch.Tensor) -> torch.Tensor:
        # Project to same dimension
        h_drug_proj = self.proj_drug(h_drug)
        h_cell_proj = self.proj_cell(h_cell)
        h_fp_proj = self.proj_fp(h_fp)

        # Gating mechanism (Eq. 13-14)
        concat_feats = torch.cat([h_drug_proj, h_cell_proj, h_fp_proj], dim=-1)
        gates = torch.sigmoid(self.gate_layer(concat_feats))

        h_drug_gated = h_drug_proj * gates[:, 0:1]
        h_cell_gated = h_cell_proj * gates[:, 1:2]
        h_fp_gated = h_fp_proj * gates[:, 2:3]

        # Stack as 3 nodes
        h_nodes = torch.stack([h_drug_gated, h_cell_gated, h_fp_gated], dim=1)  # (B, 3, out_dim)

        # GAT attention (Eq. 15-17)
        h_prime = self.gat_attn(h_nodes)  # (B, 3, out_dim)

        # Flatten (Eq. 18)
        h_flat = h_prime.flatten(1)  # (B, 3 * out_dim)

        # Dropout and final fully connected layer (Eq. 19)
        h_flat = F.dropout(h_flat, p=self.dropout, training=self.training)
        out = self.fc(h_flat)
        return out


# ---------------------------------------------------------------------------
# Main MGATAF model
# ---------------------------------------------------------------------------

class MGATAF(nn.Module):
    """MGATAF: Multi-channel Graph Attention Network with Adaptive Fusion."""

    def __init__(
        self,
        cell_dim: int,
        drug_graphs: list[Data],
        cell_table: torch.Tensor,
        drug_fp_table: torch.Tensor,
        gnn_hidden_dim: int = 64,
        num_gnn_layers: int = 3,
        gat_heads_first: int = 8,
        gat_heads_rest: int = 1,
        cell_output_dim: int = 64,
        fp_output_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Module 1: GNN node representation
        self.gnn_node = GNNNodeRepr(
            in_dim=ATOM_DIM,
            hidden_dim=gnn_hidden_dim,
            num_layers=num_gnn_layers,
            dropout=dropout,
        )

        # Module 2: Multi-channel graph attention
        num_channels = num_gnn_layers + 1
        in_dims = [ATOM_DIM] + [gnn_hidden_dim] * num_gnn_layers
        self.multi_channel_gat = MultiChannelGAT(
            num_channels=num_channels,
            in_dims=in_dims,
            hidden_dim=gnn_hidden_dim,
            heads_first=gat_heads_first,
            heads_rest=gat_heads_rest,
            dropout=dropout,
        )

        # Module 3: encoders + adaptive fusion
        self.cell_encoder = CellLineEncoder(cell_dim, cell_output_dim, dropout=dropout)
        self.fp_encoder = FingerprintEncoder(
            drug_fp_table.shape[1] if drug_fp_table is not None else 1024,
            fp_output_dim,
            dropout=dropout,
        )
        self.adaptive_fusion = AdaptiveFusion(
            drug_dim=gnn_hidden_dim,
            cell_dim=cell_output_dim,
            fp_dim=fp_output_dim,
            out_dim=64,
            dropout=dropout,
        )

        self.register_buffer("cell_table", cell_table)
        self.drug_graphs = drug_graphs
        self.register_buffer("drug_fp_table", drug_fp_table)

    @classmethod
    def from_config(cls, cell_dim, cfg, drug_graphs, cell_table, drug_fp_table):
        m = cfg.model
        return cls(
            cell_dim=cell_dim,
            drug_graphs=drug_graphs,
            cell_table=cell_table,
            drug_fp_table=drug_fp_table,
            gnn_hidden_dim=m.get("gnn_hidden_dim", 64),
            num_gnn_layers=m.get("num_gnn_layers", 3),
            gat_heads_first=m.get("gat_heads_first", 8),
            gat_heads_rest=m.get("gat_heads_rest", 1),
            cell_output_dim=m.get("cell_output_dim", 64),
            fp_output_dim=m.get("fp_output_dim", 64),
            dropout=m.get("dropout", 0.3),
        )

    def get_branch_params(self) -> dict[str, list[nn.Parameter]]:
        return {
            "drug": list(self.gnn_node.parameters()) + list(self.multi_channel_gat.parameters()),
            "cell": list(self.cell_encoder.parameters()),
            "shared": list(self.fp_encoder.parameters()) + list(self.adaptive_fusion.parameters()),
        }

    def get_branch_grad_norms(self) -> dict[str, float]:
        norms: dict[str, float] = {}
        key_map = {"cell": "cell", "drug": "drug_gnn", "shared": "shared"}
        for branch, params in self.get_branch_params().items():
            target = key_map[branch]
            total = 0.0
            rms_sq = 0.0
            count = 0
            for p in params:
                if p.grad is not None:
                    total += p.grad.norm(2).item() ** 2
                    rms_sq += (p.grad ** 2).sum().item()
                    count += p.grad.numel()
            norms[target] = float(total ** 0.5) if total > 0 else 0.0
            norms[f"{target}_rms"] = float((rms_sq / count) ** 0.5) if count > 0 else 0.0
        if "drug_gnn_rms" in norms:
            norms["drug_rms"] = norms["drug_gnn_rms"]
        return norms

    def forward(self, batch):
        drug_ids = batch["drug_ids"]
        cell_ids = batch["cell_ids"]
        device = drug_ids.device

        graph_list = [self.drug_graphs[int(i)] for i in drug_ids]
        bg = Batch.from_data_list(graph_list).to(device)
        layer_outs = self.gnn_node(bg.x, bg.edge_index)
        h_drug = self.multi_channel_gat(layer_outs, bg.edge_index, bg.batch)

        cidx = cell_ids.to(self.cell_table.device)
        cell_input = self.cell_table[cidx].to(device)
        h_cell = self.cell_encoder(cell_input)

        fp_input = self.drug_fp_table[drug_ids].to(device)
        h_fp = self.fp_encoder(fp_input)

        pred = self.adaptive_fusion(h_drug, h_cell, h_fp)
        return {"prediction": pred}
