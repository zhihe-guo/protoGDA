"""Cross-attention, GNN evolvers, and memory accumulator layers."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GATv2Conv, GCNConv, GINEConv, EGConv
from torch_geometric.utils import scatter


class GraphCrossAttention(nn.Module):
    """
    One-way cross-attention: cell query Q probes graph nodes (K, V).

    Q: (B, M, D), graph nodes batched -> output (B, M, D) per sample.

    Uses padded batched attention (no per-sample loop) for GPU efficiency.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _batch_padded_nodes(
        node_feats: torch.Tensor,
        batch_index: torch.Tensor,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad node features into (B, N_max, D) with key_padding_mask (B, N_max)."""
        device = node_feats.device
        D = node_feats.size(-1)

        # Count nodes per sample
        node_counts = torch.bincount(batch_index, minlength=batch_size)  # (B,)
        N_max = int(node_counts.max().item())
        if N_max == 0:
            return node_feats.new_zeros(batch_size, 0, D), node_feats.new_ones(batch_size, 0, dtype=torch.bool)

        # Compute scatter index: within each sample, local index 0..n_i-1
        arange = torch.arange(node_feats.size(0), device=device)
        local_idx = arange - torch.cat(
            [torch.zeros(1, device=device, dtype=torch.long),
             torch.cumsum(node_counts, dim=0)[:-1]]
        )[batch_index]

        # Scatter into padded tensor
        padded = node_feats.new_zeros(batch_size, N_max, D)
        padded[batch_index, local_idx] = node_feats

        # Key padding mask: True where padded
        key_padding_mask = node_feats.new_ones(batch_size, N_max, dtype=torch.bool)
        for i in range(batch_size):
            n = int(node_counts[i].item())
            key_padding_mask[i, :n] = False

        return padded, key_padding_mask

    def forward(
        self,
        query: torch.Tensor,
        node_feats: torch.Tensor,
        batch_index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            query: (B, M, D)
            node_feats: either (N_total, D) with batch_index (graph nodes)
                        or (B, N, D) already shaped (fixed-count tokens, e.g. cell tokens)
            batch_index: (N_total,) graph id per node, or None if node_feats is (B,N,D)
        Returns:
            extracted: (B, M, D)
            attn_weights: list of (M, V_i) per batch element
        """
        b, m, d = query.shape
        nh = self.num_heads
        hd = self.head_dim

        q = self.q_proj(query)                                # (B, M, D)

        # Fast path: node_feats already (B, N, D) — fixed token count, no padding
        if node_feats.ndim == 3:
            flat_nodes = node_feats.reshape(-1, d)            # (B*N, D)
            k = self.k_proj(flat_nodes)                       # (B*N, D)
            v = self.v_proj(flat_nodes)                       # (B*N, D)
        else:
            k = self.k_proj(node_feats)                       # (N_total, D)
            v = self.v_proj(node_feats)                       # (N_total, D)

        if node_feats.ndim == 3 or batch_index is None:
            # Fixed token count — no padding needed
            N = node_feats.size(1)
            k_h = k.view(b, N, nh, hd).transpose(1, 2)       # (B, nh, N, hd)
            v_h = v.view(b, N, nh, hd).transpose(1, 2)       # (B, nh, N, hd)
            key_mask = query.new_zeros(b, 1, 1, N, dtype=torch.bool)
        else:
            # Pad nodes into dense batched tensors
            k_padded, key_mask = self._batch_padded_nodes(k, batch_index, b)
            v_padded, _ = self._batch_padded_nodes(v, batch_index, b)
            k_h = k_padded.view(b, -1, nh, hd).transpose(1, 2)
            v_h = v_padded.view(b, -1, nh, hd).transpose(1, 2)
            key_mask = key_mask.unsqueeze(1).unsqueeze(2)    # (B, 1, 1, Np)

        q_h = q.view(b, m, nh, hd).transpose(1, 2)    # (B, nh, M, hd)

        # Scaled dot-product attention with key padding mask
        scale = hd ** -0.5
        attn_logits = torch.matmul(q_h, k_h.transpose(-2, -1)) * scale  # (B, nh, M, N)

        attn_logits = attn_logits.masked_fill(key_mask, float("-inf"))

        attn = F.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn)

        out_h = torch.matmul(attn, v_h)  # (B, nh, M, hd)
        out_h = out_h.transpose(1, 2).reshape(b, m, d)  # (B, M, D)
        extracted = self.out_proj(out_h)
        extracted = self.norm(extracted)

        # Build per-sample attention weights
        attn_mean = attn.mean(dim=1)  # (B, M, N)
        attn_weights_list: list[torch.Tensor] = []
        for i in range(b):
            attn_weights_list.append(attn_mean[i].detach())  # (M, N)

        return extracted, attn_weights_list


class MemoryAccumulator(nn.Module):
    """
    Gated residual accumulator with Pre-LN and FFN.

    For each layer's extraction, applies:
      1. Pre-LN on current memory
      2. FFN on (normed_memory + extraction)
      3. Sigmoid gate to control information flow
      4. memory = memory + gate * ffn_out
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def init_memory(self, query: torch.Tensor, mode: str = "cell_probe") -> torch.Tensor:
        """Initialize M_accum.

        Args:
            query: (B, M, D) cell probe or drug probe
            mode: "cell_probe"       = clone query as initial memory (original behaviour)
                  "zero"             = start from zero tensor
                  "cell_probe_gated" = start from zero, but caller should immediately
                                       run accumulator(memory, query) to gate-write
                                       the probe info into memory
        """
        if mode in ("zero", "cell_probe_gated"):
            return torch.zeros_like(query)
        return query.clone()

    def forward(
        self,
        memory: torch.Tensor,
        extraction: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            memory: (B, M, D)
            extraction: (B, M, D)
        """
        x = self.norm(memory)
        h = self.ffn(x + extraction)
        g = self.gate(h)
        return memory + g * h


class CellTowerBlock(nn.Module):
    """Intra-tower update for cell probes Q between interaction layers.

    Pre-LN residual block: optional probe self-attention + FFN. Applied to the
    probe query Q (B, M, D) at each layer so the cell-side tower keeps training
    across layers instead of being a static one-shot projection.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_self_attn: bool = True,
    ):
        super().__init__()
        self.use_self_attn = use_self_attn
        if use_self_attn:
            assert hidden_dim % num_heads == 0
            self.attn_norm = nn.LayerNorm(hidden_dim)
            self.self_attn = nn.MultiheadAttention(
                hidden_dim, num_heads, dropout=dropout, batch_first=True
            )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """q: (B, M, D) → (B, M, D)"""
        if self.use_self_attn:
            h = self.attn_norm(q)
            h, _ = self.self_attn(h, h, h)
            q = q + h
        q = q + self.ffn(self.ffn_norm(q))
        return q


class EGNNLayer(nn.Module):
    """E(n) Equivariant Graph Neural Network layer (Satorras et al., 2021).

    Operates on node features (h) and 3D coordinates (x) jointly:
      - Edge message:  m_ij = φ_e(h_i, h_j, ||x_i-x_j||², a_ij)
      - Coordinate update (equivariant):
          x_i' = x_i + C * Σ_j (x_i-x_j) * φ_x(m_ij)
      - Feature update (invariant):
          h_i' = φ_h(h_i, Σ_j m_ij)

    Only h is fed downstream to cross-attention; x stays inside the GNN.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int | None = None,
        dropout: float = 0.1,
        rbf_type: str = "saturated",
        rbf_centers: list[float] | None = None,
        rbf_sigma: float = 0.5,
        use_attention: bool = False,
        attn_heads: int = 1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rbf_type = rbf_type
        self.use_attention = use_attention
        self.attn_heads = attn_heads

        if rbf_type == "gaussian":
            # Fixed Gaussian RBF basis over distance: exp(-(d-c_k)^2 / 2*sigma^2).
            # Bond lengths ~1.0-3.5 A; a multi-center basis lets the network
            # resolve subtle differences (e.g. aromatic vs single bond) instead
            # of saturating to ~constant like 1-exp(-d^2).
            centers = rbf_centers if rbf_centers else [1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.5, 4.0]
            self.register_buffer("rbf_centers", torch.tensor(centers, dtype=torch.float))
            self.rbf_sigma = rbf_sigma
            self.rbf_n = len(centers)
            dist_dim = self.rbf_n
        else:
            self.rbf_n = 1
            dist_dim = 1

        edge_in = 2 * hidden_dim + dist_dim  # h_i, h_j, dist encoding(s)
        if edge_dim is not None:
            edge_in += edge_dim

        # Edge MLP: φ_e
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Coordinate MLP: φ_x
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Node feature MLP: φ_h
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        if self.use_attention:
            # Attention logit from rotation-invariant edge message (E, D) -> heads.
            self.attn_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, attn_heads),
            )

        self.norm_h = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: node features (N, D)
            pos: 3D atom coordinates (N, 3)
            edge_index: (2, E) int64
            edge_attr: optional edge features (E, D_edge)
        Returns:
            h': updated node features (N, D)
            pos': updated coordinates (N, 3)
        """
        row, col = edge_index
        # -- distance encoding --------------------------------------------------
        d2 = torch.sum((pos[row] - pos[col]) ** 2, dim=-1, keepdim=True)  # (E, 1)
        if self.rbf_type == "gaussian":
            # Multi-center Gaussian basis: each column is exp(-(d-c_k)^2 / 2*sigma^2)
            d = torch.sqrt(d2.clamp(min=1e-8))  # (E, 1)
            rbf = torch.exp(-((d - self.rbf_centers.view(1, -1)) ** 2)
                            / (2.0 * self.rbf_sigma ** 2))  # (E, K)
        else:
            rbf = 1.0 - torch.exp(-d2)  # squash to ~[0, 1), (E, 1)
        m_inputs = [h[row], h[col], rbf]
        if edge_attr is not None:
            m_inputs.append(edge_attr)
        m = self.edge_mlp(torch.cat(m_inputs, dim=-1))  # (E, D)

        # -- coordinate update (equivariant) -------------------------------
        coord_weight = self.coord_mlp(m)  # (E, 1)
        delta_pos = (pos[row] - pos[col]) * coord_weight  # (E, 3)

        # -- attention weights (rotation-invariant, per-neighbor softmax) ----
        if self.use_attention:
            logits = self.attn_mlp(m)  # (E, H)
            # softmax over neighbors of the same source node (grouped by row)
            logits_max = scatter(logits, row, dim=0, dim_size=h.size(0), reduce="max")[row]
            logits = logits - logits_max  # numeric stability
            logits_exp = logits.exp()  # (E, H)
            norm = scatter(logits_exp, row, dim=0, dim_size=h.size(0), reduce="sum")[row]  # (E, H)
            alpha = logits_exp / (norm + 1e-8)  # (E, H)  attention coefficients

            # split message into H heads for attention-weighted aggregation
            head_dim = m.size(-1) // self.attn_heads
            m_h = m.view(-1, self.attn_heads, head_dim)  # (E, H, D/H)
            agg_h = alpha.unsqueeze(-1) * m_h  # (E, H, D/H)
            agg = scatter(agg_h, row, dim=0, dim_size=h.size(0), reduce="sum")  # (N, H, D/H)
            agg = agg.reshape(h.size(0), -1)  # (N, D)

            # coordinate update also attends to the most relevant neighbors
            coord_alpha = alpha.mean(dim=-1, keepdim=True)  # (E, 1) scalar attention
            delta_pos = delta_pos * coord_alpha
            pos = pos + scatter(delta_pos, row, dim=0, dim_size=h.size(0),
                                reduce="sum") / (h.size(0) ** 0.5 + 1e-6)
        else:
            agg = scatter(m, row, dim=0, dim_size=h.size(0), reduce="mean")  # (N, D)
            pos = pos + scatter(delta_pos, row, dim=0, dim_size=h.size(0),
                                reduce="mean") / (h.size(0) ** 0.5 + 1e-6)

        # -- node feature update (invariant) -------------------------------
        h_new = self.node_mlp(torch.cat([h, agg], dim=-1))  # (N, D)
        h_new = self.norm_h(h_new)
        if h.size(-1) == h_new.size(-1):
            h_new = h_new + h  # residual
        return h_new, pos


class GraphEvolver(nn.Module):
    """Single GNN layer for molecular graph evolution."""

    def __init__(
        self,
        hidden_dim: int,
        gnn_type: str = "gat",
        heads: int = 4,
        dropout: float = 0.1,
        edge_dim: int | None = None,
        rbf_type: str = "saturated",
        rbf_centers: list[float] | None = None,
        rbf_sigma: float = 0.5,
        use_attention: bool = False,
        attn_heads: int = 1,
        gat_edge_attr: bool = False,
    ):
        super().__init__()
        self.gnn_type = gnn_type.lower()
        self.dropout = dropout
        self.gat_edge_attr = gat_edge_attr

        if self.gnn_type == "gat":
            # GATv2Conv consumes raw edge features (bond type) when edge_dim is set.
            # This gives the attention a "bonding context" so atom-level features
            # like pharmacophore covariance can be read meaningfully.
            self.conv = GATv2Conv(
                hidden_dim,
                hidden_dim // heads,
                heads=heads,
                dropout=dropout,
                concat=True,
                edge_dim=edge_dim if gat_edge_attr and edge_dim is not None else None,
            )
            self.out_dim = hidden_dim
        elif self.gnn_type == "gcn":
            self.conv = GCNConv(hidden_dim, hidden_dim)
            self.out_dim = hidden_dim
        elif self.gnn_type == "gine":
            if edge_dim is None:
                edge_dim = 6
            mlp = nn.Sequential(
                nn.Linear(edge_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.conv = GINEConv(mlp, edge_dim=edge_dim)
            self.out_dim = hidden_dim
        elif self.gnn_type == "egconv":
            self.conv = EGConv(
                hidden_dim,
                hidden_dim,
                aggregators=["symnorm", "mean", "sum"],
                num_heads=heads,
                num_bases=4,
            )
            self.out_dim = hidden_dim
        elif self.gnn_type == "egnn":
            self.conv = EGNNLayer(
                hidden_dim, edge_dim=edge_dim, dropout=dropout,
                rbf_type=rbf_type, rbf_centers=rbf_centers, rbf_sigma=rbf_sigma,
                use_attention=use_attention, attn_heads=attn_heads,
            )
            self.out_dim = hidden_dim
        else:
            raise ValueError(f"Unknown gnn_type: {gnn_type}")

        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.gnn_type == "egnn":
            if pos is None:
                raise RuntimeError(
                    "EGNN requires 3D atom positions (data.pos). "
                    "Set with_conformers=True when building graphs."
                )
            h, pos = self.conv(x, pos, edge_index, edge_attr)
            h = self.norm(h)
            h = self.act(h)
            h = self.drop(h)
            if h.size(-1) == x.size(-1):
                h = h + x
            return h, pos
        elif self.gnn_type == "gine" and edge_attr is not None:
            h = self.conv(x, edge_index, edge_attr)
        elif self.gnn_type == "gat" and edge_attr is not None:
            h = self.conv(x, edge_index, edge_attr)
        else:
            h = self.conv(x, edge_index)
        h = self.norm(h)
        h = self.act(h)
        h = self.drop(h)
        if h.size(-1) == x.size(-1):
            h = h + x
        return h


class InteractionBlock(nn.Module):
    """One stage: GNN evolve first, then cross-attention extract.

    Order: H = GNN(H) → LN(H) → CrossAttn(Q, H) → extracted
    This lets the probe query attend to the *updated* graph at each layer.
    """

    def __init__(
        self,
        hidden_dim: int,
        gnn_type: str,
        gnn_heads: int,
        cross_attn_heads: int,
        dropout: float,
        edge_dim: int | None,
        rbf_type: str = "saturated",
        rbf_centers: list[float] | None = None,
        rbf_sigma: float = 0.5,
        use_equiv_feats: bool = False,
        equiv_feat_dim: int = 8,
        use_attention: bool = False,
        attn_heads: int = 1,
        gat_edge_attr: bool = False,
    ):
        super().__init__()
        self.gnn = GraphEvolver(
            hidden_dim,
            gnn_type=gnn_type,
            heads=gnn_heads,
            dropout=dropout,
            edge_dim=edge_dim,
            rbf_type=rbf_type,
            rbf_centers=rbf_centers,
            rbf_sigma=rbf_sigma,
            use_attention=use_attention,
            attn_heads=attn_heads,
            gat_edge_attr=gat_edge_attr,
        )
        self.use_equiv_feats = use_equiv_feats and gnn_type.lower() == "egnn"
        if self.use_equiv_feats:
            # Lifts the 2 raw rotation-invariant descriptors (centroid distance,
            # layer displacement) to equiv_feat_dim, then projects to hidden_dim
            # via a residual add (downstream dim unchanged).
            self.equiv_proj = nn.Sequential(
                nn.Linear(2, equiv_feat_dim),
                nn.GELU(),
                nn.Linear(equiv_feat_dim, hidden_dim),
            )
        self.gnn_norm = nn.LayerNorm(hidden_dim)  # Post-GNN LN (#5)
        self.cross_attn = GraphCrossAttention(
            hidden_dim, num_heads=cross_attn_heads, dropout=dropout
        )
        self.extract_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        query: torch.Tensor,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        batch_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], torch.Tensor | None]:
        pos_in = pos  # coords before this layer's EGNN evolution
        if self.gnn.gnn_type == "egnn":
            evolved, pos = self.gnn(node_feats, edge_index, edge_attr, pos=pos)
        else:
            evolved = self.gnn(node_feats, edge_index, edge_attr)
        evolved = self.gnn_norm(evolved)                      # LN after GNN

        # Rotation-invariant descriptors from evolved coords (EGNN only).
        # Keeps the cross-attention input dim unchanged via residual add.
        if self.use_equiv_feats and pos is not None and pos_in is not None:
            # (1) distance from each atom to its sample's centroid
            centroid = scatter(pos, batch_index, dim=0, reduce="mean")  # (N_samples, 3)
            dist_centroid = torch.norm(pos - centroid[batch_index], dim=-1, keepdim=True)  # (N,1)
            # (2) per-atom displacement norm between input and evolved coords
            disp = torch.norm(pos - pos_in, dim=-1, keepdim=True)  # (N,1)
            equiv = torch.cat([dist_centroid, disp], dim=-1)  # (N, 2)
            evolved = evolved + self.equiv_proj(equiv)

        extracted, attn = self.cross_attn(query, evolved, batch_index)
        extracted = self.extract_norm(extracted)
        return extracted, evolved, attn, pos


class ReversedInteractionBlock(nn.Module):
    """One stage for reversed cross-attention (drug probes → cell tokens).

    Order: Cell_tokens = Cell_tokens + MLP(LN(Cell_tokens)) → CrossAttn(Q, Cell_tokens) → extracted

    Cell tokens have fixed count per sample (no variable-node graph),
    so GraphCrossAttention receives already-shaped (B, N_cell, D) tensors.
    """

    def __init__(self, hidden_dim: int, cross_attn_heads: int, dropout: float):
        super().__init__()
        self.cell_update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cell_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = GraphCrossAttention(
            hidden_dim, num_heads=cross_attn_heads, dropout=dropout
        )
        self.extract_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        query_drug: torch.Tensor,     # (B, M_drug, D)
        cell_tokens: torch.Tensor,    # (B, N_cell, D) — fixed count, no batch_index needed
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        # 1. MLP evolution of cell tokens (Pre-LN residual)
        cell_tokens = cell_tokens + self.cell_update(self.cell_norm(cell_tokens))
        # 2. Cross-attention: drug probes → cell tokens (3D fast path)
        extracted, attn = self.cross_attn(query_drug, cell_tokens, batch_index=None)
        extracted = self.extract_norm(extracted)
        return extracted, cell_tokens, attn
