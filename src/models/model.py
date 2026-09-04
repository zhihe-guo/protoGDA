"""Top-level CellDrugModel.

Supports two cross-attention directions:
  cell_to_drug  – cell-line probes query drug graph nodes (original)
  drug_to_cell  – drug probes query cell-line feature tokens (reversed)

Triple-stream drug representation (GNN + Morgan + ChemBERTa) fused before
the readout MLP in both directions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch_geometric.data import Batch, Data
from torch_geometric.utils import scatter

from src.data.preprocessing import get_atom_feature_dim, get_bond_feature_dim
from src.models.cellquery import CellQueryProjector
from src.models.layers import (
    CellTowerBlock,
    InteractionBlock,
    MemoryAccumulator,
    ReversedInteractionBlock,
)


class CellDrugModel(nn.Module):
    """Multi-stage cross-attention drug-cell interaction model.

    direction='cell_to_drug': cell probes → drug graph K/V
    direction='drug_to_cell': drug probes → cell tokens K/V
    """

    def __init__(
        self,
        cell_dim: int,
        cfg: DictConfig,
        drug_graphs: list[Data] | None = None,
        cell_features_table: torch.Tensor | None = None,
        drug_morgan_table: torch.Tensor | None = None,
        drug_chemberta_table: torch.Tensor | None = None,
        atom_dim: int | None = None,
        edge_dim: int | None = None,
    ):
        super().__init__()
        mcfg = cfg.model
        self.num_probes = mcfg.num_probes          # for cell_to_drug
        self.hidden_dim = mcfg.hidden_dim
        self.memory_init = mcfg.get("memory_init", "cell_probe")  # "cell_probe" | "zero"
        self.interaction_mode = mcfg.get("interaction_mode", "memory")  # "memory" | "residual"
        self.num_layers = mcfg.num_gnn_layers
        self.cell_tower_interleaved = mcfg.get("cell_tower_interleaved", False)
        gnn_type = mcfg.gnn_type.lower()
        self.gat_edge_attr = mcfg.get("gat_edge_attr", False)
        self.use_morgan = mcfg.get("use_morgan", False)
        self.use_chemberta = mcfg.get("use_chemberta", False)
        # How pretrained drug features (Morgan/ChemBERTa) enter the model:
        #   gated_broadcast = atom-level gated broadcast into node features (default)
        #   late_concat    = drug-level features concatenated right before readout
        self.pretrain_fusion = mcfg.get("pretrain_fusion", "gated_broadcast")
        self.direction = mcfg.get("cross_attn_direction", "cell_to_drug")
        self.num_drug_probes = mcfg.get("num_drug_probes", self.num_probes)
        self.num_cell_tokens = mcfg.get("num_cell_tokens", self.num_probes)

        atom_dim = atom_dim or get_atom_feature_dim()
        edge_dim = edge_dim or get_bond_feature_dim()

        # --- Global registries -------------------------------------------------
        self.drug_graphs = drug_graphs
        if cell_features_table is not None:
            self.register_buffer("_cell_table", cell_features_table)
        else:
            self._cell_table = None

        if self.use_morgan and drug_morgan_table is not None:
            self.register_buffer("_morgan_table", drug_morgan_table)
        else:
            self._morgan_table = None

        if self.use_chemberta and drug_chemberta_table is not None:
            self.register_buffer("_chemberta_table", drug_chemberta_table)
        else:
            self._chemberta_table = None

        # --- Morgan / ChemBERTa → atomic-level broadcast projections ----------
        if self.use_morgan:
            self.morgan_atom_proj = nn.Linear(mcfg.get("morgan_n_bits", 1024), 32)
            if self.pretrain_fusion == "gated_broadcast":
                atom_dim += 32
        else:
            self.morgan_atom_proj = None
        if self.use_chemberta:
            self.chemberta_atom_proj = nn.Linear(768, 32)
            if self.pretrain_fusion == "gated_broadcast":
                atom_dim += 32
        else:
            self.chemberta_atom_proj = None

        # --- Pharmacophore covariance-eigenvalue features ----------------------
        # Per-atom (N, P) tensor precomputed offline (scripts/precompute_pharmacophores.py).
        # Concatenated directly into atom features — no broadcast / gating, the
        # information is already atom-local. Keeps the EGNN equivariance intact.
        self.use_pharmacophore = mcfg.get("use_pharmacophore", False)
        self.pharm_feat_dim = mcfg.get("pharm_feat_dim", 18)
        # Residual injection: project pharm separately and add to atom_proj output,
        # bypassing the shared LayerNorm. Prevents LayerNorm from amplifying the
        # sparse zero-vs-nonzero pattern into a pseudo-signal (GAT attention issue).
        self.pharm_residual = mcfg.get("pharm_residual", False)
        if self.use_pharmacophore:
            if self.pharm_residual:
                self.pharm_proj = nn.Sequential(
                    nn.Linear(self.pharm_feat_dim, mcfg.hidden_dim),
                    nn.GELU(),
                )
            else:
                atom_dim += self.pharm_feat_dim

        # --- Atom-type gating for Morgan / ChemBERTa injection -----------------
        afeat_dim = get_atom_feature_dim()  # 158
        if self.use_morgan:
            self.gate_morgan = nn.Sequential(
                nn.Linear(afeat_dim, 32),
                nn.Sigmoid(),
            )
        else:
            self.gate_morgan = None
        if self.use_chemberta:
            self.gate_bert = nn.Sequential(
                nn.Linear(afeat_dim, 32),
                nn.Sigmoid(),
            )
        else:
            self.gate_bert = None

        # --- Drug-side training controls (mirror of cell-side) -----------------
        self.enable_drug_dropout = mcfg.get("enable_drug_dropout", False)
        drug_dropout = mcfg.get("drug_dropout", mcfg.dropout) if self.enable_drug_dropout else mcfg.dropout
        self.enable_drug_noise = mcfg.get("enable_drug_noise", False)
        self.drug_noise_std = mcfg.get("drug_noise_std", 0.0)
        self.enable_drug_drop = mcfg.get("enable_drug_drop", False)
        self.drug_drop_prob = mcfg.get("drug_drop_prob", 0.0)

        # --- Atom / edge projections -------------------------------------------
        self.atom_proj = nn.Sequential(
            nn.Linear(atom_dim, mcfg.hidden_dim),
            nn.LayerNorm(mcfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(drug_dropout),
        )
        self.edge_proj = (
            nn.Linear(edge_dim, mcfg.hidden_dim)
            if gnn_type in ("gine", "egnn")
            else None
        )

        # --- Cell projector (cell_to_drug mode) --------------------------------
        self.enable_cell_dropout = mcfg.get("enable_cell_dropout", False)
        cell_dropout = mcfg.get("cell_dropout", mcfg.dropout) if self.enable_cell_dropout else mcfg.dropout
        self.enable_cell_noise = mcfg.get("enable_cell_noise", False)
        self.cell_noise_std = mcfg.get("cell_noise_std", 0.0)
        self.enable_cell_drop = mcfg.get("enable_cell_drop", False)
        self.cell_drop_prob = mcfg.get("cell_drop_prob", 0.0)
        self.cell_projector = CellQueryProjector(
            cell_dim=cell_dim,
            hidden_dim=mcfg.hidden_dim,
            num_probes=mcfg.num_probes,
            dropout=mcfg.dropout,
            cell_dropout=cell_dropout,
        )

        # --- Drug projector (drug_to_cell mode) ---------------------------------
        # Morgan/ChemBERTa are now fused at atomic level → GNN pool is complete drug feats
        self.drug_projector = nn.Sequential(
            nn.Linear(mcfg.hidden_dim, mcfg.hidden_dim * self.num_drug_probes),
            nn.Dropout(mcfg.dropout),
        )
        self.drug_probe_bias = nn.Parameter(
            torch.randn(self.num_drug_probes, mcfg.hidden_dim) * 0.02,
        )

        # --- Cell token encoder (drug_to_cell mode) ----------------------------
        chunk_dim = cell_dim // self.num_cell_tokens
        self._cell_chunk_dim = chunk_dim
        self.cell_token_encoder = nn.Sequential(
            nn.Linear(chunk_dim, mcfg.hidden_dim),
            nn.LayerNorm(mcfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(mcfg.dropout),
        )

        # --- GNN-only evolvers (for drug_to_cell prerun) -----------------------
        self.gnn_layers = nn.ModuleList(
            [
                _make_gnn_layer(mcfg, gnn_type)
                for _ in range(self.num_layers)
            ]
        )

        # --- cell_to_drug interaction blocks -----------------------------------
        self.blocks = nn.ModuleList(
            [
                InteractionBlock(
                    hidden_dim=mcfg.hidden_dim,
                    gnn_type=mcfg.gnn_type,
                    gnn_heads=mcfg.gnn_heads,
                    cross_attn_heads=mcfg.num_heads_cross_attn,
                    dropout=mcfg.dropout,
                    edge_dim=(mcfg.hidden_dim if gnn_type in ("gine", "egnn")
                              else edge_dim if mcfg.get("gat_edge_attr", False) else None),
                    rbf_type=mcfg.get("rbf_type", "saturated"),
                    rbf_centers=mcfg.get("rbf_centers"),
                    rbf_sigma=mcfg.get("rbf_sigma", 0.5),
                    use_equiv_feats=mcfg.get("use_equiv_feats", False),
                    equiv_feat_dim=mcfg.get("equiv_feat_dim", 8),
                    use_attention=mcfg.get("egnn_use_attention", False),
                    attn_heads=mcfg.get("egnn_attn_heads", 1),
                    gat_edge_attr=mcfg.get("gat_edge_attr", False),
                )
                for _ in range(self.num_layers)
            ]
        )

        # --- Interleaved intra-tower updates for the cell probes (ablation) -----
        # Off by default. When enabled, each interaction layer is preceded by a
        # CellTowerBlock (self-attn + FFN over the M probes), turning the
        # cell-side tower from a static one-shot projection into a per-layer
        # trainable tower interleaved with the cross-attention stages.
        self.cell_tower_blocks = (
            nn.ModuleList(
                [
                    CellTowerBlock(
                        hidden_dim=mcfg.hidden_dim,
                        num_heads=mcfg.get("cell_tower_heads", mcfg.num_heads_cross_attn),
                        dropout=mcfg.dropout,
                        use_self_attn=mcfg.get("cell_tower_self_attn", True),
                    )
                    for _ in range(self.num_layers)
                ]
            )
            if self.cell_tower_interleaved
            else None
        )

        # --- drug_to_cell interaction blocks -----------------------------------
        self.reversed_blocks = nn.ModuleList(
            [
                ReversedInteractionBlock(
                    hidden_dim=mcfg.hidden_dim,
                    cross_attn_heads=mcfg.num_heads_cross_attn,
                    dropout=mcfg.dropout,
                )
                for _ in range(self.num_layers)
            ]
        )

        # --- Memory accumulator ------------------------------------------------
        self.accumulator = MemoryAccumulator(
            hidden_dim=mcfg.hidden_dim,
            dropout=mcfg.dropout,
        )

        # --- Query-update FFN for residual mode ---------------------------------
        if self.interaction_mode == "residual":
            self.query_update = nn.Sequential(
                nn.Linear(mcfg.hidden_dim, mcfg.hidden_dim),
                nn.GELU(),
                nn.Dropout(mcfg.dropout),
            )
            self.query_norm = nn.LayerNorm(mcfg.hidden_dim)
        else:
            self.query_update = None
            self.query_norm = None

        # --- Readout MLP (clean — no late feature concatenation) ---------------
        max_probes = max(self.num_probes, self.num_drug_probes)
        readout_in = max_probes * mcfg.hidden_dim  # e.g. max(8,16)*64=1024
        if self.pretrain_fusion == "late_concat":
            readout_in += (32 if self.use_morgan else 0) + (32 if self.use_chemberta else 0)

        layers: list[nn.Module] = []
        prev = readout_in
        for h in mcfg.readout_hidden:
            layers.extend(
                [
                    nn.Linear(prev, h),
                    nn.LayerNorm(h),
                    nn.GELU(),
                    nn.Dropout(mcfg.dropout),
                ]
            )
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.readout = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _batch_graphs(self, drug_ids: torch.Tensor) -> Batch:
        device = drug_ids.device
        graphs = []
        for did in drug_ids.cpu().tolist():
            g = self.drug_graphs[did]
            graphs.append(g)
        return Batch.from_data_list(graphs).to(device)

    def _run_gnn_pretrain(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
        pos: torch.Tensor | None,
        batch_index: torch.Tensor,
    ) -> torch.Tensor:
        """Run GNN layers as a pure feature extractor, return graph-level readout."""
        for layer in self.gnn_layers:
            if getattr(layer, "gnn_type", None) == "egnn":
                x, pos = layer(x, edge_index, edge_attr, pos=pos)
            elif edge_attr is not None:
                x = layer(x, edge_index, edge_attr=edge_attr)
            else:
                x = layer(x, edge_index)
        return scatter(x, batch_index, dim=0, reduce="mean")  # (B, D)

    def _encode_cell_tokens(self, cell_feats: torch.Tensor) -> torch.Tensor:
        B, D = cell_feats.shape
        cell_feats = cell_feats.view(B, self.num_cell_tokens, self._cell_chunk_dim)
        return self.cell_token_encoder(cell_feats).contiguous()  # (B, N_cell, hidden_dim)

    # ------------------------------------------------------------------
    #  Broadcast drug features (shared by both cross-attention paths)
    # ------------------------------------------------------------------

    def _broadcast_drug_features(
        self,
        drug_ids: torch.Tensor,
        graph: Batch,
    ) -> tuple[list[torch.Tensor], torch.Tensor | None, list[torch.Tensor]]:
        """Build atom-level node feature list with gated Morgan/ChemBERTa broadcast.

        Applies per-sample drug noise + independent drop masks (training only).
        Called identically from both _forward_cell_to_drug and _forward_drug_to_cell,
        so both paths use the same random state within a single batch.

        Args:
            drug_ids: (B,) drug indices in the batch
            graph: PyG Batch containing .x (N, 158) and .batch (N,)

        Returns:
            (node_feats, pharm_feats, drug_feats)
            node_feats: list of atom-level tensors to concatenate into atom_proj.
                First element is always graph.x (158-dim), followed by optional
                gated Morgan (32-dim), gated ChemBERTa (32-dim), and per-atom
                pharmacophore covariance-eigenvalue features (P-dim).
            pharm_feats: per-atom pharmacophore features for residual injection,
                or None.
            drug_feats: list of drug-level (B, 32) tensors (Morgan/ChemBERTa) for
                late-concat fusion; empty when pretrain_fusion == gated_broadcast.
        """
        drug_to_node = graph.batch  # (total_nodes,), values 0..B-1
        node_feats: list[torch.Tensor] = [graph.x]  # (total_nodes, 158)
        drug_feats: list[torch.Tensor] = []

        if self.use_morgan and self._morgan_table is not None:
            morgan_32 = self.morgan_atom_proj(self._morgan_table[drug_ids])  # (B, 32)
            if self.training and self.enable_drug_noise and self.drug_noise_std > 0:
                morgan_32 = morgan_32 + torch.randn_like(morgan_32) * self.drug_noise_std
            if self.training and self.enable_drug_drop and self.drug_drop_prob > 0:
                m_mask = (torch.rand(drug_ids.size(0), 1, device=morgan_32.device) > self.drug_drop_prob).float()
                morgan_32 = morgan_32 * m_mask
            if self.pretrain_fusion == "late_concat":
                drug_feats.append(morgan_32)
            else:
                atom_gate_m = self.gate_morgan(graph.x)
                node_feats.append(morgan_32[drug_to_node] * atom_gate_m)

        if self.use_chemberta and self._chemberta_table is not None:
            bert_32 = self.chemberta_atom_proj(self._chemberta_table[drug_ids])  # (B, 32)
            if self.training and self.enable_drug_noise and self.drug_noise_std > 0:
                bert_32 = bert_32 + torch.randn_like(bert_32) * self.drug_noise_std
            if self.training and self.enable_drug_drop and self.drug_drop_prob > 0:
                b_mask = (torch.rand(drug_ids.size(0), 1, device=bert_32.device) > self.drug_drop_prob).float()
                bert_32 = bert_32 * b_mask
            if self.pretrain_fusion == "late_concat":
                drug_feats.append(bert_32)
            else:
                atom_gate_b = self.gate_bert(graph.x)
                node_feats.append(bert_32[drug_to_node] * atom_gate_b)

        # Pharmacophore features are injected residually (pharm_residual=True),
        # bypassing the shared LayerNorm inside atom_proj. This avoids the
        # LayerNorm amplifying the zero-vs-nonzero sparse pattern into a
        # pseudo-signal that distracts GAT attention. Kept as a separate
        # tensor so both forward paths can project it independently.
        pharm_feats: torch.Tensor | None = None
        if self.use_pharmacophore:
            pharm = getattr(graph, "pharm", None)
            if pharm is None:
                # Fallback: graph built without the cache entry (e.g. augmented
                # SMILES) — keep batch shape uniform with zero features.
                pharm = graph.x.new_zeros(graph.x.size(0), self.pharm_feat_dim)
            if self.pharm_residual:
                pharm_feats = pharm
            else:
                node_feats.append(pharm)

        return node_feats, pharm_feats, drug_feats

    # ------------------------------------------------------------------
    #  Forward (cell_to_drug, original)
    # ------------------------------------------------------------------

    def _forward_cell_to_drug(
        self,
        drug_ids: torch.Tensor,
        cell_ids: torch.Tensor,
        graph: Batch,
        return_attention: bool,
    ) -> dict[str, torch.Tensor]:
        cell_feats = self._cell_table[cell_ids]

        if self.training and self.enable_cell_noise and self.cell_noise_std > 0:
            cell_feats = cell_feats + torch.randn_like(cell_feats) * self.cell_noise_std

        if self.training and self.enable_cell_drop and self.cell_drop_prob > 0:
            mask = (torch.rand(cell_feats.size(0), 1, device=cell_feats.device) > self.cell_drop_prob).float()
            cell_feats = cell_feats * mask

        # --- Atomic-level broadcast: Morgan + ChemBERTa → atom nodes --------------
        node_feats, pharm_feats, drug_feats = self._broadcast_drug_features(drug_ids, graph)

        x = self.atom_proj(torch.cat(node_feats, dim=-1))          # (total_nodes, 64)
        if pharm_feats is not None:
            x = x + self.pharm_proj(pharm_feats)
        edge_attr = None
        if graph.edge_attr is not None:
            if self.gat_edge_attr:
                # GATv2Conv consumes raw bond features via its own lin_edge
                edge_attr = graph.edge_attr
            elif self.edge_proj is not None:
                edge_attr = self.edge_proj(graph.edge_attr)

        pos = getattr(graph, "pos", None)

        query = self.cell_projector(cell_feats)                    # (B, M, D)

        all_attn: list[list] = []

        if self.interaction_mode == "residual":
            # Residual mode: Q ← Q + FFN(LN(Q + extracted)), no memory accumulator
            for i, block in enumerate(self.blocks):
                if self.cell_tower_blocks is not None:
                    query = self.cell_tower_blocks[i](query)
                extracted, x, attn, pos = block(
                    query, x, graph.edge_index, graph.batch, edge_attr,
                    pos=pos,
                )
                query = query + self.query_update(self.query_norm(query + extracted))
                all_attn.append(attn)
            flat = query.view(query.size(0), -1)
        else:
            # Memory mode: use MemoryAccumulator with gated write
            memory = self.accumulator.init_memory(query, self.memory_init)
            # cell_probe_gated: pre-write cell probe info through the same gate
            if self.memory_init == "cell_probe_gated":
                memory = self.accumulator(memory, query)
            for i, block in enumerate(self.blocks):
                if self.cell_tower_blocks is not None:
                    query = self.cell_tower_blocks[i](query)
                extracted, x, attn, pos = block(
                    query, x, graph.edge_index, graph.batch, edge_attr,
                    pos=pos,
                )
                memory = self.accumulator(memory, extracted)
                all_attn.append(attn)
            flat = memory.view(memory.size(0), -1)

        # pad to uniform readout input dim (cell_to_drug: M*64, may need zero-pad)
        if drug_feats:
            flat = torch.cat([flat, *drug_feats], dim=-1)
        if flat.size(1) < self.readout[0].in_features:
            pad = torch.zeros(flat.size(0), self.readout[0].in_features - flat.size(1), device=flat.device)
            flat = torch.cat([flat, pad], dim=-1)

        pred = self.readout(flat).squeeze(-1)

        out: dict[str, torch.Tensor] = {"prediction": pred}
        if return_attention:
            out["attention_weights"] = all_attn
            out["memory"] = query if self.interaction_mode == "residual" else memory
        return out

    # ------------------------------------------------------------------
    #  Forward (drug_to_cell, reversed)
    # ------------------------------------------------------------------

    def _forward_drug_to_cell(
        self,
        drug_ids: torch.Tensor,
        cell_ids: torch.Tensor,
        graph: Batch,
        return_attention: bool,
    ) -> dict[str, torch.Tensor]:
        cell_feats = self._cell_table[cell_ids]  # (B, feat_dim)

        if self.training and self.enable_cell_noise and self.cell_noise_std > 0:
            cell_feats = cell_feats + torch.randn_like(cell_feats) * self.cell_noise_std

        if self.training and self.enable_cell_drop and self.cell_drop_prob > 0:
            mask = (torch.rand(cell_feats.size(0), 1, device=cell_feats.device) > self.cell_drop_prob).float()
            cell_feats = cell_feats * mask

        # --- Atomic-level broadcast: Morgan + ChemBERTa → atom nodes --------------
        node_feats, pharm_feats, drug_feats = self._broadcast_drug_features(drug_ids, graph)

        # -- Drug GNN feature extraction → mean pool (B, D=64) -------------------
        x = self.atom_proj(torch.cat(node_feats, dim=-1))          # (total_nodes, 64)
        if pharm_feats is not None:
            x = x + self.pharm_proj(pharm_feats)
        edge_attr = None
        if graph.edge_attr is not None:
            if self.gat_edge_attr:
                edge_attr = graph.edge_attr
            elif self.edge_proj is not None:
                edge_attr = self.edge_proj(graph.edge_attr)
        pos = getattr(graph, "pos", None)

        drug_graph_feat = self._run_gnn_pretrain(
            x, graph.edge_index, edge_attr, pos, graph.batch,
        )  # (B, 64) — already contains Morgan + ChemBERTa via atomic broadcast

        # -- Drug projector → Q_drug (pure GNN pool, no late fusion) -------------
        B = drug_graph_feat.size(0)
        drug_q = self.drug_projector(drug_graph_feat)  # Linear(64, M*64)
        drug_q = drug_q.view(B, self.num_drug_probes, self.hidden_dim)
        drug_q = drug_q + self.drug_probe_bias.unsqueeze(0)

        # -- Cell token encoder → cell_tokens K/V --------------------------------
        cell_tokens = self._encode_cell_tokens(cell_feats)  # (B, N_cell, D)

        # -- Reversed interaction blocks ----------------------------------------
        all_attn: list[list] = []

        if self.interaction_mode == "residual":
            for block in self.reversed_blocks:
                extracted, cell_tokens, attn = block(drug_q, cell_tokens)
                drug_q = drug_q + self.query_update(self.query_norm(drug_q + extracted))
                all_attn.append(attn)
            flat = drug_q.view(B, -1)
        else:
            drug_memory = self.accumulator.init_memory(drug_q, self.memory_init)
            # cell_probe_gated: pre-write drug probe info through the same gate
            if self.memory_init == "cell_probe_gated":
                drug_memory = self.accumulator(drug_memory, drug_q)
            for block in self.reversed_blocks:
                extracted, cell_tokens, attn = block(drug_q, cell_tokens)
                drug_memory = self.accumulator(drug_memory, extracted)
                all_attn.append(attn)
            flat = drug_memory.view(B, -1)

        if drug_feats:
            flat = torch.cat([flat, *drug_feats], dim=-1)
        pred = self.readout(flat).squeeze(-1)

        out: dict[str, torch.Tensor] = {"prediction": pred}
        if return_attention:
            out["attention_weights"] = all_attn
            out["memory"] = drug_q if self.interaction_mode == "residual" else drug_memory
        return out

    # ------------------------------------------------------------------
    #  Public forward
    # ------------------------------------------------------------------

    def forward(
        self,
        batch: dict,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor]:
        drug_ids = batch["drug_ids"]
        cell_ids = batch["cell_ids"]
        graph = self._batch_graphs(drug_ids)

        if self.direction == "drug_to_cell":
            return self._forward_drug_to_cell(
                drug_ids, cell_ids, graph, return_attention,
            )
        return self._forward_cell_to_drug(
            drug_ids, cell_ids, graph, return_attention,
        )

    # ------------------------------------------------------------------
    #  Gradient monitoring for branch-balance diagnosis
    # ------------------------------------------------------------------

    def get_branch_grad_norms(self) -> dict[str, float]:
        norms: dict[str, float] = {}

        # Compute on the SAME parameter groups as get_branch_params()
        # so that monitored norms match what clip_grad_norm_ operates on.
        bp = self.get_branch_params()
        norms["cell"] = _param_grad_norm(bp["cell"])
        norms["drug_gnn"] = _param_grad_norm(bp["drug"])
        norms["shared"] = _param_grad_norm(bp["shared"])
        norms["cell_rms"] = _param_grad_rms(bp["cell"])
        norms["drug_rms"] = _param_grad_rms(bp["drug"])
        norms["shared_rms"] = _param_grad_rms(bp["shared"])

        norms["morgan"] = 0.0
        norms["bert"] = 0.0
        if self.morgan_atom_proj is not None:
            norms["morgan"] = _param_grad_norm(list(self.morgan_atom_proj.parameters()))
        if self.chemberta_atom_proj is not None:
            norms["bert"] = _param_grad_norm(list(self.chemberta_atom_proj.parameters()))

        return norms

    def get_branch_params(self) -> dict[str, list[torch.nn.Parameter]]:
        """Return {cell, drug, shared} parameter groups for per-branch gradient clipping."""
        cell_params: list[torch.nn.Parameter] = []
        drug_params: list[torch.nn.Parameter] = []
        shared_params: list[torch.nn.Parameter] = []

        # --- Cell branch -------------------------------------------------------
        cell_params += list(self.cell_projector.parameters())

        # --- Drug branch -------------------------------------------------------
        drug_params += list(self.atom_proj.parameters())
        if self.edge_proj is not None:
            drug_params += list(self.edge_proj.parameters())
        if self.direction == "drug_to_cell":
            for blk in self.reversed_blocks:
                drug_params += list(blk.parameters())
        else:
            for blk in self.blocks:
                drug_params += list(blk.parameters())
        if self.morgan_atom_proj is not None:
            drug_params += list(self.morgan_atom_proj.parameters())
        if self.chemberta_atom_proj is not None:
            drug_params += list(self.chemberta_atom_proj.parameters())
        if self.gate_morgan is not None:
            drug_params += list(self.gate_morgan.parameters())
        if self.gate_bert is not None:
            drug_params += list(self.gate_bert.parameters())
        if self.drug_projector is not None:
            drug_params += list(self.drug_projector.parameters())
        if self.drug_probe_bias is not None:
            drug_params.append(self.drug_probe_bias)
        # GNN-only layers (used in drug_to_cell prerun)
        if hasattr(self, "gnn_layers") and self.gnn_layers is not None:
            for layer in self.gnn_layers:
                drug_params += list(layer.parameters())
        # Cell token encoder (used in drug_to_cell, processes cell features)
        if hasattr(self, "cell_token_encoder") and self.cell_token_encoder is not None:
            cell_params += list(self.cell_token_encoder.parameters())
        # Interleaved intra-tower cell blocks (cell_to_drug path)
        if getattr(self, "cell_tower_blocks", None) is not None:
            cell_params += [p for blk in self.cell_tower_blocks for p in blk.parameters()]

        # --- Shared (accumulator + readout) ------------------------------------
        shared_params += list(self.accumulator.parameters())
        shared_params += list(self.readout.parameters())

        # Query-update FFN (residual mode) — belongs to whichever branch IS the query
        if hasattr(self, "query_update") and self.query_update is not None:
            if self.direction == "drug_to_cell":
                drug_params += list(self.query_update.parameters())
                drug_params += list(self.query_norm.parameters())
            else:
                cell_params += list(self.query_update.parameters())
                cell_params += list(self.query_norm.parameters())

        return {"cell": cell_params, "drug": drug_params, "shared": shared_params}

    # ------------------------------------------------------------------
    #  Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        cell_dim: int,
        cfg: DictConfig,
        drug_graphs: list[Data],
        cell_features_table: torch.Tensor,
        drug_morgan_table: torch.Tensor | None = None,
        drug_chemberta_table: torch.Tensor | None = None,
    ) -> "CellDrugModel":
        return cls(
            cell_dim=cell_dim, cfg=cfg,
            drug_graphs=drug_graphs,
            cell_features_table=cell_features_table,
            drug_morgan_table=drug_morgan_table,
            drug_chemberta_table=drug_chemberta_table,
        )


def _make_gnn_layer(mcfg, gnn_type: str) -> nn.Module:
    """Build a single standalone GNN layer (no cross-attn) for drug pretraining."""
    from src.models.layers import GraphEvolver
    gat_ea = mcfg.get("gat_edge_attr", False)
    return GraphEvolver(
        hidden_dim=mcfg.hidden_dim,
        gnn_type=gnn_type,
        heads=mcfg.gnn_heads,
        dropout=mcfg.dropout,
        edge_dim=(mcfg.hidden_dim if gnn_type in ("gine", "egnn")
                  else get_bond_feature_dim() if gnn_type == "gat" and gat_ea else None),
        gat_edge_attr=gat_ea,
    )


def _param_grad_norm(params: list[torch.nn.Parameter]) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.data.norm().item() ** 2
    return total ** 0.5


def _param_grad_rms(params: list[torch.nn.Parameter]) -> float:
    """Per-element RMS gradient norm — decouples from parameter count.

    RMS = sqrt( sum(||g_i||^2) / total_numel ),  where total_numel
    is the sum of p.numel() over all params with non-null grad.
    """
    total = 0.0
    total_elements = 0
    for p in params:
        if p.grad is not None:
            total += p.grad.data.norm().item() ** 2
            total_elements += p.numel()
    if total_elements == 0:
        return 0.0
    return (total / total_elements) ** 0.5
