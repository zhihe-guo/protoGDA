"""Cell-line feature projector to query matrix Q (M x D)."""

from __future__ import annotations

import torch
import torch.nn as nn


class CellQueryProjector(nn.Module):
    """
    Project high-dimensional cell-line features to M probe tokens.

    Output shape: (batch, num_probes, hidden_dim)

    Uses a tight bottleneck with independent cell-side dropout (typically
    higher than the global model dropout) to suppress cell dominance
    over drug signals.
    """

    def __init__(
        self,
        cell_dim: int,
        hidden_dim: int,
        num_probes: int,
        dropout: float = 0.1,
        cell_dropout: float | None = None,
    ):
        super().__init__()
        self.num_probes = num_probes
        self.hidden_dim = hidden_dim
        out_dim = hidden_dim * num_probes  # e.g. 16 * 64 = 1024
        cd = cell_dropout if cell_dropout is not None else dropout

        self.encoder = nn.Sequential(
            nn.Linear(cell_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(cd),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(cd),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(cd),
        )
        self.probe_bias = nn.Parameter(torch.randn(num_probes, hidden_dim) * 0.02)

    def forward(self, cell_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cell_features: (B, cell_dim)
        Returns:
            Q: (B, M, D)
        """
        b = cell_features.size(0)
        out = self.encoder(cell_features)
        out = out.view(b, self.num_probes, self.hidden_dim)
        out = out + self.probe_bias.unsqueeze(0)
        return out
