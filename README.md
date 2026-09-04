# CellQuery — Cell-Drug Cross-Attention Model

Drug–cell line interaction (IC50) prediction using a multi-stage cross-modal architecture: cell-line features act as query probes that perform one-way cross-attention on drug molecular graphs at every GNN layer, with results accumulated into a memory object for final prediction.

## Setup

### Linux

```bash
# One-shot: core + data + dev
pip install -e ".[linux-data,dev]"
```

### Windows

`pytdc` pulls in `tiledbsoma` (C++ build on Windows fails and GDSC2 does not need it), so install it with `--no-deps`:

```bash
# 1. Core dependencies
pip install -e .

# 2. pytdc without heavy sub-deps (only GDSC2 DrugRes works)
pip install pytdc --no-deps
pip install -r requirements-data-windows.txt

# 3. Optional: Jupyter / visualization
pip install -e ".[dev]"
```

> **Why separate?** `tiledbsoma` has no pre-built wheel for Windows; `pytdc` only uses it for single-cell modules. The GDSC2 `DrugRes` loader works fine without it.

## Quick configuration

See [`config/default.yaml`](config/default.yaml) for all knobs. Key architecture params:

| Param | Value | Meaning |
|-------|-------|---------|
| `M` | 16 | cell-line pathway probe tokens (`num_probes`) |
| `D` | 256 | hidden dimension (`hidden_dim`) |
| `num_gnn_layers` | 4 | stacked GNN + cross-attention blocks |
| `gnn_type` | `egnn` | backbone conv (`egnn` / `gine` / `gat` / `gcn` / `egconv`) |
| `accumulator` | `residual` | M_accum update (`residual` / `gru`) |
| cell dim | ~17,737 | gene expression (RMA-normalized, GDSC2) |

## Train

```bash
python scripts/train.py
python scripts/train.py --config config/default.yaml
```

## Evaluate

```bash
python scripts/eval.py --checkpoint checkpoints/best.pt
```

## Architecture

`Q` = cell-line probe matrix. At each GNN layer `Q` extracts from graph nodes, writing to `M_accum`. Only `M_accum` feeds the final readout MLP.

```
Cell_Features ──► MLP Projector ──► Q (M×D) ──────────────────┐
                                                               │  (re-used every layer)
Drug_SMILES ──► Mol_Graph (V×D) ──► GNN_0 ──► Graph_L1 ──► ... ──► Graph_LL
                    │                   │            │
                    ▼                   ▼            ▼
               CrossAttn_0       CrossAttn_1    CrossAttn_L
                    │                   │            │
                    ▼                   ▼            ▼
        M_accum ──► + ──► M_accum ──► + ──► ... ──► M_accum ──► Flatten ──► MLP ──► y_hat
```

See [idea.md](idea.md) for the research motivation and design rationale.
