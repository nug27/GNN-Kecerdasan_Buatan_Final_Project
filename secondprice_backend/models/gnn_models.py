"""
SecondPrice — Definisi Arsitektur GNN
Diekstrak dari SecondPrice_GNN.ipynb

Berisi:
  - GraphSAGERegressor  : Heterogeneous GraphSAGE untuk prediksi harga
  - GATRegressor        : Heterogeneous GAT untuk prediksi harga
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, HeteroConv


# ── Tipe edge yang digunakan dalam graph ──────────────────────────────────────
# (product, has_brand, brand)
# (brand, rev_has_brand, product)
# (product, in_category, category)
# (category, rev_in_category, product)
EDGE_TYPES = [
    ("product", "has_brand",          "brand"),
    ("brand",   "rev_has_brand",      "product"),
    ("product", "in_category",        "category"),
    ("category","rev_in_category",    "product"),
]


class GraphSAGERegressor(nn.Module):
    """
    Heterogeneous GraphSAGE model untuk prediksi harga (log-scale).

    Arsitektur:
        Input Projection (per node type)
        → HeteroConv[SAGEConv] × 2
        → MLP Regressor (product nodes only)

    Parameters
    ----------
    in_channels_dict : dict[str, int]
        {"product": D_p, "brand": D_b, "category": D_c}
    hidden_channels : int
    dropout : float
    edge_types : list[tuple]
        Daftar edge type yang ada di graph; default = EDGE_TYPES di atas.
    """

    def __init__(
        self,
        in_channels_dict: dict,
        hidden_channels: int = 128,
        dropout: float = 0.3,
        edge_types: list = None,
    ):
        super().__init__()
        self.dropout = dropout
        _et = edge_types or EDGE_TYPES

        self.input_proj = nn.ModuleDict({
            nt: nn.Linear(in_ch, hidden_channels)
            for nt, in_ch in in_channels_dict.items()
        })

        self.conv1 = HeteroConv({
            et: SAGEConv(hidden_channels, hidden_channels, normalize=True)
            for et in _et
        }, aggr="mean")

        self.conv2 = HeteroConv({
            et: SAGEConv(hidden_channels, hidden_channels, normalize=True)
            for et in _et
        }, aggr="mean")

        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x_dict: dict, edge_index_dict: dict) -> torch.Tensor:
        h = {nt: F.relu(self.input_proj[nt](x)) for nt, x in x_dict.items()}

        h = self.conv1(h, edge_index_dict)
        h = {nt: F.relu(x) for nt, x in h.items()}
        h = {nt: F.dropout(x, p=self.dropout, training=self.training)
             for nt, x in h.items()}

        h = self.conv2(h, edge_index_dict)
        h = {nt: F.relu(x) for nt, x in h.items()}

        return self.mlp(h["product"]).squeeze(-1)  # (N_product,)


class GATRegressor(nn.Module):
    """
    Heterogeneous GAT (Graph Attention Network) untuk prediksi harga.

    Arsitektur:
        Input Projection (per node type)
        → HeteroConv[GATConv, multi-head] × 2
        → MLP Regressor (product nodes only)

    Parameters
    ----------
    in_channels_dict : dict[str, int]
    hidden_channels : int
    heads : int   — jumlah attention head pada layer 1
    dropout : float
    edge_types : list[tuple]
    """

    def __init__(
        self,
        in_channels_dict: dict,
        hidden_channels: int = 64,
        heads: int = 4,
        dropout: float = 0.3,
        edge_types: list = None,
    ):
        super().__init__()
        self.dropout = dropout
        _et = edge_types or EDGE_TYPES

        self.input_proj = nn.ModuleDict({
            nt: nn.Linear(in_ch, hidden_channels)
            for nt, in_ch in in_channels_dict.items()
        })

        self.conv1 = HeteroConv({
            et: GATConv(
                hidden_channels,
                hidden_channels // heads,
                heads=heads,
                dropout=dropout,
                add_self_loops=False,
            )
            for et in _et
        }, aggr="mean")

        self.conv2 = HeteroConv({
            et: GATConv(
                hidden_channels,
                hidden_channels,
                heads=1,
                dropout=dropout,
                add_self_loops=False,
            )
            for et in _et
        }, aggr="mean")

        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, x_dict: dict, edge_index_dict: dict) -> torch.Tensor:
        h = {nt: F.elu(self.input_proj[nt](x)) for nt, x in x_dict.items()}

        h = self.conv1(h, edge_index_dict)
        h = {nt: F.elu(x) for nt, x in h.items()}
        h = {nt: F.dropout(x, p=self.dropout, training=self.training)
             for nt, x in h.items()}

        h = self.conv2(h, edge_index_dict)
        h = {nt: F.elu(x) for nt, x in h.items()}

        return self.mlp(h["product"]).squeeze(-1)  # (N_product,)
