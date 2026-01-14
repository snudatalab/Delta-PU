import torch
from torch.nn import functional as func
from torch import nn

from torch_geometric import nn as gnn
from torch_geometric.nn import (
    global_add_pool, global_mean_pool, global_max_pool,
    GlobalAttention, Set2Set, GraphMultisetTransformer
)
from torch.func import functional_call

import os

class MLP(nn.Module):
    def __init__(self, num_features, num_classes, hidden_units=32, num_layers=1):
        """
        Initialize a multi-layer perceptron (MLP).

        Args:
            num_features (int): Input feature dimension.
            num_classes (int): Output dimension (e.g., number of classes).
            hidden_units (int): Hidden layer size.
            num_layers (int): Number of total MLP layers (≥1).
        """

        super(MLP, self).__init__()
        if num_layers == 1:
            self.layers = nn.Linear(num_features, num_classes)
        elif num_layers > 1:
            layers = [nn.Linear(num_features, hidden_units),
                      nn.BatchNorm1d(hidden_units),
                      nn.ReLU()]
            for _ in range(num_layers - 2):
                layers.extend([nn.Linear(hidden_units, hidden_units),
                               nn.BatchNorm1d(hidden_units),
                               nn.ReLU()])
            layers.append(nn.Linear(hidden_units, num_classes))
            self.layers = nn.Sequential(*layers)
        else:
            raise ValueError()

    def forward(self, x):
        """
        Forward pass for MLP.

        Args:
            x (Tensor): Input node features.

        Returns:
            Tensor: Output logits.
        """

        return self.layers(x)


class GIN(nn.Module):
    def __init__(self, num_features, num_classes, hidden_units=32, num_layers=3, dropout=0.15,
                 mlp_layers=2, train_eps=False, pooling='mean'):
        """
        Initialize a GIN (Graph Isomorphism Network) with optional hop-weighted output.

        Args:
            num_features (int): Input feature dimension.
            num_classes (int): Output dimension (e.g., number of classes).
            hidden_units (int): Hidden dimension for convolution layers.
            num_layers (int): Total number of GIN layers.
            dropout (float): Dropout rate for output combination.
            mlp_layers (int): Number of layers in the MLP used in each GINConv.
            train_eps (bool): Whether to learn the epsilon weighting in GINConv.
            pooling (str): Global pooling type ('mean', 'attention', or 'gmt').
        """

        super(GIN, self).__init__()
        convs, bns = [], []
        linears = [nn.Linear(num_features, num_classes)]
        for i in range(num_layers - 1):
            input_dim = num_features if i == 0 else hidden_units
            convs.append(gnn.GINConv(MLP(input_dim, hidden_units, hidden_units, mlp_layers),
                                     train_eps=train_eps))
            bns.append(nn.BatchNorm1d(hidden_units))
            linears.append(nn.Linear(hidden_units, num_classes))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.linears = nn.ModuleList(linears)
        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = pooling
        self.inst_w = None

    def forward(self, x, edge_index, batch, hop_weights=None):
        """
        Standard forward pass through the GIN model.

        Args:
            x (Tensor): Node features.
            edge_index (LongTensor): Edge list in COO format.
            batch (LongTensor): Batch assignment vector for each node.
            hop_weights (Tensor, optional): [B, L] tensor of per-hop combination weights.
                                             If None, uniform weighting is used.

        Returns:
            Tensor: [B, C] logits after hop-weighted combination.
        """

        h_list = [x]
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h_list[-1], edge_index)
            h_list.append(torch.relu(bn(h)))

        out = 0
        pooled_list = []
        for i in range(self.num_layers):
            h_pooled = global_mean_pool(h_list[i], batch)
            pooled_list.append(self.linears[i](h_pooled))

        if hop_weights is None:
            hop_weights = torch.ones((pooled_list[0].size(0), self.num_layers), device=x.device) / self.num_layers

        for i in range(self.num_layers):
            out += func.dropout(pooled_list[i], self.dropout, self.training) * hop_weights[:, i].unsqueeze(1)

        return out

    def parameterized_forward(self, x, edge_index, batch, weights, hop_weights=None):
        """
        Functional forward pass using externally supplied parameters.

        Args:
            x (Tensor): Node features.
            edge_index (LongTensor): Edge list.
            batch (LongTensor): Batch vector.
            weights (list[Tensor]): List of parameters aligned with model.named_parameters().
            hop_weights (Tensor, optional): Per-hop weights, same as in `forward`.

        Returns:
            Tensor: Output logits using functional weights.
        """

        # 현재 모델의 파라미터 이름 순서대로 매핑
        param_names = [name for name, _ in self.named_parameters()]
        param_dict = dict(zip(param_names, weights))

        # functional_call은 autograd 그래프를 안전하게 유지하면서 모델을 호출함
        return functional_call(self, param_dict, (x, edge_index, batch, hop_weights))