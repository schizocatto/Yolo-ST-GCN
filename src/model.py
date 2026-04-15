"""
model.py
ST-GCN model components with joint-spec-aware graph construction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import IN_CHANNELS, NUM_CLASSES
from src.graph import GraphSkeleton


class STGCN_Block(nn.Module):
    """One ST-GCN block: graph conv + temporal conv + residual."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        A: torch.Tensor,
        stride: int = 1,
        residual: bool = True,
        edge_importance: bool = False,
    ):
        super().__init__()
        self.K = A.size(0)
        self.A = nn.Parameter(A, requires_grad=False)

        if edge_importance:
            self.edge_weight = nn.Parameter(torch.ones(A.size()))
        else:
            self.edge_weight = 1.0

        self.gcn = nn.Conv2d(in_ch, out_ch * self.K, kernel_size=1)
        self.tcn = nn.Conv2d(
            out_ch,
            out_ch,
            kernel_size=(9, 1),
            padding=(4, 0),
            stride=(stride, 1),
        )

        if not residual:
            self.res = lambda x: 0
        elif in_ch == out_ch and stride == 1:
            self.res = lambda x: x
        else:
            self.res = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=(stride, 1))

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, t, v = x.size()
        r = self.res(x)

        x = self.gcn(x).view(n, self.K, -1, t, v)
        weighted_A = self.A * self.edge_weight
        x = torch.einsum('nkctv,kvw->nctw', x, weighted_A)
        x = self.tcn(x)

        return self.relu(x + r)


class Model_STGCN(nn.Module):
    """ST-GCN classifier for a selectable joint spec."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        in_channels: int = IN_CHANNELS,
        joint_spec: str = 'penn14',
        edge_importance: bool = True,
    ):
        super().__init__()
        self.graph = GraphSkeleton(joint_spec=joint_spec)
        A = self.graph.A

        self.data_bn = nn.BatchNorm1d(in_channels * self.graph.num_node)

        self.st_gcn_networks = nn.ModuleList([
            STGCN_Block(in_channels, 64, A, residual=False, edge_importance=edge_importance),
            STGCN_Block(64, 64, A, edge_importance=edge_importance),
            STGCN_Block(64, 64, A, edge_importance=edge_importance),
            STGCN_Block(64, 64, A, edge_importance=edge_importance),
            STGCN_Block(64, 128, A, stride=2, edge_importance=edge_importance),
            STGCN_Block(128, 128, A, edge_importance=edge_importance),
            STGCN_Block(128, 128, A, edge_importance=edge_importance),
            STGCN_Block(128, 256, A, stride=2, edge_importance=edge_importance),
            STGCN_Block(256, 256, A, edge_importance=edge_importance),
            STGCN_Block(256, 256, A, edge_importance=edge_importance),
        ])

        self.fcn = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, t, v, m = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(n, m * v * c, t)
        x = self.data_bn(x)
        x = x.view(n, m, v, c, t).permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(n * m, c, t, v)

        for gcn in self.st_gcn_networks:
            x = gcn(x)

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(n, m, -1, 1, 1).mean(dim=1)
        x = self.fcn(x)
        return x.view(x.size(0), -1)
