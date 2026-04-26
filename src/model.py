"""
model.py
ST-GCN model components with joint-spec-aware graph construction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import IN_CHANNELS, NUM_CLASSES
from src.graph import GraphSkeleton

# Each entry is (out_channels, temporal_stride) for one block.
# The first block always has residual=False; in_channels flows from the previous block.
# All configs end at 256 output channels so the classifier head stays identical.
_DEPTH_CONFIGS: dict[int, list[tuple[int, int]]] = {
    10: [(64,1),(64,1),(64,1),(64,1),(128,2),(128,1),(128,1),(256,2),(256,1),(256,1)],
    8:  [(64,1),(64,1),(64,1),(128,2),(128,1),(256,2),(256,1),(256,1)],
    6:  [(64,1),(64,1),(128,2),(128,1),(256,2),(256,1)],
    4:  [(64,1),(128,2),(256,2),(256,1)],
}


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
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.K = A.size(0)
        self.A = nn.Parameter(A, requires_grad=False)

        if edge_importance:
            self.edge_weight = nn.Parameter(torch.ones(A.size()))
        else:
            self.edge_weight = 1.0

        self.gcn = nn.Conv2d(in_ch, out_ch * self.K, kernel_size=1)
        self.gcn_bn = nn.BatchNorm2d(out_ch)
        self.tcn = nn.Conv2d(
            out_ch,
            out_ch,
            kernel_size=(9, 1),
            padding=(4, 0),
            stride=(stride, 1),
        )
        self.tcn_bn = nn.BatchNorm2d(out_ch)

        if not residual:
            self.res = lambda x: 0
        elif in_ch == out_ch and stride == 1:
            self.res = lambda x: x
        else:
            self.res = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=(stride, 1))

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob) if dropout_prob > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, t, v = x.size()
        r = self.res(x)

        x = self.gcn(x).view(n, self.K, -1, t, v)
        weighted_A = self.A * self.edge_weight
        x = torch.einsum('nkctv,kvw->nctw', x, weighted_A)
        x = self.gcn_bn(x)
        x = self.tcn(x)
        x = self.tcn_bn(x)

        return self.dropout(self.relu(x + r))


class Model_STGCN(nn.Module):
    """ST-GCN classifier for a selectable joint spec."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        in_channels: int = IN_CHANNELS,
        joint_spec: str = 'penn14',
        edge_importance: bool = True,
        block_dropout: float = 0.0,
        classifier_dropout: float = 0.3,
        depth: int = 10,
    ):
        if depth not in _DEPTH_CONFIGS:
            raise ValueError(f'Unsupported depth {depth}. Choose from {sorted(_DEPTH_CONFIGS)}.')
        super().__init__()
        self.graph = GraphSkeleton(joint_spec=joint_spec)
        A = self.graph.A

        self.data_bn = nn.BatchNorm1d(in_channels * self.graph.num_node)

        blocks = []
        ch = in_channels
        for i, (out_ch, stride) in enumerate(_DEPTH_CONFIGS[depth]):
            blocks.append(STGCN_Block(
                ch, out_ch, A,
                stride=stride,
                residual=(i > 0),
                edge_importance=edge_importance,
                dropout_prob=block_dropout,
            ))
            ch = out_ch
        self.st_gcn_networks = nn.ModuleList(blocks)

        self.classifier_dropout = (
            nn.Dropout(p=classifier_dropout) if classifier_dropout > 0 else nn.Identity()
        )
        self.fcn = nn.Conv2d(ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, t, v, m = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(n, m * v * c, t)
        x = self.data_bn(x)
        x = x.view(n, m, v, c, t).permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(n * m, c, t, v)

        for gcn in self.st_gcn_networks:
            x = gcn(x)

        x = F.avg_pool2d(x, x.size()[2:])
        x = self.classifier_dropout(x)
        x = x.view(n, m, -1, 1, 1).mean(dim=1)
        x = self.fcn(x)
        return x.view(x.size(0), -1)
