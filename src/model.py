"""
model.py
ST-GCN model components.

STGCN_Block — one spatial-temporal graph-convolution + temporal-convolution block.
Model_STGCN — full 6-block ST-GCN classifier.

Input tensor shape: (N, C, T, V, M)
  N — batch size
  C — channels  (2 for x,y coordinates)
  T — frames     (64)
  V — joints     (14)
  M — persons    (1 in Penn Action)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.graph import Graph_PennAction_14Nodes, Graph_COCO17_18Nodes
from src.config import NUM_CLASSES, IN_CHANNELS


class STGCN_Block(nn.Module):
    """
    One ST-GCN block: spatial graph convolution followed by temporal convolution.

    Parameters
    ----------
    in_ch     : input channels
    out_ch    : output channels
    A         : adjacency tensor  (K, V, V)
    stride    : temporal stride  (1 or 2)
    residual  : whether to use a residual connection
    """
class STGCN_Block(nn.Module):
    """
    One ST-GCN block: spatial graph convolution followed by temporal convolution.
    (Đã tích hợp Edge Importance Weighting)
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        A: torch.Tensor,
        stride: int = 1,
        residual: bool = True,
        edge_importance: bool = False, # BỔ SUNG: Cờ bật/tắt trọng số cạnh
    ):
        super().__init__()
        K = A.size(0)
        self.K = K
        
        # Ma trận kề gốc giữ cố định
        self.A = nn.Parameter(A, requires_grad=False)
        
        # ==========================================
        # BỔ SUNG 1: Khởi tạo Trọng số quan trọng cạnh
        # ==========================================
        if edge_importance:
            # Khởi tạo ma trận toàn số 1, cùng size (K, V, V) với A
            self.edge_weight = nn.Parameter(torch.ones(A.size()))
        else:
            # Nếu tắt, trọng số chỉ là 1 số vô hướng
            self.edge_weight = 1.0

        # Spatial graph convolution (1x1 across joints)
        self.gcn = nn.Conv2d(in_ch, out_ch * K, kernel_size=1)

        # Temporal convolution (9x1 kernel with same-padding)
        self.tcn = nn.Conv2d(
            out_ch, out_ch,
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
        N, C, T, V = x.size()
        r = self.res(x)

        x = self.gcn(x).view(N, self.K, -1, T, V)          # (N, K, out_ch, T, V)
        
        # ==========================================
        # BỔ SUNG 2: Áp dụng trọng số trước khi einsum
        # ==========================================
        weighted_A = self.A * self.edge_weight             # (K, V, V) * (K, V, V)
        
        x = torch.einsum('nkctv,kvw->nctw', x, weighted_A) # (N, out_ch, T, V)
        x = self.tcn(x)
        
        return self.relu(x + r)

class Model_STGCN(nn.Module):
    """
    6-block ST-GCN for exercise action recognition on Penn Action (14 joints).

    Architecture
    ------------
    BatchNorm1d → Block×6 → GlobalAvgPool → 1×1-Conv classifier
    """

    def __init__(self, num_classes: int = NUM_CLASSES, in_channels: int = IN_CHANNELS):
        super().__init__()
        self.graph = Graph_PennAction_14Nodes()
        A = self.graph.A

        self.data_bn = nn.BatchNorm1d(in_channels * self.graph.num_node)

        self.st_gcn_networks = nn.ModuleList([
            STGCN_Block(in_channels, 64,  A, residual=False),
            STGCN_Block(64,  64,  A),
            STGCN_Block(64,  64,  A),
            STGCN_Block(64,  64,  A),
            STGCN_Block(64,  128, A, stride=2),
            STGCN_Block(128, 128, A),
            STGCN_Block(128, 128, A),
            STGCN_Block(128, 256, A, stride=2),
            STGCN_Block(256, 256, A),
            STGCN_Block(256, 256, A),
        ])

        self.fcn = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, T, V, M = x.size()

        # Batch-normalise over (M, V, C) as a flat feature vector
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        for gcn in self.st_gcn_networks:
            x = gcn(x)

        # Global average pool → classify
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)
        x = self.fcn(x)
        return x.view(x.size(0), -1)


class Model_STGCN_COCO18(nn.Module):
    """
    ST-GCN for full COCO-17 + virtual center (18 joints).

    Uses Graph_COCO17_18Nodes which builds centripetal/centrifugal partitions
    over all 17 COCO keypoints (including face landmarks) plus virtual center.

    Input tensor shape: (N, C=2, T, V=18, M=1)
    """

    def __init__(self, num_classes: int = NUM_CLASSES, in_channels: int = IN_CHANNELS):
        super().__init__()
        self.graph = Graph_COCO17_18Nodes()
        A = self.graph.A

        self.data_bn = nn.BatchNorm1d(in_channels * self.graph.num_node)

        self.st_gcn_networks = nn.ModuleList([
            STGCN_Block(in_channels, 64,  A, residual=False),
            STGCN_Block(64,  64,  A),
            STGCN_Block(64,  64,  A),
            STGCN_Block(64,  64,  A),
            STGCN_Block(64,  128, A, stride=2),
            STGCN_Block(128, 128, A),
            STGCN_Block(128, 128, A),
            STGCN_Block(128, 256, A, stride=2),
            STGCN_Block(256, 256, A),
            STGCN_Block(256, 256, A),
        ])

        self.fcn = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        for gcn in self.st_gcn_networks:
            x = gcn(x)

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)
        x = self.fcn(x)
        return x.view(x.size(0), -1)
