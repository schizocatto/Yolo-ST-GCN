"""
two_stream_stgcn.py
Two-stream ST-GCN with late fusion (joint stream + bone stream).
"""

import torch
import torch.nn as nn

from src.model import Model_STGCN


class TwoStream_STGCN(nn.Module):
    """
    Late-fusion two-stream ST-GCN.

    Inputs
    ------
    joint_data : (N, C, T, V, M)
    bone_data  : (N, C, T, V, M)
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 2,
        joint_spec: str = 'penn14',
        edge_importance: bool = True,
    ):
        super().__init__()
        self.joint_stream = Model_STGCN(
            num_classes=num_classes,
            in_channels=in_channels,
            joint_spec=joint_spec,
            edge_importance=edge_importance,
        )
        self.bone_stream = Model_STGCN(
            num_classes=num_classes,
            in_channels=in_channels,
            joint_spec=joint_spec,
            edge_importance=edge_importance,
        )

        # Learnable fusion gate in [0, 1] after sigmoid.
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, joint_data: torch.Tensor, bone_data: torch.Tensor) -> torch.Tensor:
        out_joint = self.joint_stream(joint_data)
        out_bone = self.bone_stream(bone_data)

        alpha = torch.sigmoid(self.alpha_logit)
        return alpha * out_joint + (1.0 - alpha) * out_bone
