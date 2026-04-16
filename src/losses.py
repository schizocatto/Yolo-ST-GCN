"""
losses.py
Classification loss utilities for ST-GCN training/evaluation.
"""

from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss over raw logits."""

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
    ):
        super().__init__()
        if reduction not in {'mean', 'sum', 'none'}:
            raise ValueError("reduction must be one of: 'mean', 'sum', 'none'.")
        if gamma < 0:
            raise ValueError('gamma must be >= 0.')

        self.gamma = float(gamma)
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer('alpha', alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        focal_term = torch.pow(1.0 - pt, self.gamma)
        loss = focal_term * ce

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            loss = alpha_t * loss

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


def compute_smoothed_alpha(
    labels: Union[np.ndarray, torch.Tensor],
    num_classes: int,
    mode: str = 'sqrt_inverse',
) -> torch.Tensor:
    """
    Compute class alpha weights from label frequencies.

    Supported modes:
    - none: all ones
    - inverse: 1 / count
    - sqrt_inverse: 1 / sqrt(count)

    The resulting vector is normalized to mean=1.
    """
    if num_classes <= 0:
        raise ValueError('num_classes must be > 0 when computing alpha weights.')

    mode_norm = mode.strip().lower()
    if mode_norm not in {'none', 'inverse', 'sqrt_inverse'}:
        raise ValueError("mode must be one of: 'none', 'inverse', 'sqrt_inverse'.")

    if mode_norm == 'none':
        return torch.ones(num_classes, dtype=torch.float32)

    labels_np = labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
    labels_np = labels_np.astype(np.int64).reshape(-1)
    if labels_np.size == 0:
        raise ValueError('Cannot compute alpha weights from empty labels.')

    if labels_np.min() < 0 or labels_np.max() >= num_classes:
        raise ValueError('labels contain class ids outside [0, num_classes).')

    counts = np.bincount(labels_np, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)

    if mode_norm == 'inverse':
        alpha = 1.0 / counts
    else:
        alpha = 1.0 / np.sqrt(counts)

    alpha = alpha / np.mean(alpha)
    return torch.tensor(alpha, dtype=torch.float32)


def build_classification_criterion(
    loss_name: str,
    device: torch.device,
    focal_gamma: float = 2.0,
    focal_alpha: Optional[torch.Tensor] = None,
) -> nn.Module:
    """Factory for classification criterion used by train/inference scripts."""
    name = loss_name.strip().lower()
    if name in {'ce', 'cross_entropy', 'crossentropyloss'}:
        return nn.CrossEntropyLoss()
    if name == 'focal':
        alpha = focal_alpha.to(device) if focal_alpha is not None else None
        return FocalLoss(alpha=alpha, gamma=focal_gamma)

    raise ValueError(
        f"Unsupported loss_name='{loss_name}'. Expected one of: 'ce', 'cross_entropy', 'focal'."
    )
