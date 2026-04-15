"""
dataset.py
Unified dataset entrypoint for Penn and COCO keypoint sources.

Key functions
-------------
build_data_tensors(labels_dir, dataset_format, ...)
    Dispatch to Penn/COCO loaders and always return ST-GCN-ready tensors
    in Penn joint layout: (N, 2, T, 14, 1).

PennActionDataset
    PyTorch Dataset wrapping pre-built numpy arrays.
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import CLASS_TO_ID, EXERCISE_CLASSES, TARGET_FRAMES
from src.coco_dataset import build_coco_data_tensors
from src.gym288_dataset import build_gym288_data_tensors
from src.penn_dataset import build_penn_data_tensors, load_mat_index
from src.skeleton_utils import (
    add_virtual_center_joint,
    remap_coco17_to_penn13,
    temporal_align,
    to_stgcn_input_from_coco17,
    to_stgcn_input_from_penn13,
)


def build_data_tensors(
    labels_dir: str,
    dataset_format: str = 'penn',
    exercise_classes: List[str] = EXERCISE_CLASSES,
    class_to_id: Dict[str, int] = CLASS_TO_ID,
    target_frames: int = TARGET_FRAMES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], List[str]]:
    """
    Load and preprocess keypoint files into unified ST-GCN tensors.

    Parameters
    ----------
    labels_dir      : input directory (Penn .mat or COCO .npz)
    dataset_format  : 'penn', 'coco', or 'gym288'
    exercise_classes: class whitelist
    class_to_id     : class mapping
    target_frames   : temporal alignment length

    Returns
    -------
    data             : float32 ndarray  (N, 2, target_frames, 14, 1)
    labels           : int64 ndarray    (N,)
    flags            : int8 ndarray     (N,)  (1=train, 0=test)
    raw_frame_counts : list[int]
    video_ids        : list[str]
    """
    fmt = dataset_format.strip().lower()
    if fmt == 'penn':
        return build_penn_data_tensors(
            labels_dir=labels_dir,
            exercise_classes=exercise_classes,
            class_to_id=class_to_id,
            target_frames=target_frames,
        )
    if fmt == 'coco':
        return build_coco_data_tensors(
            labels_dir=labels_dir,
            exercise_classes=exercise_classes,
            class_to_id=class_to_id,
            target_frames=target_frames,
        )
    if fmt == 'gym288':
        return build_gym288_data_tensors(
            dataset_path=labels_dir,
            target_frames=target_frames,
            split='all',
            keep_unknown_split=False,
        )

    raise ValueError(
        f"Unsupported dataset_format='{dataset_format}'. Expected one of: 'penn', 'coco', 'gym288'."
    )


class PennActionDataset(Dataset):
    """
    Wrap pre-built arrays as a PyTorch Dataset.

    Parameters
    ----------
    data   : float32 ndarray  (N, 2, T, 14, 1)
    labels : int64 ndarray    (N,)
    """

    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.data[idx], self.labels[idx]


# Backward-compatible alias for clearer naming in future upgrades.
COCOKeypointsDataset = PennActionDataset
