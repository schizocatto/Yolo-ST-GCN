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

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import CLASS_TO_ID, EXERCISE_CLASSES, PENN_BONE_PAIRS_14, TARGET_FRAMES
from src.coco_dataset import build_coco_data_tensors
from src.gym288_dataset import build_gym288_data_tensors
from src.gym99_dataset import build_gym99_data_tensors
from src.penn_dataset import build_penn_data_tensors, load_mat_index
from src.skeleton_utils import (
    add_virtual_center_joint,
    calculate_bone_data,
    remap_coco17_to_penn13,
    temporal_align,
    to_stgcn_input_from_coco17,
    to_stgcn_input_from_penn13,
)
from src.joint_specs import get_joint_spec


def build_data_tensors(
    labels_dir: str,
    dataset_format: str = 'penn',
    exercise_classes: List[str] = EXERCISE_CLASSES,
    class_to_id: Dict[str, int] = CLASS_TO_ID,
    target_frames: int = TARGET_FRAMES,
    joint_spec_name: str = 'penn14',
    return_bone_data: bool = False,
    bone_pairs: List[Tuple[int, int]] = PENN_BONE_PAIRS_14,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], List[str]]:
    """
    Load and preprocess keypoint files into unified ST-GCN tensors.

    Parameters
    ----------
    labels_dir      : input directory (Penn .mat or COCO .npz)
    dataset_format  : 'penn', 'coco', 'gym288', or 'gym99'
    exercise_classes: class whitelist
    class_to_id     : class mapping
    target_frames   : temporal alignment length

    Returns
    -------
    data             : float32 ndarray  (N, 2, target_frames, 14, 1)
    bone_data        : float32 ndarray  (N, 2, target_frames, 14, 1), optional
    labels           : int64 ndarray    (N,)
    flags            : int8 ndarray     (N,)  (1=train, 0=test)
    raw_frame_counts : list[int]
    video_ids        : list[str]
    """
    fmt = dataset_format.strip().lower()
    spec = get_joint_spec(joint_spec_name)
    if bone_pairs is PENN_BONE_PAIRS_14:
        bone_pairs = spec.bone_pairs
    if fmt == 'penn':
        return build_penn_data_tensors(
            labels_dir=labels_dir,
            exercise_classes=exercise_classes,
            class_to_id=class_to_id,
            target_frames=target_frames,
            return_bone_data=return_bone_data,
            bone_pairs=bone_pairs,
        )
    if fmt == 'coco':
        return build_coco_data_tensors(
            labels_dir=labels_dir,
            exercise_classes=exercise_classes,
            class_to_id=class_to_id,
            target_frames=target_frames,
            return_bone_data=return_bone_data,
            bone_pairs=bone_pairs,
        )
    if fmt == 'gym288':
        return build_gym288_data_tensors(
            dataset_path=labels_dir,
            target_frames=target_frames,
            joint_spec_name=joint_spec_name,
            split='all',
            keep_unknown_split=False,
            return_bone_data=return_bone_data,
            bone_pairs=bone_pairs,
        )
    if fmt == 'gym99':
        return build_gym99_data_tensors(
            dataset_path=labels_dir,
            target_frames=target_frames,
            joint_spec_name=joint_spec_name,
            split='all',
            keep_unknown_split=False,
            return_bone_data=return_bone_data,
            bone_pairs=bone_pairs,
        )

    raise ValueError(
        f"Unsupported dataset_format='{dataset_format}'. Expected one of: 'penn', 'coco', 'gym288', 'gym99'."
    )


class PennActionDataset(Dataset):
    """
    Wrap pre-built arrays as a PyTorch Dataset.

    Parameters
    ----------
    data   : float32 ndarray  (N, 2, T, 14, 1)
    labels : int64 ndarray    (N,)
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        bone_data: Optional[np.ndarray] = None,
        include_bone: bool = False,
        bone_pairs: Optional[List[Tuple[int, int]]] = None,
        joint_spec_name: str = 'penn14',
    ):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.include_bone = include_bone
        self.bone_pairs = bone_pairs or get_joint_spec(joint_spec_name).bone_pairs

        self.bone_data: Optional[torch.Tensor] = None
        if include_bone:
            if bone_data is not None:
                self.bone_data = torch.FloatTensor(bone_data)
            else:
                self.bone_data = calculate_bone_data(self.data, self.bone_pairs)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        if self.include_bone:
            assert self.bone_data is not None
            return (self.data[idx], self.bone_data[idx]), self.labels[idx]
        return self.data[idx], self.labels[idx]


# Backward-compatible alias for clearer naming in future upgrades.
COCOKeypointsDataset = PennActionDataset
