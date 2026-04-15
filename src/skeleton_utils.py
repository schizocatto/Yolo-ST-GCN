"""
skeleton_utils.py
Shared keypoint preprocessing helpers for Penn/COCO data sources.
"""

from typing import List, Tuple

import numpy as np
import torch

from src.config import COCO_TO_PENN_IDX, TARGET_FRAMES


def add_virtual_center_joint(kpts: np.ndarray) -> np.ndarray:
    """
    Append virtual center joint (index 13) as mean of joints 1,2,7,8.

    Parameters
    ----------
    kpts : (T, 13, 2)
    """
    center = (kpts[:, 1, :] + kpts[:, 2, :] + kpts[:, 7, :] + kpts[:, 8, :]) / 4.0
    return np.concatenate((kpts, center[:, np.newaxis, :]), axis=1)


def temporal_align(kpts: np.ndarray, target_frames: int = TARGET_FRAMES) -> np.ndarray:
    """
    Uniformly resample a keypoint sequence to exactly target_frames.

    Parameters
    ----------
    kpts : (T, J, 2)
    """
    T = kpts.shape[0]
    if T == target_frames:
        return kpts
    indices = np.linspace(0, T - 1, target_frames).astype(int)
    return kpts[indices]


def remap_coco17_to_penn13(coco_kpts: np.ndarray) -> np.ndarray:
    """
    Remap COCO 17-joint keypoints to Penn 13-joint layout.

    Parameters
    ----------
    coco_kpts : (T, 17, 2)
    """
    return coco_kpts[:, COCO_TO_PENN_IDX, :]


def to_stgcn_input_from_penn13(
    kpts13: np.ndarray,
    target_frames: int = TARGET_FRAMES,
) -> np.ndarray:
    """
    Convert (T,13,2) keypoints into ST-GCN input layout (2,T,14,1).
    """
    kpts14 = add_virtual_center_joint(temporal_align(kpts13, target_frames))
    return np.expand_dims(np.transpose(kpts14, (2, 0, 1)), axis=-1).astype(np.float32)


def to_stgcn_input_from_coco17(
    coco_kpts: np.ndarray,
    target_frames: int = TARGET_FRAMES,
) -> np.ndarray:
    """
    Convert (T,17,2) COCO keypoints into ST-GCN input layout (2,T,14,1).
    """
    return to_stgcn_input_from_penn13(remap_coco17_to_penn13(coco_kpts), target_frames)


def ensure_t_j_2(kpts: np.ndarray, expected_joints: int) -> np.ndarray:
    """
    Normalize keypoint arrays to (T, J, 2) for common upstream formats.

    Accepted common shapes include:
    - (T, J, 2)
    - (J, 2, T)
    - (2, T, J)
    - (T, 2, J)
    - (J, T, 2)
    """
    arr = np.asarray(kpts, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f'Expected 3D keypoints array, got shape {arr.shape}')

    if arr.shape[2] == 2 and arr.shape[1] == expected_joints:
        return arr
    if arr.shape[0] == expected_joints and arr.shape[1] == 2:
        return np.transpose(arr, (2, 0, 1))
    if arr.shape[0] == 2 and arr.shape[2] == expected_joints:
        return np.transpose(arr, (1, 2, 0))
    if arr.shape[1] == 2 and arr.shape[2] == expected_joints:
        return np.transpose(arr, (0, 2, 1))
    if arr.shape[0] == expected_joints and arr.shape[2] == 2:
        return np.transpose(arr, (1, 0, 2))

    raise ValueError(
        f'Unsupported keypoints shape {arr.shape}; could not coerce to (T, {expected_joints}, 2)'
    )


def add_virtual_center_joint_coco17(kpts: np.ndarray) -> np.ndarray:
    """
    Append virtual center joint (index 17) as mean of COCO-17 joints 5,6,11,12
    (left_shoulder, right_shoulder, left_hip, right_hip).

    Parameters
    ----------
    kpts : (T, 17, 2)

    Returns
    -------
    (T, 18, 2)
    """
    center = (kpts[:, 5, :] + kpts[:, 6, :] + kpts[:, 11, :] + kpts[:, 12, :]) / 4.0
    return np.concatenate((kpts, center[:, np.newaxis, :]), axis=1)


def to_stgcn_input_from_coco17_full(
    coco_kpts: np.ndarray,
    target_frames: int = TARGET_FRAMES,
) -> np.ndarray:
    """
    Convert (T, 17, 2) COCO-17 keypoints to ST-GCN input (2, T, 18, 1),
    keeping ALL 17 joints (including face landmarks) and appending a
    virtual center joint (joint 17 = mean of shoulders + hips).
    """
    kpts17 = temporal_align(coco_kpts, target_frames)
    kpts18 = add_virtual_center_joint_coco17(kpts17)
    return np.expand_dims(np.transpose(kpts18, (2, 0, 1)), axis=-1).astype(np.float32)


def calculate_bone_data(
    joint_data,
    bone_pairs: List[Tuple[int, int]],
):
    """
    Build bone tensor from joint tensor via (child - parent) for each bone pair.

    Supported shapes:
    - (C, T, V, M)
    - (N, C, T, V, M)

    Returns tensor/array with same shape and dtype as input.
    """
    if torch.is_tensor(joint_data):
        bone_data = torch.zeros_like(joint_data)
        if joint_data.dim() == 4:
            for parent_idx, child_idx in bone_pairs:
                bone_data[:, :, child_idx, :] = (
                    joint_data[:, :, child_idx, :] - joint_data[:, :, parent_idx, :]
                )
            return bone_data
        if joint_data.dim() == 5:
            for parent_idx, child_idx in bone_pairs:
                bone_data[:, :, :, child_idx, :] = (
                    joint_data[:, :, :, child_idx, :] - joint_data[:, :, :, parent_idx, :]
                )
            return bone_data
        raise ValueError(f'Unsupported torch joint_data shape: {tuple(joint_data.shape)}')

    arr = np.asarray(joint_data)
    bone_data = np.zeros_like(arr)
    if arr.ndim == 4:
        for parent_idx, child_idx in bone_pairs:
            bone_data[:, :, child_idx, :] = arr[:, :, child_idx, :] - arr[:, :, parent_idx, :]
        return bone_data
    if arr.ndim == 5:
        for parent_idx, child_idx in bone_pairs:
            bone_data[:, :, :, child_idx, :] = arr[:, :, :, child_idx, :] - arr[:, :, :, parent_idx, :]
        return bone_data
    raise ValueError(f'Unsupported numpy joint_data shape: {arr.shape}')
