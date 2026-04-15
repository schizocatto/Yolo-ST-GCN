"""
coco_dataset.py
COCO keypoint dataset loading utilities with automatic remap to Penn format.

Expected input directory layout:
    labels_dir/
      sample_0001.npz
      sample_0002.npz
      ...

Each .npz file must contain:
    keypoints : array with shape convertible to (T,17,2)

Optional fields:
    action     : class name (string)
    train      : 1/0 split flag
    video_id   : identifier string
"""

import glob
import os
from typing import Dict, List, Tuple

import numpy as np

from src.config import CLASS_TO_ID, EXERCISE_CLASSES, PENN_BONE_PAIRS_14, TARGET_FRAMES
from src.skeleton_utils import calculate_bone_data, ensure_t_j_2, to_stgcn_input_from_coco17


def _to_scalar_string(value, default: str = '') -> str:
    arr = np.asarray(value)
    if arr.size == 0:
        return default
    return str(arr.reshape(-1)[0])


def _to_scalar_int(value, default: int = 1) -> int:
    arr = np.asarray(value)
    if arr.size == 0:
        return default
    return int(arr.reshape(-1)[0])


def build_coco_data_tensors(
    labels_dir: str,
    exercise_classes: List[str] = EXERCISE_CLASSES,
    class_to_id: Dict[str, int] = CLASS_TO_ID,
    target_frames: int = TARGET_FRAMES,
    return_bone_data: bool = False,
    bone_pairs: List[Tuple[int, int]] = PENN_BONE_PAIRS_14,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], List[str]]:
    """
    Load COCO-keypoint .npz files and convert them to ST-GCN tensors.

    Returns
    -------
    data             : float32 ndarray  (N, 2, target_frames, 14, 1)
    labels           : int64 ndarray    (N,)
    flags            : int8 ndarray     (N,)  (1=train, 0=test)
    raw_frame_counts : list[int]
    video_ids        : list[str]
    """
    all_data, all_bone_data, all_labels, all_flags, raw_frame_counts = [], [], [], [], []
    all_video_ids = []

    npz_paths = sorted(glob.glob(os.path.join(labels_dir, '*.npz')))
    if not npz_paths:
        print(f'[dataset] No .npz files found in {labels_dir} for COCO dataset format.')

    for npz_path in npz_paths:
        try:
            payload = np.load(npz_path, allow_pickle=True)
            if 'keypoints' not in payload:
                raise KeyError("Missing required key 'keypoints'")

            action = _to_scalar_string(payload['action'], default='') if 'action' in payload else ''
            if action not in exercise_classes:
                continue

            train_flag = _to_scalar_int(payload['train'], default=1) if 'train' in payload else 1
            video_id = (
                _to_scalar_string(payload['video_id'])
                if 'video_id' in payload
                else os.path.splitext(os.path.basename(npz_path))[0]
            )

            kpts17 = ensure_t_j_2(payload['keypoints'], expected_joints=17)
            raw_frame_counts.append(int(kpts17.shape[0]))

            tensor = to_stgcn_input_from_coco17(kpts17, target_frames)

            all_data.append(tensor)
            if return_bone_data:
                all_bone_data.append(calculate_bone_data(tensor, bone_pairs).astype(np.float32))
            all_labels.append(class_to_id[action])
            all_flags.append(train_flag)
            all_video_ids.append(video_id)

        except Exception as exc:
            print(f'  [skip] {os.path.basename(npz_path)}: {exc}')

    data = np.array(all_data, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int64)
    flags = np.array(all_flags, dtype=np.int8)

    # Generic fallback split: last 30% per class if test split is missing.
    if (flags == 0).sum() == 0 and len(flags) > 0:
        print('[dataset] No test split found for COCO data; using last 30% per class as test.')
        flags = np.ones(len(flags), dtype=np.int8)
        for cls in exercise_classes:
            idx = [i for i, y in enumerate(all_labels) if y == class_to_id[cls]]
            if not idx:
                continue
            n_test = max(1, int(round(len(idx) * 0.3)))
            for i in idx[-n_test:]:
                flags[i] = 0

    if return_bone_data:
        bone_data = np.array(all_bone_data, dtype=np.float32)
        return data, bone_data, labels, flags, raw_frame_counts, all_video_ids

    return data, labels, flags, raw_frame_counts, all_video_ids
