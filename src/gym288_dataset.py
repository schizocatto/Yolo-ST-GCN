"""
gym288_dataset.py
Gym288-skeleton dataset loading utilities.

The dataset is expected to be a pickle file containing:
    {
      'split': {'train': [frame_dir...], 'test': [frame_dir...]},
      'annotations': [
        {
          'frame_dir': str,
          'label': int,
          'keypoint': np.ndarray (1, T, 17, 2),
          ...
        },
        ...
      ]
    }
"""

import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.config import TARGET_FRAMES
from src.skeleton_utils import ensure_t_j_2, to_stgcn_input_from_coco17


def _extract_coco17_xy(annotation: Dict) -> np.ndarray:
    """Extract keypoints from a Gym288 annotation as (T, 17, 2)."""
    if 'keypoint' in annotation:
        kp = np.asarray(annotation['keypoint'], dtype=np.float32)
        if kp.ndim == 4 and kp.shape[0] == 1:
            kp = kp[0]
        return ensure_t_j_2(kp, expected_joints=17)

    if 'kp_w_gt' in annotation:
        kp_w_gt = np.asarray(annotation['kp_w_gt'], dtype=np.float32)
        if kp_w_gt.ndim != 3 or kp_w_gt.shape[-1] < 2:
            raise ValueError(f"Unsupported 'kp_w_gt' shape: {kp_w_gt.shape}")
        return ensure_t_j_2(kp_w_gt[..., :2], expected_joints=17)

    raise KeyError("Annotation must contain 'keypoint' or 'kp_w_gt'")


def build_gym288_data_tensors(
    dataset_path: str,
    target_frames: int = TARGET_FRAMES,
    split: str = 'all',
    keep_unknown_split: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], List[str]]:
    """
    Build ST-GCN tensors from Gym288-skeleton pickle.

    Parameters
    ----------
    dataset_path       : path to gym288_skeleton.pkl
    target_frames      : temporal alignment length
    split              : 'all' | 'train' | 'test'
    keep_unknown_split : include samples not present in split lists (flag=-1)

    Returns
    -------
    data             : float32 ndarray  (N, 2, target_frames, 14, 1)
    labels           : int64 ndarray    (N,)
    flags            : int8 ndarray     (N,)  (1=train, 0=test, -1=unknown)
    raw_frame_counts : list[int]
    video_ids        : list[str]
    """
    split_norm = split.strip().lower()
    if split_norm not in {'all', 'train', 'test'}:
        raise ValueError("split must be one of: 'all', 'train', 'test'")

    with open(dataset_path, 'rb') as f:
        payload = pickle.load(f)

    split_info = payload.get('split', {})
    train_ids = set(split_info.get('train', []))
    test_ids = set(split_info.get('test', []))
    annotations = payload.get('annotations', [])

    all_data: List[np.ndarray] = []
    all_labels: List[int] = []
    all_flags: List[int] = []
    raw_frame_counts: List[int] = []
    all_video_ids: List[str] = []

    for ann in annotations:
        try:
            video_id = str(ann['frame_dir'])
            label = int(ann['label'])
            kpts17 = _extract_coco17_xy(ann)

            if video_id in train_ids:
                flag = 1
            elif video_id in test_ids:
                flag = 0
            else:
                flag = -1

            if split_norm == 'train' and flag != 1:
                continue
            if split_norm == 'test' and flag != 0:
                continue
            if split_norm == 'all' and (flag == -1 and not keep_unknown_split):
                continue

            raw_frame_counts.append(int(kpts17.shape[0]))
            tensor = to_stgcn_input_from_coco17(kpts17, target_frames)

            all_data.append(tensor)
            all_labels.append(label)
            all_flags.append(flag)
            all_video_ids.append(video_id)

        except Exception as exc:
            vid = ann.get('frame_dir', '<unknown>')
            print(f'  [skip] {vid}: {exc}')

    data = np.array(all_data, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int64)
    flags = np.array(all_flags, dtype=np.int8)
    return data, labels, flags, raw_frame_counts, all_video_ids


def infer_num_gym288_classes(dataset_path: str, fallback: int = 288) -> int:
    """Infer number of classes from annotations labels in dataset pickle."""
    with open(dataset_path, 'rb') as f:
        payload = pickle.load(f)
    annotations = payload.get('annotations', [])
    if not annotations:
        return fallback

    labels = [int(ann['label']) for ann in annotations if 'label' in ann]
    if not labels:
        return fallback
    return max(labels) + 1
