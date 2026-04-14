"""
dataset.py
Data loading, preprocessing, and PyTorch Dataset for Penn Action.

Key functions
-------------
load_mat_index(labels_dir)
    Scan .mat files → pandas DataFrame with columns [video_id, action, nframes].

build_data_tensors(labels_dir, class_to_id, target_frames)
    Load and preprocess all exercise .mat files into (N, 2, T, 14, 1) numpy arrays.

PennActionDataset
    PyTorch Dataset wrapping pre-built numpy arrays.
"""

import glob
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import scipy.io
import torch
from torch.utils.data import Dataset

from src.config import EXERCISE_CLASSES, CLASS_TO_ID, TARGET_FRAMES


# ---------------------------------------------------------------------------
# Low-level keypoint helpers
# ---------------------------------------------------------------------------

def add_virtual_center_joint(kpts: np.ndarray) -> np.ndarray:
    """
    Append a virtual center joint to a keypoint sequence.

    Parameters
    ----------
    kpts : (T, 13, 2)  — raw Penn Action keypoints

    Returns
    -------
    (T, 14, 2)  — with joint 13 = mean of l_sho, r_sho, l_hip, r_hip
    """
    center = (kpts[:, 1, :] + kpts[:, 2, :] + kpts[:, 7, :] + kpts[:, 8, :]) / 4.0
    return np.concatenate((kpts, center[:, np.newaxis, :]), axis=1)


def temporal_align(kpts: np.ndarray, target_frames: int = TARGET_FRAMES) -> np.ndarray:
    """
    Uniformly resample a keypoint sequence to exactly `target_frames` frames.

    Parameters
    ----------
    kpts          : (T, J, 2)
    target_frames : desired output length

    Returns
    -------
    (target_frames, J, 2)
    """
    T = kpts.shape[0]
    if T == target_frames:
        return kpts
    indices = np.linspace(0, T - 1, target_frames).astype(int)
    return kpts[indices]


# ---------------------------------------------------------------------------
# Dataset scanning
# ---------------------------------------------------------------------------

def load_mat_index(labels_dir: str) -> pd.DataFrame:
    """
    Scan all .mat files in `labels_dir` and return a DataFrame with
    columns: video_id, action, nframes.
    """
    records = []
    for mat_path in sorted(glob.glob(os.path.join(labels_dir, '*.mat'))):
        video_id    = os.path.basename(mat_path).replace('.mat', '')
        mat_content = scipy.io.loadmat(mat_path)

        action = mat_content['action'][0]
        if isinstance(action, np.ndarray) and len(action) > 0:
            action = action[0]
        nframes = int(mat_content['nframes'][0][0])
        records.append({'video_id': video_id, 'action': str(action), 'nframes': nframes})

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------

def build_data_tensors(
    labels_dir: str,
    exercise_classes: List[str] = EXERCISE_CLASSES,
    class_to_id: dict = CLASS_TO_ID,
    target_frames: int = TARGET_FRAMES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """
    Load, preprocess and stack all exercise-class .mat files.

    Uses the official Penn Action ``train`` field (1 = train, 0 = test) for
    subject-isolated splitting.  Do NOT use sklearn train_test_split on the
    returned arrays — split by the returned ``flags`` array instead.

    Returns
    -------
    data             : float32 ndarray  (N, 2, target_frames, 14, 1)
    labels           : int64 ndarray    (N,)
    flags            : int8 ndarray     (N,)  — 1 = official train, 0 = official test
    raw_frame_counts : list of original video lengths (before alignment)
    video_ids        : list of video ID strings (stem of each .mat filename)
    """
    all_data, all_labels, all_flags, raw_frame_counts = [], [], [], []
    all_video_ids, all_actions = [], []

    for mat_path in sorted(glob.glob(os.path.join(labels_dir, '*.mat'))):
        try:
            mat_data = scipy.io.loadmat(mat_path)
            action   = mat_data['action'][0]
            if isinstance(action, np.ndarray) and len(action) > 0:
                action = action[0]
            action = str(action)
            if action not in exercise_classes:
                continue

            train_flag = int(mat_data['train'][0][0]) if 'train' in mat_data else 1

            # Stack x, y → (T, 13, 2)
            kpts = np.stack((mat_data['x'], mat_data['y']), axis=-1).astype(np.float32)
            raw_frame_counts.append(kpts.shape[0])

            # Align temporally, add virtual center joint
            kpts = add_virtual_center_joint(temporal_align(kpts, target_frames))

            # Reshape to (2, T, 14, 1) to match model input format
            tensor = np.expand_dims(np.transpose(kpts, (2, 0, 1)), axis=-1)
            all_data.append(tensor)
            all_labels.append(class_to_id[action])
            all_flags.append(train_flag)
            all_video_ids.append(os.path.splitext(os.path.basename(mat_path))[0])
            all_actions.append(action)

        except Exception as exc:
            print(f"  [skip] {os.path.basename(mat_path)}: {exc}")

    data   = np.array(all_data,   dtype=np.float32)   # (N, 2, T, 14, 1)
    labels = np.array(all_labels, dtype=np.int64)
    flags  = np.array(all_flags,  dtype=np.int8)

    # If the dataset has no official test split (all flags == 1), fall back to a
    # subject-isolated split: within each class, videos are ordered by subject
    # (contiguous IDs = same subject), so taking the last 30% per class as test
    # avoids the same subject appearing in both partitions.
    if (flags == 0).sum() == 0:
        print('[dataset] No official test split found — applying per-class subject-isolated split (last 30% per class = test).')
        flags = np.ones(len(flags), dtype=np.int8)
        for cls in exercise_classes:
            idx = [i for i, a in enumerate(all_actions) if a == cls]
            if not idx:
                continue
            n_test = max(1, int(round(len(idx) * 0.3)))
            for i in idx[-n_test:]:
                flags[i] = 0

    return data, labels, flags, raw_frame_counts, all_video_ids


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class PennActionDataset(Dataset):
    """
    Wraps pre-built numpy arrays as a PyTorch Dataset.

    Parameters
    ----------
    data   : float32 ndarray  (N, 2, T, 14, 1)
    labels : int64 ndarray    (N,)
    """

    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data   = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]
