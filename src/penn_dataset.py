"""
penn_dataset.py
Penn Action dataset indexing and tensor building utilities.
"""

import glob
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.io

from src.config import CLASS_TO_ID, EXERCISE_CLASSES, TARGET_FRAMES
from src.skeleton_utils import to_stgcn_input_from_penn13


def load_mat_index(labels_dir: str) -> pd.DataFrame:
    """
    Scan all .mat files in labels_dir and return DataFrame columns:
    video_id, action, nframes.
    """
    records = []
    for mat_path in sorted(glob.glob(os.path.join(labels_dir, '*.mat'))):
        video_id = os.path.basename(mat_path).replace('.mat', '')
        mat_content = scipy.io.loadmat(mat_path)

        action = mat_content['action'][0]
        if isinstance(action, np.ndarray) and len(action) > 0:
            action = action[0]
        nframes = int(mat_content['nframes'][0][0])
        records.append({'video_id': video_id, 'action': str(action), 'nframes': nframes})

    return pd.DataFrame(records)


def build_penn_data_tensors(
    labels_dir: str,
    exercise_classes: List[str] = EXERCISE_CLASSES,
    class_to_id: Dict[str, int] = CLASS_TO_ID,
    target_frames: int = TARGET_FRAMES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], List[str]]:
    """
    Load Penn Action .mat files into ST-GCN tensors.

    Returns
    -------
    data             : float32 ndarray  (N, 2, target_frames, 14, 1)
    labels           : int64 ndarray    (N,)
    flags            : int8 ndarray     (N,)  (1=train, 0=test)
    raw_frame_counts : list[int]
    video_ids        : list[str]
    """
    all_data, all_labels, all_flags, raw_frame_counts = [], [], [], []
    all_video_ids, all_actions = [], []

    for mat_path in sorted(glob.glob(os.path.join(labels_dir, '*.mat'))):
        try:
            mat_data = scipy.io.loadmat(mat_path)
            action = mat_data['action'][0]
            if isinstance(action, np.ndarray) and len(action) > 0:
                action = action[0]
            action = str(action)
            if action not in exercise_classes:
                continue

            train_flag = int(mat_data['train'][0][0]) if 'train' in mat_data else 1

            # Stack x,y to (T,13,2) then convert to ST-GCN tensor format.
            kpts13 = np.stack((mat_data['x'], mat_data['y']), axis=-1).astype(np.float32)
            raw_frame_counts.append(int(kpts13.shape[0]))

            tensor = to_stgcn_input_from_penn13(kpts13, target_frames)
            all_data.append(tensor)
            all_labels.append(class_to_id[action])
            all_flags.append(train_flag)
            all_video_ids.append(os.path.splitext(os.path.basename(mat_path))[0])
            all_actions.append(action)

        except Exception as exc:
            print(f'  [skip] {os.path.basename(mat_path)}: {exc}')

    data = np.array(all_data, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int64)
    flags = np.array(all_flags, dtype=np.int8)

    # Fallback split if no official test partition exists.
    if (flags == 0).sum() == 0 and len(flags) > 0:
        print('[dataset] No official test split found for Penn data; using last 30% per class as test.')
        flags = np.ones(len(flags), dtype=np.int8)
        for cls in exercise_classes:
            idx = [i for i, a in enumerate(all_actions) if a == cls]
            if not idx:
                continue
            n_test = max(1, int(round(len(idx) * 0.3)))
            for i in idx[-n_test:]:
                flags[i] = 0

    return data, labels, flags, raw_frame_counts, all_video_ids
