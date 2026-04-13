"""
inference.py
YOLO + ST-GCN inference helpers.

Functions
---------
select_best_person(results)
    Pick the largest-bounding-box detection from a YOLO result.

compute_iou(boxA, boxB)
    Intersection-over-Union between two (x1,y1,x2,y2) boxes.

extract_yolo_keypoints(frames_dir, video_id, model_yolo, target_frames, iou_threshold)
    Run YOLOv8-pose on a video with single-person selection and IoU tracking.

run_stgcn_inference(kpts_aligned, model_stgcn, device)
    Feed preprocessed keypoints through ST-GCN.

load_stgcn_weights(model, weights_path, device)
    Load checkpoint with legacy key remapping.
"""

import glob
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.config import COCO_TO_PENN_IDX, TARGET_FRAMES
from src.dataset import add_virtual_center_joint, temporal_align


# ---------------------------------------------------------------------------
# Single-person selection helpers
# ---------------------------------------------------------------------------

def select_best_person(results) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    From a YOLO result, pick the detection with the largest bounding box area.

    Parameters
    ----------
    results : ultralytics Results object (single frame)

    Returns
    -------
    kp13 : (13, 2) Penn Action keypoints, or None if no detection
    bbox : (4,) array [x1, y1, x2, y2], or None if no detection
    """
    boxes = results[0].boxes
    kps   = results[0].keypoints

    if boxes is None or kps is None or len(boxes) == 0:
        return None, None

    xyxy   = boxes.xyxy.cpu().numpy()    # (D, 4)
    areas  = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    best   = int(np.argmax(areas))

    kp17 = kps.xy[best].cpu().numpy()   # (17, 2)  COCO keypoints
    kp13 = kp17[COCO_TO_PENN_IDX]       # (13, 2)  Penn Action joints
    return kp13, xyxy[best]


def compute_iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
    """
    Compute Intersection-over-Union between two bounding boxes.

    Parameters
    ----------
    boxA, boxB : (4,) arrays [x1, y1, x2, y2]

    Returns
    -------
    iou : float in [0, 1]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    if inter == 0.0:
        return 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter + 1e-6)


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract_yolo_keypoints(
    frames_dir: str,
    video_id: str,
    model_yolo,
    target_frames: int = TARGET_FRAMES,
    iou_threshold: float = 0.3,
) -> Tuple[Optional[np.ndarray], List[str]]:
    """
    Run YOLOv8-pose on every JPEG frame of a video with single-person
    selection and IoU-based frame-to-frame tracking.

    Strategy
    --------
    - Frame 0  : pick the detection with the largest bounding box area.
    - Frame 1+ : pick the detection with the highest IoU to the previous
                 frame's bounding box.  Falls back to largest-bbox if no
                 candidate exceeds `iou_threshold`.

    Parameters
    ----------
    frames_dir    : root directory containing per-video frame subdirectories
    video_id      : subdirectory name (e.g. "0001")
    model_yolo    : loaded ultralytics YOLO model
    target_frames : desired output sequence length
    iou_threshold : minimum IoU to accept a tracked detection (default 0.3)

    Returns
    -------
    kpts_aligned : (target_frames, 14, 2)  or None if no frames found
    frame_paths  : sorted list of JPEG paths (empty if none found)
    """
    frame_paths = sorted(glob.glob(os.path.join(frames_dir, video_id, '*.jpg')))
    if not frame_paths:
        return None, []

    kpts_seq   = []
    prev_bbox  = None   # (4,) bbox of the tracked person in the previous frame

    for fp in frame_paths:
        results = model_yolo(fp, verbose=False)
        boxes   = results[0].boxes
        kps     = results[0].keypoints

        kp13 = np.zeros((13, 2), dtype=np.float32)

        if boxes is not None and kps is not None and len(boxes) > 0:
            xyxy  = boxes.xyxy.cpu().numpy()   # (D, 4)
            kp_all = kps.xy.cpu().numpy()       # (D, 17, 2)

            if prev_bbox is None:
                # Frame 0: pick largest bbox
                areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
                best  = int(np.argmax(areas))
            else:
                # Subsequent frames: pick best IoU with previous bbox
                ious = np.array([compute_iou(prev_bbox, b) for b in xyxy])
                best = int(np.argmax(ious))
                if ious[best] < iou_threshold:
                    # IoU too low — fall back to largest bbox
                    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
                    best  = int(np.argmax(areas))

            kp13      = kp_all[best][COCO_TO_PENN_IDX]   # (13, 2)
            prev_bbox = xyxy[best]

        kpts_seq.append(kp13)

    kpts    = np.array(kpts_seq, dtype=np.float32)    # (T, 13, 2)
    kpts14  = add_virtual_center_joint(kpts)           # (T, 14, 2)
    aligned = temporal_align(kpts14, target_frames)    # (target_frames, 14, 2)
    return aligned, frame_paths


# ---------------------------------------------------------------------------
# ST-GCN inference
# ---------------------------------------------------------------------------

def run_stgcn_inference(
    kpts_aligned: np.ndarray,
    model_stgcn: nn.Module,
    device: torch.device,
) -> Tuple[int, np.ndarray]:
    """
    Run ST-GCN inference on pre-processed keypoints.

    Parameters
    ----------
    kpts_aligned : (target_frames, 14, 2)
    model_stgcn  : loaded Model_STGCN in eval mode
    device       : torch device

    Returns
    -------
    pred_idx : int             — argmax class index
    probs    : (num_classes,)  — softmax probabilities
    """
    tensor = np.transpose(kpts_aligned, (2, 0, 1))     # (2, T, 14)
    tensor = np.expand_dims(tensor, axis=(0, -1))       # (1, 2, T, 14, 1)
    x      = torch.FloatTensor(tensor).to(device)

    with torch.no_grad():
        logits = model_stgcn(x)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

    return int(np.argmax(probs)), probs


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def load_stgcn_weights(model: nn.Module, weights_path: str, device: torch.device) -> nn.Module:
    """
    Load ST-GCN weights, remapping legacy key names if necessary.

    The training notebook used ``gcn_conv`` / ``tcn_conv`` / ``residual``;
    the current model uses ``gcn`` / ``tcn`` / ``res``.

    Returns model in eval mode.
    """
    raw = torch.load(weights_path, map_location=device)

    remap = {'gcn_conv': 'gcn', 'tcn_conv': 'tcn', 'residual': 'res'}
    state_dict = {}
    for k, v in raw.items():
        new_k = k
        for old, new in remap.items():
            new_k = new_k.replace(old, new)
        state_dict[new_k] = v

    model.load_state_dict(state_dict)
    return model.eval()
