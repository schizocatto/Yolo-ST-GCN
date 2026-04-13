"""
scripts/inference_demo.py
Run YOLO + ST-GCN on a single Penn Action video and visualise the result.

Usage
-----
python scripts/inference_demo.py \
    --labels_dir  /path/to/Penn_Action/labels \
    --frames_dir  /path/to/Penn_Action/frames \
    --weights     outputs/stgcn_penn_action.pth \
    --video_id    0001 \
    --out_dir     outputs/

If --video_id is omitted, the first available exercise video is used.
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np
import scipy.io
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import EXERCISE_CLASSES, COCO_TO_PENN_IDX
from src.model import Model_STGCN
from src.inference import extract_yolo_keypoints, run_stgcn_inference, load_stgcn_weights
from src.visualize import plot_inference_result


def parse_args():
    p = argparse.ArgumentParser(description='YOLO + ST-GCN inference demo')
    p.add_argument('--labels_dir', required=True)
    p.add_argument('--frames_dir', required=True)
    p.add_argument('--weights',    required=True)
    p.add_argument('--video_id',   default=None,
                   help='4-digit video ID (e.g. 0001). Auto-selects if omitted.')
    p.add_argument('--out_dir',    default='outputs')
    return p.parse_args()


def find_exercise_video(labels_dir: str, frames_dir: str) -> tuple:
    """Return (video_id, gt_action) for the first exercise video that has frames."""
    for mat_path in sorted(glob.glob(os.path.join(labels_dir, '*.mat'))):
        md     = scipy.io.loadmat(mat_path)
        action = md['action'][0]
        if isinstance(action, np.ndarray):
            action = action[0]
        action = str(action)
        if action not in EXERCISE_CLASSES:
            continue
        vid_id = os.path.splitext(os.path.basename(mat_path))[0]
        if glob.glob(os.path.join(frames_dir, vid_id, '*.jpg')):
            return vid_id, action
    raise RuntimeError('No exercise video with frames found.')


def get_gt_action(labels_dir: str, video_id: str) -> str:
    mat_path = os.path.join(labels_dir, f'{video_id}.mat')
    md       = scipy.io.loadmat(mat_path)
    action   = md['action'][0]
    if isinstance(action, np.ndarray):
        action = action[0]
    return str(action)


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Choose video ─────────────────────────────────────────────────────
    if args.video_id:
        video_id  = args.video_id
        gt_action = get_gt_action(args.labels_dir, video_id)
    else:
        video_id, gt_action = find_exercise_video(args.labels_dir, args.frames_dir)
    print(f'Video: {video_id}  |  GT action: {gt_action}')

    # ── Load models ───────────────────────────────────────────────────────
    from ultralytics import YOLO
    model_yolo  = YOLO('yolov8n-pose.pt')
    model_stgcn = load_stgcn_weights(Model_STGCN().to(device), args.weights, device)

    # ── Run pipeline ──────────────────────────────────────────────────────
    kpts_aligned, frame_paths = extract_yolo_keypoints(
        args.frames_dir, video_id, model_yolo
    )
    if kpts_aligned is None:
        print('No frames found. Exiting.')
        return

    pred_idx, probs = run_stgcn_inference(kpts_aligned, model_stgcn, device)
    pred_action     = EXERCISE_CLASSES[pred_idx]
    print(f'Predicted: {pred_action}  (GT: {gt_action})  '
          f'{"✓" if pred_action == gt_action else "✗"}')

    # ── Get first-frame skeleton for visualization ────────────────────────
    results = model_yolo(frame_paths[0], verbose=False)
    frame_rgb = cv2.cvtColor(cv2.imread(frame_paths[0]), cv2.COLOR_BGR2RGB)

    kp14 = np.zeros((14, 2), dtype=np.float32)
    if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
        kp17   = results[0].keypoints.xy[0].cpu().numpy()
        kp13   = kp17[COCO_TO_PENN_IDX]
        center = (kp13[1] + kp13[2] + kp13[7] + kp13[8]) / 4.0
        kp14   = np.vstack([kp13, center])

    plot_inference_result(
        frame_rgb, kp14, probs, pred_idx,
        video_id, gt_action, pred_action,
        out_dir=args.out_dir,
        filename=f'inference_{video_id}.png',
    )


if __name__ == '__main__':
    main()
