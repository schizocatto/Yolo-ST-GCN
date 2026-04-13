"""
scripts/evaluate.py
Full validation-set evaluation of the YOLO + ST-GCN pipeline.

Reproduces the same 80/20 split used during training (random_state=42, stratified)
and reports accuracy, macro-F1, and a confusion matrix.

Usage
-----
python scripts/evaluate.py \
    --labels_dir /path/to/Penn_Action/labels \
    --frames_dir /path/to/Penn_Action/frames \
    --weights    outputs/stgcn_penn_action.pth \
    --out_dir    outputs/
"""

import argparse
import glob
import os
import sys

import numpy as np
import scipy.io
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import EXERCISE_CLASSES, CLASS_TO_ID
from src.inference import extract_yolo_keypoints, run_stgcn_inference, load_stgcn_weights
from src.model import Model_STGCN
from src.visualize import plot_confusion_matrix, plot_keypoint_quality, plot_per_class_f1


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate YOLO + ST-GCN pipeline')
    p.add_argument('--labels_dir', required=True)
    p.add_argument('--frames_dir', required=True)
    p.add_argument('--weights',    required=True)
    p.add_argument('--out_dir',    default='outputs')
    return p.parse_args()


def build_test_file_list(labels_dir: str):
    """
    Return (mat_paths, labels) for official test-set exercise videos only.
    Uses the Penn Action ``train`` flag (0 = official test partition).
    """
    mat_paths, labels = [], []
    for mf in sorted(glob.glob(os.path.join(labels_dir, '*.mat'))):
        md     = scipy.io.loadmat(mf)
        action = md['action'][0]
        if isinstance(action, np.ndarray):
            action = action[0]
        action = str(action)
        if action not in EXERCISE_CLASSES:
            continue
        train_flag = int(md['train'][0][0]) if 'train' in md else 1
        if train_flag == 0:   # official test set only
            mat_paths.append(mf)
            labels.append(CLASS_TO_ID[action])
    return mat_paths, labels


def evaluate_keypoint_quality(mat_files_all, frames_dir, labels_dir, model_yolo):
    """Compute per-joint YOLO keypoint error normalised by person height."""
    from src.dataset import temporal_align
    from src.config import COCO_TO_PENN_IDX

    per_video_errors, n_skipped = [], 0

    for mat_file in tqdm(mat_files_all, desc='Keypoint quality'):
        md     = scipy.io.loadmat(mat_file)
        action = md['action'][0]
        if isinstance(action, np.ndarray):
            action = action[0]
        if str(action) not in EXERCISE_CLASSES:
            continue

        vid_id      = os.path.splitext(os.path.basename(mat_file))[0]
        frame_paths = sorted(glob.glob(os.path.join(frames_dir, vid_id, '*.jpg')))
        if not frame_paths:
            n_skipped += 1
            continue

        gt_kpts   = np.stack((md['x'], md['y']), axis=-1)
        gt_aligned = temporal_align(gt_kpts, 64)

        yolo_seq = []
        for fp in frame_paths:
            res = model_yolo(fp, verbose=False)
            if res[0].keypoints is not None and len(res[0].keypoints.xy) > 0:
                kp17 = res[0].keypoints.xy[0].cpu().numpy()
                yolo_seq.append(kp17[COCO_TO_PENN_IDX])
            else:
                yolo_seq.append(np.zeros((13, 2)))

        yolo_aligned = temporal_align(np.array(yolo_seq), 64)
        valid        = yolo_aligned.sum(axis=(1, 2)) > 0
        if valid.sum() < 10:
            n_skipped += 1
            continue

        gt_v, yolo_v = gt_aligned[valid], yolo_aligned[valid]
        head_y  = gt_v[:, 0, 1].mean()
        ankle_y = ((gt_v[:, 11, 1] + gt_v[:, 12, 1]) / 2).mean()
        height  = max(abs(ankle_y - head_y), 1.0)

        dist = np.sqrt(((yolo_v - gt_v) ** 2).sum(axis=-1))
        per_video_errors.append((dist / height).mean(axis=0))

    print(f'Keypoint eval — {len(per_video_errors)} videos, {n_skipped} skipped')
    return np.mean(per_video_errors, axis=0) if per_video_errors else None


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load models ───────────────────────────────────────────────────────
    from ultralytics import YOLO
    model_yolo  = YOLO('yolov8n-pose.pt')
    model_stgcn = load_stgcn_weights(Model_STGCN().to(device), args.weights, device)

    # ── Official test set (train flag = 0) ───────────────────────────────
    mat_val, lbl_val = build_test_file_list(args.labels_dir)
    print(f'Test set (official split): {len(mat_val)} videos')

    # ── Pipeline classification ───────────────────────────────────────────
    all_preds, all_gt, n_failed = [], [], 0
    for mat_file, gt_label in tqdm(zip(mat_val, lbl_val), total=len(mat_val),
                                   desc='Pipeline eval'):
        vid_id       = os.path.splitext(os.path.basename(mat_file))[0]
        kpts_aligned, _ = extract_yolo_keypoints(args.frames_dir, vid_id, model_yolo)
        if kpts_aligned is None:
            n_failed += 1
            continue
        pred_idx, _ = run_stgcn_inference(kpts_aligned, model_stgcn, device)
        all_preds.append(pred_idx)
        all_gt.append(gt_label)

    print(f'\nEvaluated: {len(all_preds)}  |  Failed (no frames): {n_failed}')
    print(f'Accuracy  : {accuracy_score(all_gt, all_preds):.4f}')
    print(f'Macro F1  : {f1_score(all_gt, all_preds, average="macro", zero_division=0):.4f}')
    print('\n--- Classification Report ---')
    print(classification_report(all_gt, all_preds,
                                target_names=EXERCISE_CLASSES, zero_division=0))

    plot_confusion_matrix(all_gt, all_preds,
                          title='Confusion Matrix — YOLO + ST-GCN Pipeline (Val Set)',
                          out_dir=args.out_dir,
                          filename='pipeline_confusion_matrix.png')
    plot_per_class_f1(all_gt, all_preds, out_dir=args.out_dir,
                      filename='pipeline_per_class_f1.png')

    # ── Keypoint quality (optional, slow) ─────────────────────────────────
    all_mat = sorted(glob.glob(os.path.join(args.labels_dir, '*.mat')))
    mean_errors = evaluate_keypoint_quality(all_mat, args.frames_dir, args.labels_dir, model_yolo)
    if mean_errors is not None:
        overall = float(mean_errors.mean())
        print(f'\nOverall normalised joint error: {overall:.4f}')
        plot_keypoint_quality(mean_errors, overall, out_dir=args.out_dir)


if __name__ == '__main__':
    main()
