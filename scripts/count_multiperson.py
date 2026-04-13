"""
scripts/count_multiperson.py
Diagnostic: count frames where YOLO detects >= 2 people across all exercise videos.

Outputs a per-class breakdown to stdout and saves a JSON summary to
outputs/multiperson_stats.json for use in the report.

Usage
-----
python scripts/count_multiperson.py \
    --labels_dir /path/to/Penn_Action/labels \
    --frames_dir /path/to/Penn_Action/frames \
    --out_dir    outputs/
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import scipy.io
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import EXERCISE_CLASSES


def parse_args():
    p = argparse.ArgumentParser(description='Count multi-person YOLO detections')
    p.add_argument('--labels_dir', required=True)
    p.add_argument('--frames_dir', required=True)
    p.add_argument('--out_dir',    default='outputs')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from ultralytics import YOLO
    model_yolo = YOLO('yolov8n-pose.pt')

    # Per-class accumulators
    stats = defaultdict(lambda: {
        'total_videos': 0,
        'videos_with_multiperson': 0,
        'total_frames': 0,
        'frames_with_multiperson': 0,
    })

    mat_files = sorted(glob.glob(os.path.join(args.labels_dir, '*.mat')))

    for mat_path in tqdm(mat_files, desc='Scanning videos'):
        md = scipy.io.loadmat(mat_path)
        action = md['action'][0]
        if isinstance(action, np.ndarray):
            action = action[0]
        action = str(action)
        if action not in EXERCISE_CLASSES:
            continue

        vid_id      = os.path.splitext(os.path.basename(mat_path))[0]
        frame_paths = sorted(glob.glob(os.path.join(args.frames_dir, vid_id, '*.jpg')))
        if not frame_paths:
            continue

        s = stats[action]
        s['total_videos'] += 1
        video_has_multiperson = False

        for fp in frame_paths:
            results = model_yolo(fp, verbose=False)
            n_detections = len(results[0].boxes) if results[0].boxes is not None else 0
            s['total_frames'] += 1
            if n_detections >= 2:
                s['frames_with_multiperson'] += 1
                video_has_multiperson = True

        if video_has_multiperson:
            s['videos_with_multiperson'] += 1

    # ── Summary ───────────────────────────────────────────────────────────
    print('\n=== Multi-person Detection Summary ===\n')
    total_videos = total_vmp = total_frames = total_fmp = 0

    for action in EXERCISE_CLASSES:
        s = stats[action]
        v_pct = 100 * s['videos_with_multiperson'] / max(s['total_videos'], 1)
        f_pct = 100 * s['frames_with_multiperson'] / max(s['total_frames'], 1)
        print(f'{action:<20}  '
              f'videos: {s["videos_with_multiperson"]}/{s["total_videos"]} ({v_pct:.1f}%)  '
              f'frames: {s["frames_with_multiperson"]}/{s["total_frames"]} ({f_pct:.1f}%)')
        total_videos += s['total_videos']
        total_vmp    += s['videos_with_multiperson']
        total_frames += s['total_frames']
        total_fmp    += s['frames_with_multiperson']

    print(f'\n{"TOTAL":<20}  '
          f'videos: {total_vmp}/{total_videos} ({100*total_vmp/max(total_videos,1):.1f}%)  '
          f'frames: {total_fmp}/{total_frames} ({100*total_fmp/max(total_frames,1):.1f}%)')

    # ── Save JSON ─────────────────────────────────────────────────────────
    out = {
        'per_class': {k: dict(v) for k, v in stats.items()},
        'overall': {
            'total_videos': total_videos,
            'videos_with_multiperson': total_vmp,
            'total_frames': total_frames,
            'frames_with_multiperson': total_fmp,
        },
    }
    out_path = os.path.join(args.out_dir, 'multiperson_stats.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved to {out_path}')


if __name__ == '__main__':
    main()
