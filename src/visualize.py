"""
visualize.py
All plotting functions for the YOLO + ST-GCN pipeline.

Every function accepts an `out_dir` (str | None) parameter.
When provided, the figure is saved there; when None, plt.show() is called.
"""

import os
from collections import Counter
from typing import Dict, List, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.metrics import confusion_matrix, f1_score

from src.config import (
    EXERCISE_CLASSES,
    JOINT_NAMES,
    JOINT_NAMES_13,
    JOINT_POS,
    PENN_BONES_14,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_or_show(fig, out_dir: Optional[str], filename: str) -> None:
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  saved {path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Data exploration
# ---------------------------------------------------------------------------

def plot_action_distribution(
    df: pd.DataFrame,
    out_dir: Optional[str] = None,
    filename: str = 'action_distribution.png',
) -> None:
    """Bar chart of action class counts from the full dataset DataFrame."""
    fig, ax = plt.subplots(figsize=(12, 5))
    df['action'].value_counts().plot(kind='bar', color='skyblue', edgecolor='black', ax=ax)
    ax.set_title('Action Distribution — Penn Action Dataset')
    ax.set_xlabel('Action')
    ax.set_ylabel('Number of videos')
    fig.tight_layout()
    _save_or_show(fig, out_dir, filename)


def plot_data_stats(
    all_labels: List[int],
    raw_frame_counts: List[int],
    class_to_id: dict,
    target_frames: int = 64,
    out_dir: Optional[str] = None,
    filename: str = 'data_stats.png',
) -> None:
    """Sample-count bar + raw-length histogram side by side."""
    label_counts  = Counter(all_labels)
    counts_sorted = [label_counts[class_to_id[c]] for c in EXERCISE_CLASSES]
    colors        = plt.cm.Set2(np.linspace(0, 1, len(EXERCISE_CLASSES)))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Penn Action — Exercise Subset Analysis', fontsize=13, fontweight='bold')

    bars = axes[0].bar(EXERCISE_CLASSES, counts_sorted, color=colors,
                       edgecolor='white', linewidth=1.2)
    axes[0].set_title('Sample Count per Action Class')
    axes[0].set_xlabel('Action')
    axes[0].set_ylabel('Number of videos')
    axes[0].tick_params(axis='x', rotation=35)
    for bar, cnt in zip(bars, counts_sorted):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     str(cnt), ha='center', va='bottom', fontsize=9, fontweight='bold')

    axes[1].hist(raw_frame_counts, bins=15, color='#5dade2', edgecolor='white', linewidth=0.8)
    axes[1].axvline(x=target_frames, color='#e74c3c', linewidth=2,
                    linestyle='--', label=f'Target T={target_frames}')
    axes[1].set_title(f'Raw Video Length Distribution\n(before temporal alignment)')
    axes[1].set_xlabel('Number of frames')
    axes[1].set_ylabel('Number of videos')
    axes[1].legend(fontsize=10)

    fig.tight_layout()
    _save_or_show(fig, out_dir, filename)


# ---------------------------------------------------------------------------
# Graph / skeleton structure
# ---------------------------------------------------------------------------

def plot_skeleton_graph(
    graph,
    out_dir: Optional[str] = None,
    filename: str = 'skeleton_graph.png',
) -> None:
    """Draw the Penn Action 14-joint skeleton with spatial partition coloring."""
    A     = graph.A.numpy()
    C_SAME = '#3498db'
    C_IN   = '#27ae60'
    C_OUT  = '#e74c3c'

    fig, ax = plt.subplots(figsize=(7, 9))
    ax.set_xlim(-0.05, 1.15)
    ax.set_ylim(-0.05, 1.05)
    ax.axis('off')
    ax.set_title(
        'Penn Action Skeleton Graph\nSpatial Configuration Partitioning (14 joints)',
        fontsize=13, fontweight='bold', pad=12,
    )

    for i, j in graph.edges:
        xi, yi = JOINT_POS[i]
        xj, yj = JOINT_POS[j]
        if A[1, j, i] > 0 or A[1, i, j] > 0:
            color = C_IN
        elif A[2, j, i] > 0 or A[2, i, j] > 0:
            color = C_OUT
        else:
            color = C_SAME
        ax.plot([xi, xj], [yi, yj], color=color, linewidth=2.8, zorder=1, alpha=0.85)

    for idx, (x, y) in JOINT_POS.items():
        c = '#f39c12' if idx == 13 else '#2c3e50'
        ax.scatter(x, y, s=280 if idx == 13 else 200, color=c,
                   zorder=3, edgecolors='white', linewidths=1.2)
        ha = 'left' if x <= 0.5 else 'right'
        ax.text(x + (0.04 if x <= 0.5 else -0.04), y,
                f'{idx}: {JOINT_NAMES[idx]}',
                fontsize=8, va='center', ha=ha, zorder=4, color='#1a1a2e',
                fontweight='bold' if idx == 13 else 'normal')

    patches = [
        mpatches.Patch(color=C_IN,      label='Centripetal (toward center)'),
        mpatches.Patch(color=C_OUT,     label='Centrifugal (away from center)'),
        mpatches.Patch(color=C_SAME,    label='Same distance / self-loop'),
        mpatches.Patch(color='#f39c12', label='Virtual center joint (added)'),
    ]
    ax.legend(handles=patches, loc='lower center', fontsize=9, ncol=2, framealpha=0.9)
    fig.tight_layout()
    _save_or_show(fig, out_dir, filename)


def plot_adjacency_matrices(
    graph,
    out_dir: Optional[str] = None,
    filename: str = 'adjacency_matrices.png',
) -> None:
    """Heatmaps of the 3 spatial-partition adjacency matrices."""
    A_np   = graph.A.numpy()
    titles = [
        'A[0]  Self-loop / Same-distance',
        'A[1]  Centripetal  (→ center)',
        'A[2]  Centrifugal  (← center)',
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle('ST-GCN Adjacency Matrices — Spatial Configuration Partitioning',
                 fontsize=13, fontweight='bold', y=1.01)

    for k, ax in enumerate(axes):
        sns.heatmap(
            A_np[k], ax=ax, cmap='YlOrRd',
            xticklabels=JOINT_NAMES, yticklabels=JOINT_NAMES,
            linewidths=0.4, linecolor='#e0e0e0',
            vmin=0, vmax=A_np[k].max() + 0.01,
            cbar_kws={'shrink': 0.75, 'label': 'weight'},
        )
        ax.set_title(titles[k], fontsize=11, fontweight='bold', pad=8)
        ax.tick_params(axis='x', rotation=45, labelsize=7.5)
        ax.tick_params(axis='y', rotation=0,  labelsize=7.5)

    fig.tight_layout()
    _save_or_show(fig, out_dir, filename)


def plot_sample_skeleton(
    xy: np.ndarray,
    action_label: str,
    out_dir: Optional[str] = None,
    filename: str = 'sample_skeleton.png',
) -> None:
    """
    Draw a single-frame skeleton overlay.

    Parameters
    ----------
    xy           : (2, 14) array — [x-coords; y-coords] for all 14 joints
    action_label : string label for the figure title
    """
    fig, ax = plt.subplots(figsize=(5, 7))
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.axis('off')
    ax.set_title(f'Sample Skeleton — class: {action_label}\nFrame 0 | ★ = virtual center',
                 fontsize=11)

    from src.config import PENN_BONES_13, PENN_BONES_VIRTUAL
    for i, j in PENN_BONES_13:
        ax.plot([xy[0, i], xy[0, j]], [xy[1, i], xy[1, j]],
                color='#2c3e50', linewidth=2.5, zorder=1)
    for i, j in PENN_BONES_VIRTUAL:
        ax.plot([xy[0, i], xy[0, j]], [xy[1, i], xy[1, j]],
                color='#e67e22', linewidth=2.0, linestyle='--', zorder=1)

    ax.scatter(xy[0, :13], xy[1, :13], s=120, color='#2980b9', zorder=3,
               edgecolors='white', linewidths=1.2, label='GT joint (13)')
    ax.scatter(xy[0, 13], xy[1, 13], s=200, color='#27ae60', marker='*',
               zorder=4, edgecolors='white', linewidths=1.0, label='Virtual center (14th)')
    for i in range(14):
        ax.text(xy[0, i] + 1, xy[1, i] - 1, JOINT_NAMES[i], fontsize=7.5,
                color='#1a1a2e', zorder=5)
    ax.legend(fontsize=9, loc='upper right')
    fig.tight_layout()
    _save_or_show(fig, out_dir, filename)


# ---------------------------------------------------------------------------
# Training results
# ---------------------------------------------------------------------------

def plot_training_curves(
    history: Dict[str, list],
    out_dir: Optional[str] = None,
    filename: str = 'training_curves.png',
) -> None:
    """Loss, accuracy, and validation F1 curves."""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Training History — ST-GCN on Penn Action', fontsize=13, fontweight='bold')

    axes[0].plot(epochs, history['train_loss'], label='Train', color='#e74c3c', linewidth=2)
    axes[0].plot(epochs, history['val_loss'],   label='Val',   color='#3498db', linewidth=2)
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history['train_acc'], label='Train', color='#e74c3c', linewidth=2)
    axes[1].plot(epochs, history['val_acc'],   label='Val',   color='#3498db', linewidth=2)
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, history['val_f1'], color='#8e44ad', linewidth=2)
    axes[2].set_title('Validation Macro F1')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylim(0, 1)
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    _save_or_show(fig, out_dir, filename)


def plot_confusion_matrix(
    all_labels: List[int],
    all_preds: List[int],
    title: str = 'Confusion Matrix — Validation Set',
    out_dir: Optional[str] = None,
    filename: str = 'confusion_matrix.png',
) -> None:
    conf_mat = confusion_matrix(all_labels, all_preds)
    fig, ax  = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        conf_mat, annot=True, fmt='d', cmap='Blues',
        xticklabels=EXERCISE_CLASSES, yticklabels=EXERCISE_CLASSES,
        linewidths=0.5, linecolor='#f0f0f0', ax=ax,
        cbar_kws={'label': 'count'},
    )
    ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)
    ax.tick_params(axis='x', rotation=35, labelsize=9)
    ax.tick_params(axis='y', rotation=0,  labelsize=9)
    fig.tight_layout()
    _save_or_show(fig, out_dir, filename)


def plot_per_class_f1(
    all_labels: List[int],
    all_preds: List[int],
    out_dir: Optional[str] = None,
    filename: str = 'per_class_f1.png',
) -> None:
    f1_per_class = f1_score(all_labels, all_preds, average=None,
                            labels=list(range(len(EXERCISE_CLASSES))), zero_division=0)
    macro_f1     = float(np.mean(f1_per_class))
    bar_colors   = ['#27ae60' if f >= 0.8 else '#f39c12' if f >= 0.5 else '#e74c3c'
                    for f in f1_per_class]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(EXERCISE_CLASSES, f1_per_class, color=bar_colors,
                  edgecolor='white', linewidth=1.2)
    ax.axhline(y=macro_f1, color='#2c3e50', linewidth=1.8, linestyle='--')
    ax.set_ylim(0, 1.1)
    ax.set_title('Per-class F1 Score — Validation Set', fontsize=12, fontweight='bold')
    ax.set_xlabel('Action class')
    ax.set_ylabel('F1 Score')
    ax.tick_params(axis='x', rotation=35)

    for bar, f in zip(bars, f1_per_class):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{f:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.legend(handles=[
        Patch(color='#27ae60', label='F1 >= 0.80'),
        Patch(color='#f39c12', label='0.50 <= F1 < 0.80'),
        Patch(color='#e74c3c', label='F1 < 0.50'),
        plt.Line2D([0], [0], color='#2c3e50', linewidth=2, linestyle='--',
                   label=f'Macro F1 = {macro_f1:.3f}'),
    ], fontsize=9, loc='lower right')

    fig.tight_layout()
    _save_or_show(fig, out_dir, filename)


def plot_keypoint_quality(
    mean_per_joint: np.ndarray,
    overall_mean: float,
    out_dir: Optional[str] = None,
    filename: str = 'keypoint_quality.png',
) -> None:
    """Bar chart of per-joint YOLO keypoint error normalised by person height."""
    bar_colors = ['#27ae60' if e < 0.10 else '#f39c12' if e < 0.20 else '#e74c3c'
                  for e in mean_per_joint]
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(JOINT_NAMES_13, mean_per_joint, color=bar_colors,
                  edgecolor='white', linewidth=1.2)
    ax.axhline(overall_mean, color='#2c3e50', linestyle='--', linewidth=1.8)
    ax.set_title('YOLO Keypoint Error vs Ground Truth\n(Normalised by Person Height)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Joint')
    ax.set_ylabel('Normalised Distance')
    ax.tick_params(axis='x', rotation=35)

    for bar, e in zip(bars, mean_per_joint):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f'{e:.3f}', ha='center', va='bottom', fontsize=8)

    ax.legend(handles=[
        Patch(color='#27ae60', label='< 0.10 (good)'),
        Patch(color='#f39c12', label='0.10–0.20 (moderate)'),
        Patch(color='#e74c3c', label='> 0.20 (poor)'),
        plt.Line2D([0], [0], color='#2c3e50', linewidth=2, linestyle='--',
                   label=f'Mean = {overall_mean:.3f}'),
    ], fontsize=9)
    fig.tight_layout()
    _save_or_show(fig, out_dir, filename)


def plot_inference_result(
    frame_rgb: np.ndarray,
    kp14: np.ndarray,
    probs: np.ndarray,
    pred_idx: int,
    video_id: str,
    gt_action: str,
    pred_action: str,
    out_dir: Optional[str] = None,
    filename: str = 'inference_result.png',
) -> None:
    """Side-by-side: annotated frame + confidence bar chart."""
    correct = '✓' if pred_action == gt_action else '✗'
    colors  = ['#2ecc71' if i == pred_idx else '#95a5a6' for i in range(len(EXERCISE_CLASSES))]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].imshow(frame_rgb)
    for i, j in PENN_BONES_14:
        if kp14[i].sum() > 0 and kp14[j].sum() > 0:
            axes[0].plot([kp14[i, 0], kp14[j, 0]], [kp14[i, 1], kp14[j, 1]],
                         color='cyan', linewidth=2)
    axes[0].scatter(kp14[:, 0], kp14[:, 1], c='red', s=30, zorder=5)
    axes[0].scatter([kp14[13, 0]], [kp14[13, 1]], c='yellow', s=60,
                    zorder=6, label='virtual center')
    axes[0].legend(fontsize=8)
    axes[0].set_title(
        f'YOLO Skeleton — video {video_id}\nGT: {gt_action}  |  Pred: {pred_action} {correct}',
        fontsize=11,
    )
    axes[0].axis('off')

    axes[1].barh(EXERCISE_CLASSES, probs, color=colors)
    axes[1].set_xlim(0, 1.05)
    axes[1].set_title('ST-GCN Action Confidence', fontsize=11)
    axes[1].set_xlabel('Softmax probability')
    axes[1].axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    for i, p in enumerate(probs):
        axes[1].text(p + 0.01, i, f'{p:.2f}', va='center', fontsize=9)

    fig.tight_layout()
    _save_or_show(fig, out_dir, filename)
