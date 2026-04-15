"""
config.py
Global constants shared across training, inference, and visualization.
"""

EXERCISE_CLASSES = [
    'bench_press', 'clean_and_jerk', 'jump_rope', 'jumping_jacks',
    'pullup', 'pushup', 'situp', 'squat',
]

CLASS_TO_ID = {cls: idx for idx, cls in enumerate(EXERCISE_CLASSES)}
ID_TO_CLASS = {idx: cls for idx, cls in enumerate(EXERCISE_CLASSES)}

NUM_CLASSES    = len(EXERCISE_CLASSES)
TARGET_FRAMES  = 64
NUM_JOINTS     = 14   # 13 Penn Action + 1 virtual center
IN_CHANNELS    = 2    # (x, y) coordinates

# Gym288-skeleton dataset defaults
GYM288_NUM_CLASSES = 288

# Penn Action joint names (0-12 = GT, 13 = virtual center)
JOINT_NAMES = [
    'head', 'l_sho', 'r_sho', 'l_elbow', 'r_elbow',
    'l_wrist', 'r_wrist', 'l_hip', 'r_hip',
    'l_knee', 'r_knee', 'l_ankle', 'r_ankle', 'center*',
]

JOINT_NAMES_13 = JOINT_NAMES[:13]

# Approximate 2-D positions for skeleton-graph visualizations (unit square)
JOINT_POS = {
    0:  (0.50, 0.95),  # head
    1:  (0.35, 0.76),  # l_sho
    2:  (0.65, 0.76),  # r_sho
    3:  (0.22, 0.58),  # l_elbow
    4:  (0.78, 0.58),  # r_elbow
    5:  (0.14, 0.40),  # l_wrist
    6:  (0.86, 0.40),  # r_wrist
    7:  (0.40, 0.52),  # l_hip
    8:  (0.60, 0.52),  # r_hip
    9:  (0.38, 0.30),  # l_knee
    10: (0.62, 0.30),  # r_knee
    11: (0.36, 0.08),  # l_ankle
    12: (0.64, 0.08),  # r_ankle
    13: (0.50, 0.64),  # virtual center
}

# Bone pairs for the 14-joint skeleton
PENN_BONES_13 = [
    (0, 1), (0, 2),
    (1, 3), (3, 5),
    (2, 4), (4, 6),
    (1, 7), (2, 8), (7, 8),
    (7, 9), (9, 11),
    (8, 10), (10, 12),
]
PENN_BONES_VIRTUAL = [(1, 13), (2, 13), (7, 13), (8, 13)]
PENN_BONES_14 = PENN_BONES_13 + PENN_BONES_VIRTUAL

# COCO (17-keypoint) → Penn Action (13-keypoint) index mapping
# Drops COCO indices 1-4 (eyes/ears); maps [0, 5..16] to Penn 0..12
COCO_TO_PENN_IDX = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
