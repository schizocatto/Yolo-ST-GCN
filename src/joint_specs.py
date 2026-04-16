"""
joint_specs.py
Joint layout registry for config-driven skeleton pipelines.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class JointSpec:
    name: str
    num_joints: int
    center_joint: int
    # Directed parent->child pairs used for bone stream and spatial graph edges.
    bone_pairs: List[Tuple[int, int]]
    # Optional mapping from COCO17 keypoints to this layout (without virtual joint).
    coco_to_layout_idx: List[int]
    # Parent joints used to compute virtual center (if virtual center exists).
    virtual_center_parents: List[int]
    has_virtual_center: bool


PENN14_BONE_PAIRS = [
    (0, 1), (0, 2),
    (1, 3), (3, 5),
    (2, 4), (4, 6),
    (1, 7), (2, 8), (7, 8),
    (7, 9), (9, 11),
    (8, 10), (10, 12),
    (1, 13), (2, 13), (7, 13), (8, 13),
]

COCO18_BONE_PAIRS = [
    # Face connections (nose → eyes → ears)
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    # COCO body limbs (17 joints)
    (0, 5), (0, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    # Virtual center at index 17 from shoulders/hips
    (5, 17), (6, 17), (11, 17), (12, 17),
]


JOINT_SPECS: Dict[str, JointSpec] = {
    'penn14': JointSpec(
        name='penn14',
        num_joints=14,
        center_joint=13,
        bone_pairs=PENN14_BONE_PAIRS,
        # COCO17 -> Penn13 indices
        coco_to_layout_idx=[0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        virtual_center_parents=[1, 2, 7, 8],
        has_virtual_center=True,
    ),
    'coco18': JointSpec(
        name='coco18',
        num_joints=18,
        center_joint=17,
        bone_pairs=COCO18_BONE_PAIRS,
        # Keep original COCO17 ordering, then add virtual center.
        coco_to_layout_idx=list(range(17)),
        virtual_center_parents=[5, 6, 11, 12],
        has_virtual_center=True,
    ),
}


def get_joint_spec(name: str) -> JointSpec:
    key = name.strip().lower()
    if key not in JOINT_SPECS:
        raise ValueError(f"Unsupported joint spec '{name}'. Available: {sorted(JOINT_SPECS.keys())}")
    return JOINT_SPECS[key]
