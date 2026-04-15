"""
graph.py
Spatial-temporal graph for the 14-joint Penn Action skeleton.

Joint layout
------------
  0  head        7  l_hip
  1  l_sho       8  r_hip
  2  r_sho       9  l_knee
  3  l_elbow    10  r_knee
  4  r_elbow    11  l_ankle
  5  l_wrist    12  r_ankle
  6  r_wrist    13  virtual center (added)

The adjacency tensor A has shape (3, 14, 14):
  A[0] — self-loop / same hop-distance partition
  A[1] — centripetal  (toward center joint 13)
  A[2] — centrifugal  (away from center joint 13)
"""

import numpy as np
import torch

from src.config import PENN_BONES_14, COCO17_BONES_18


class Graph_PennAction_14Nodes:
    """Builds the normalised spatial-partition adjacency tensor used by ST-GCN."""

    num_node    = 14
    center_node = 13
    edges       = PENN_BONES_14

    def __init__(self):
        self.A = self._build_A()

    # ------------------------------------------------------------------
    def _hop_distance(self) -> np.ndarray:
        adj = np.zeros((self.num_node, self.num_node))
        for i, j in self.edges:
            adj[i, j] = adj[j, i] = 1

        hop = np.full((self.num_node, self.num_node), np.inf)
        powers = [np.linalg.matrix_power(adj, d) for d in range(self.num_node)]
        for d in range(self.num_node - 1, -1, -1):
            hop[np.stack(powers)[d] > 0] = d
        return hop

    def _build_A(self) -> torch.Tensor:
        hop             = self._hop_distance()
        dist_to_center  = hop[self.center_node]

        A = np.zeros((3, self.num_node, self.num_node))
        for i, j in self.edges:
            A[0, i, i] = A[0, j, j] = 1
            if dist_to_center[i] > dist_to_center[j]:
                A[1, j, i] = 1
                A[2, i, j] = 1
            elif dist_to_center[i] < dist_to_center[j]:
                A[1, i, j] = 1
                A[2, j, i] = 1
            else:
                A[0, i, j] = A[0, j, i] = 1

        # Row-normalise each partition
        for k in range(3):
            row_sum = A[k].sum(axis=1)
            D_inv   = np.where(row_sum > 0, 1.0 / (row_sum + 1e-4), 0.0)
            A[k]    = A[k] * D_inv[:, np.newaxis]

        return torch.tensor(A, dtype=torch.float32)


class Graph_COCO17_18Nodes:
    """
    Spatial-partition adjacency tensor for full COCO-17 + virtual center (18 joints).

    Joint layout
    ------------
      0  nose           9  right_wrist
      1  left_eye      10  left_wrist  (note: swap — see COCO17_JOINT_NAMES)
      2  right_eye     11  left_hip
      3  left_ear      12  right_hip
      4  right_ear     13  left_knee
      5  left_shoulder 14  right_knee
      6  right_shoulder 15 left_ankle
      7  left_elbow    16  right_ankle
      8  right_elbow   17  virtual center (mean of joints 5,6,11,12)

    Adjacency tensor A has shape (3, 18, 18):
      A[0] — self-loop / equidistant partition
      A[1] — centripetal  (toward virtual center, joint 17)
      A[2] — centrifugal  (away from virtual center, joint 17)
    """

    num_node    = 18
    center_node = 17
    edges       = COCO17_BONES_18

    def __init__(self):
        self.A = self._build_A()

    def _hop_distance(self) -> np.ndarray:
        adj = np.zeros((self.num_node, self.num_node))
        for i, j in self.edges:
            adj[i, j] = adj[j, i] = 1

        hop = np.full((self.num_node, self.num_node), np.inf)
        powers = [np.linalg.matrix_power(adj, d) for d in range(self.num_node)]
        for d in range(self.num_node - 1, -1, -1):
            hop[np.stack(powers)[d] > 0] = d
        return hop

    def _build_A(self) -> torch.Tensor:
        hop            = self._hop_distance()
        dist_to_center = hop[self.center_node]

        A = np.zeros((3, self.num_node, self.num_node))
        for i, j in self.edges:
            A[0, i, i] = A[0, j, j] = 1
            if dist_to_center[i] > dist_to_center[j]:
                A[1, j, i] = 1
                A[2, i, j] = 1
            elif dist_to_center[i] < dist_to_center[j]:
                A[1, i, j] = 1
                A[2, j, i] = 1
            else:
                A[0, i, j] = A[0, j, i] = 1

        for k in range(3):
            row_sum = A[k].sum(axis=1)
            D_inv   = np.where(row_sum > 0, 1.0 / (row_sum + 1e-4), 0.0)
            A[k]    = A[k] * D_inv[:, np.newaxis]

        return torch.tensor(A, dtype=torch.float32)
