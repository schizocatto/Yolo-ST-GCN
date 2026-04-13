"""
tests/test_pipeline.py
Smoke test for the modular YOLO + ST-GCN codebase.

Uses synthetic .mat files (no real dataset required) to verify the full
data → model → training pipeline without needing Kaggle or GPU.

Run with:
    python tests/test_pipeline.py
or:
    pytest tests/test_pipeline.py -v
"""

import glob
import os
import sys
import tempfile

import matplotlib
matplotlib.use('Agg')

import numpy as np
import scipy.io
import torch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import EXERCISE_CLASSES, CLASS_TO_ID, TARGET_FRAMES
from src.graph import Graph_PennAction_14Nodes
from src.model import Model_STGCN, STGCN_Block
from src.dataset import (
    add_virtual_center_joint,
    temporal_align,
    build_data_tensors,
    PennActionDataset,
)
from src.train import train_epoch, eval_epoch
from src.inference import compute_iou
from torch.utils.data import DataLoader
import torch.nn as nn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def synthetic_labels_dir():
    """Create a temporary directory with fake Penn Action .mat files."""
    tmp = tempfile.mkdtemp(prefix='penn_fake_')
    labels_dir = os.path.join(tmp, 'labels')
    os.makedirs(labels_dir)

    np.random.seed(0)
    vid_id = 1
    for action in EXERCISE_CLASSES:
        for i in range(6):          # 6 clips per class = 48 total
            n_frames   = np.random.randint(30, 100)
            train_flag = 1 if i < 5 else 0   # 5 train, 1 test per class
            mat = {
                'x':          np.random.randint(50, 400, (n_frames, 13)).astype(float),
                'y':          np.random.randint(50, 400, (n_frames, 13)).astype(float),
                'action':     np.array([[action]]),
                'nframes':    np.array([[n_frames]]),
                'visibility': np.ones((n_frames, 13)),
                'train':      np.array([[train_flag]]),
            }
            scipy.io.savemat(os.path.join(labels_dir, f'{vid_id:04d}.mat'), mat)
            vid_id += 1

    return labels_dir


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

class TestGraph:
    def test_adjacency_shape(self):
        graph = Graph_PennAction_14Nodes()
        assert graph.A.shape == (3, 14, 14), 'Adjacency tensor shape mismatch'

    def test_adjacency_nonnegative(self):
        graph = Graph_PennAction_14Nodes()
        assert (graph.A.numpy() >= 0).all(), 'Negative adjacency values'


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

class TestPreprocessing:
    def test_add_virtual_center_joint(self):
        kpts = np.random.rand(64, 13, 2).astype(np.float32)
        out  = add_virtual_center_joint(kpts)
        assert out.shape == (64, 14, 2)
        # Virtual center should be the mean of joints 1,2,7,8
        expected = (kpts[:, 1, :] + kpts[:, 2, :] + kpts[:, 7, :] + kpts[:, 8, :]) / 4.0
        np.testing.assert_allclose(out[:, 13, :], expected, atol=1e-5)

    def test_temporal_align_upsample(self):
        kpts = np.random.rand(30, 13, 2)
        out  = temporal_align(kpts, 64)
        assert out.shape == (64, 13, 2)

    def test_temporal_align_downsample(self):
        kpts = np.random.rand(120, 13, 2)
        out  = temporal_align(kpts, 64)
        assert out.shape == (64, 13, 2)

    def test_temporal_align_exact(self):
        kpts = np.random.rand(64, 13, 2)
        out  = temporal_align(kpts, 64)
        assert out is kpts   # should return the same object unchanged


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TestDataset:
    def test_build_data_tensors(self, synthetic_labels_dir):
        data, labels, flags, raw_counts = build_data_tensors(synthetic_labels_dir)
        assert data.shape[1:] == (2, TARGET_FRAMES, 14, 1), f'Unexpected data shape: {data.shape}'
        assert len(data) == len(labels) == len(flags)
        assert len(raw_counts) == len(data)

    def test_official_split_flags(self, synthetic_labels_dir):
        _, _, flags, _ = build_data_tensors(synthetic_labels_dir)
        # Fixture creates 5 train + 1 test per class (8 classes = 8 test, 40 train)
        assert set(flags.tolist()) == {0, 1}, 'flags should contain both 0 and 1'
        assert (flags == 0).sum() == 8,  'Expected 8 official test samples'
        assert (flags == 1).sum() == 40, 'Expected 40 official train samples'

    def test_penn_action_dataset(self, synthetic_labels_dir):
        data, labels, flags, _ = build_data_tensors(synthetic_labels_dir)
        ds = PennActionDataset(data, labels)
        assert len(ds) == len(data)
        x, y = ds[0]
        assert x.shape == torch.Size([2, TARGET_FRAMES, 14, 1])
        assert y.dtype == torch.long


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

class TestInference:
    def test_iou_identical_boxes(self):
        box = np.array([10., 10., 50., 50.])
        assert abs(compute_iou(box, box) - 1.0) < 1e-5

    def test_iou_no_overlap(self):
        boxA = np.array([0.,  0., 10., 10.])
        boxB = np.array([20., 20., 30., 30.])
        assert compute_iou(boxA, boxB) == 0.0

    def test_iou_partial_overlap(self):
        boxA = np.array([0.,  0., 10., 10.])
        boxB = np.array([5.,  5., 15., 15.])
        iou  = compute_iou(boxA, boxB)
        assert 0.0 < iou < 1.0

    def test_iou_symmetry(self):
        boxA = np.array([0.,  0., 10., 10.])
        boxB = np.array([5.,  0., 15., 10.])
        assert abs(compute_iou(boxA, boxB) - compute_iou(boxB, boxA)) < 1e-6


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TestModel:
    def test_stgcn_block_forward(self):
        graph = Graph_PennAction_14Nodes()
        block = STGCN_Block(2, 64, graph.A, residual=False)
        x     = torch.randn(4, 2, 64, 14)
        out   = block(x)
        assert out.shape == (4, 64, 64, 14)

    def test_model_stgcn_forward(self):
        model = Model_STGCN(num_classes=8)
        x     = torch.randn(4, 2, 64, 14, 1)
        out   = model(x)
        assert out.shape == (4, 8), f'Expected (4,8), got {out.shape}'

    def test_model_output_type(self):
        model = Model_STGCN(num_classes=8)
        x     = torch.randn(2, 2, 64, 14, 1)
        out   = model(x)
        assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

class TestTraining:
    def _make_loader(self, synthetic_labels_dir):
        data, labels, flags, _ = build_data_tensors(synthetic_labels_dir)
        ds     = PennActionDataset(data, labels)
        return DataLoader(ds, batch_size=8, shuffle=True, drop_last=False)

    def test_train_epoch_runs(self, synthetic_labels_dir):
        device    = torch.device('cpu')
        model     = Model_STGCN(num_classes=8).to(device)
        loader    = self._make_loader(synthetic_labels_dir)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        loss, acc = train_epoch(model, loader, criterion, optimizer, device)
        assert 0.0 <= acc <= 1.0
        assert loss >= 0.0

    def test_eval_epoch_runs(self, synthetic_labels_dir):
        device    = torch.device('cpu')
        model     = Model_STGCN(num_classes=8).to(device)
        loader    = self._make_loader(synthetic_labels_dir)
        criterion = nn.CrossEntropyLoss()

        loss, acc, f1, preds, gt = eval_epoch(model, loader, criterion, device)
        assert 0.0 <= acc <= 1.0
        assert 0.0 <= f1  <= 1.0
        assert len(preds) == len(gt)


# ---------------------------------------------------------------------------
# Entry point for running without pytest
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import subprocess
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', __file__, '-v'],
        cwd=os.path.dirname(os.path.dirname(__file__)),
    )
    sys.exit(result.returncode)
