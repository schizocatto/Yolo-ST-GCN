"""
feeder.py
Imbalance-aware skeleton feeder for ST-GCN training on datasets with
extreme class skew (e.g. FineGym99, FineGym288).

Design goals
------------
1. **Class-adaptive augmentation** — minority classes receive stronger
   augmentation; majority classes receive lighter (or no) augmentation.
   This balances the *effective* training distribution without oversampling.

2. **Optional oversampling** — a ``WeightedRandomSampler`` is provided as a
   helper so the DataLoader can draw balanced mini-batches without duplicating
   data in memory.

3. **Drop-in replacement** — ``SkeletonFeeder`` extends
   ``torch.utils.data.Dataset`` with the same ``(data, label)`` outputs as
   ``PennActionDataset``, so existing ``train_model`` / ``train_model_preloaded``
   calls need zero changes.

4. **Bone-stream support** — optional pre-computed bone data is passed through
   with the same augmentation applied.

Augmentation intensity table (default)
---------------------------------------
    Tier 0 – majority  (count ≥ median_count):  very light (noise only)
    Tier 1 – moderate  (count ≥ Q1):            light (shift + noise)
    Tier 2 – minority  (count ≥ Q1/4):          medium (move + flip + noise)
    Tier 3 – rare      (count <  Q1/4):         heavy  (move + flip + scale + noise + reverse)
    
    Additionally, Tier 2/3 classes get intra-class MixUp with probability
    ``mixup_prob`` to synthesize extra variation.

Usage
-----
    from src.feeder import SkeletonFeeder, make_weighted_sampler

    train_ds = SkeletonFeeder(
        data   = train_data,    # np.ndarray (N, C, T, V, M)
        labels = train_labels,  # np.ndarray (N,)
        augment = True,
        bone_data = train_bone_data,   # optional
        include_bone = True,
        flip_pairs = [(1,2),(3,4),(5,6),(7,8),(9,10),(11,12)],  # Penn14 LR pairs
    )

    # balanced mini-batches via oversampling
    sampler = make_weighted_sampler(train_ds)
    loader  = DataLoader(train_ds, batch_size=32, sampler=sampler)

    # OR standard random shuffle (lighter memory, augmentation does the work)
    loader  = DataLoader(train_ds, batch_size=32, shuffle=True)
"""

from __future__ import annotations

import warnings
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from src.augmentation import (
    apply_augmentation_policy,
    auto_pad_seq,
    random_noise,
    skeleton_mixup,
)


# ---------------------------------------------------------------------------
# Augmentation tier configuration
# ---------------------------------------------------------------------------

class AugPolicy:
    """Configurable per-tier augmentation knobs."""

    # Default tier configs: (tier_name, param_overrides)
    DEFAULTS: Dict[int, dict] = {
        # Majority — keep as-is, very light noise so gradients are stable
        0: dict(
            noise_std=0.005,
        ),
        # Moderate — light temporal shift + noise
        1: dict(
            random_shift=True,
            noise_std=0.008,
            joint_drop_prob=0.02,
        ),
        # Minority — spatial + temporal augmentation
        2: dict(
            random_move=True,
            move_angle=8.0,
            move_scale=0.08,
            move_trans=0.04,
            horizontal_flip_prob=0.5,
            random_shift=True,
            noise_std=0.01,
            joint_drop_prob=0.03,
            scale_prob=0.4,
            scale_range=(0.88, 1.12),
            temporal_reverse_prob=0.2,
        ),
        # Rare — heavy augmentation
        3: dict(
            random_move=True,
            move_angle=12.0,
            move_scale=0.12,
            move_trans=0.06,
            horizontal_flip_prob=0.5,
            random_shift=True,
            noise_std=0.015,
            joint_drop_prob=0.05,
            scale_prob=0.6,
            scale_range=(0.85, 1.15),
            temporal_reverse_prob=0.3,
            subsample_prob=0.3,
            subsample_factor_range=(0.8, 1.2),
        ),
    }

    def __init__(self, custom: Optional[Dict[int, dict]] = None):
        self._cfg: Dict[int, dict] = {}
        for tier, defaults in self.DEFAULTS.items():
            self._cfg[tier] = dict(defaults)
            if custom and tier in custom:
                self._cfg[tier].update(custom[tier])

    def __getitem__(self, tier: int) -> dict:
        return self._cfg.get(tier, self._cfg[0])


# ---------------------------------------------------------------------------
# Tier assignment from class counts
# ---------------------------------------------------------------------------

def compute_class_tiers(labels: np.ndarray) -> Dict[int, int]:
    """
    Assign each class to an augmentation tier based on sample-count percentiles.

    Returns
    -------
    Dict[class_id → tier]  where tier ∈ {0, 1, 2, 3}
    """
    counts = Counter(labels.tolist())
    if not counts:
        return {}

    count_vals = np.array(list(counts.values()), dtype=float)
    median_count = float(np.median(count_vals))
    q1_count = float(np.percentile(count_vals, 25))
    rare_threshold = q1_count / 4.0

    tier_map: Dict[int, int] = {}
    for cls, cnt in counts.items():
        if cnt >= median_count:
            tier_map[cls] = 0
        elif cnt >= q1_count:
            tier_map[cls] = 1
        elif cnt >= rare_threshold:
            tier_map[cls] = 2
        else:
            tier_map[cls] = 3
    return tier_map


def print_tier_summary(tier_map: Dict[int, int], counts: Counter) -> None:
    """Print a human-readable summary of the tier assignment."""
    from collections import defaultdict
    by_tier = defaultdict(list)
    for cls, tier in tier_map.items():
        by_tier[tier].append((cls, counts[cls]))
    total = sum(counts.values())
    print(f"\n[SkeletonFeeder] Augmentation tier assignment ({len(counts)} classes, {total} samples)")
    tier_names = {0: 'Majority (light)', 1: 'Moderate', 2: 'Minority (medium)', 3: 'Rare (heavy)'}
    for tier in sorted(by_tier):
        classes = by_tier[tier]
        c_total = sum(c for _, c in classes)
        pct = 100.0 * c_total / total
        print(f"  Tier {tier} [{tier_names[tier]}] — {len(classes)} classes, "
              f"{c_total} samples ({pct:.1f}%)")
        # Print 5 examples
        for cls, cnt in sorted(classes, key=lambda x: x[1])[:5]:
            print(f"      class {cls:4d}: {cnt} samples")
        if len(classes) > 5:
            print(f"      ... (+{len(classes)-5} more)")
    print()


# ---------------------------------------------------------------------------
# Main Feeder Dataset
# ---------------------------------------------------------------------------

class SkeletonFeeder(Dataset):
    """
    Imbalance-aware skeleton dataset with class-adaptive augmentation.

    Parameters
    ----------
    data          : float32 ndarray (N, C, T, V, M) — joint coordinates
    labels        : int64 ndarray  (N,)
    augment       : enable augmentation (should be True only for training split)
    bone_data     : optional float32 ndarray (N, C, T, V, M) — bone vectors
    include_bone  : if True, ``__getitem__`` returns ``((joint, bone), label)``
    flip_pairs    : list of (left_idx, right_idx) joint pairs for horizontal flip.
                    Penn-14 default: head has no pair, others are symmetric.
    window_size   : if > 0, randomly crop / pad to this temporal length
    mixup_prob    : probability of intra-class MixUp for Tier 2/3 samples
    custom_policy : optional dict[tier→dict] to override default aug params
    verbose       : print the tier summary table on construction
    """

    # Penn-14 left/right joint pairs for horizontal flip
    PENN14_FLIP_PAIRS = [
        (1, 2),   # l_sho  ↔ r_sho
        (3, 4),   # l_elbow ↔ r_elbow
        (5, 6),   # l_wrist ↔ r_wrist
        (7, 8),   # l_hip  ↔ r_hip
        (9, 10),  # l_knee ↔ r_knee
        (11, 12), # l_ankle ↔ r_ankle
    ]

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        *,
        augment: bool = True,
        bone_data: Optional[np.ndarray] = None,
        include_bone: bool = False,
        flip_pairs: Optional[List[Tuple[int, int]]] = None,
        window_size: int = -1,
        mixup_prob: float = 0.2,
        custom_policy: Optional[Dict[int, dict]] = None,
        verbose: bool = True,
    ):
        super().__init__()
        assert data.ndim == 5, f"data must be (N,C,T,V,M), got shape {data.shape}"
        assert len(data) == len(labels), "data and labels must have equal length"

        self.data = np.asarray(data, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.augment = augment
        self.include_bone = include_bone
        self.flip_pairs = flip_pairs if flip_pairs is not None else self.PENN14_FLIP_PAIRS
        self.window_size = window_size
        self.mixup_prob = mixup_prob if augment else 0.0

        # Bone data
        self.bone_data: Optional[np.ndarray] = None
        if include_bone:
            if bone_data is not None:
                self.bone_data = np.asarray(bone_data, dtype=np.float32)
            else:
                warnings.warn(
                    "SkeletonFeeder: include_bone=True but bone_data not provided. "
                    "Bone stream will be None. Pass pre-computed bone_data for best performance.",
                    UserWarning,
                    stacklevel=2,
                )

        # Build tier assignment
        self.tier_map: Dict[int, int] = {}
        self.policy = AugPolicy(custom_policy)
        self._class_indices: Dict[int, List[int]] = {}  # for intra-class MixUp

        if augment:
            self.tier_map = compute_class_tiers(self.labels)
            counts = Counter(self.labels.tolist())
            if verbose:
                print_tier_summary(self.tier_map, counts)
            # Build per-class index lists for MixUp sampling
            for i, lbl in enumerate(self.labels.tolist()):
                self._class_indices.setdefault(lbl, []).append(i)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        data_np = self.data[idx].copy()         # (C, T, V, M)
        label   = int(self.labels[idx])
        bone_np = self.bone_data[idx].copy() if (self.include_bone and self.bone_data is not None) else None

        if self.augment:
            data_np, bone_np = self._augment_sample(data_np, bone_np, label)

        joint_t = torch.from_numpy(data_np)
        label_t = torch.tensor(label, dtype=torch.long)

        if self.include_bone:
            bone_t = torch.from_numpy(bone_np) if bone_np is not None else torch.zeros_like(joint_t)
            return (joint_t, bone_t), label_t
        return joint_t, label_t

    # ------------------------------------------------------------------
    # Augmentation dispatch
    # ------------------------------------------------------------------

    def _augment_sample(
        self,
        data: np.ndarray,
        bone: Optional[np.ndarray],
        label: int,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        tier = self.tier_map.get(label, 0)
        policy_kwargs = self.policy[tier]

        # Optionally inject intra-class MixUp for minority/rare classes
        if tier >= 2 and self.mixup_prob > 0 and np.random.rand() < self.mixup_prob:
            peer_idx = self._sample_peer(label, exclude=None)
            if peer_idx is not None:
                peer_data = self.data[peer_idx].copy()
                data = skeleton_mixup(data, peer_data, alpha=0.3)
                if bone is not None and self.bone_data is not None:
                    peer_bone = self.bone_data[peer_idx].copy()
                    bone = skeleton_mixup(bone, peer_bone, alpha=0.3)

        # Apply augmentation policy
        aug_kwargs = dict(
            random_choose=policy_kwargs.get('random_choose', False),
            window_size=policy_kwargs.get('window_size', self.window_size),
            random_shift=policy_kwargs.get('random_shift', False),
            random_move=policy_kwargs.get('random_move', False),
            move_angle=policy_kwargs.get('move_angle', 10.0),
            move_scale=policy_kwargs.get('move_scale', 0.1),
            move_trans=policy_kwargs.get('move_trans', 0.05),
            horizontal_flip_prob=policy_kwargs.get('horizontal_flip_prob', 0.0),
            flip_pairs=self.flip_pairs,
            scale_prob=policy_kwargs.get('scale_prob', 0.0),
            scale_range=policy_kwargs.get('scale_range', (0.85, 1.15)),
            noise_std=policy_kwargs.get('noise_std', 0.0),
            joint_drop_prob=policy_kwargs.get('joint_drop_prob', 0.0),
            temporal_reverse_prob=policy_kwargs.get('temporal_reverse_prob', 0.0),
            subsample_prob=policy_kwargs.get('subsample_prob', 0.0),
            subsample_factor_range=policy_kwargs.get('subsample_factor_range', (0.8, 1.2)),
        )

        data = apply_augmentation_policy(data, **aug_kwargs)
        if bone is not None:
            bone = apply_augmentation_policy(bone, **aug_kwargs)

        return data, bone

    def _sample_peer(self, label: int, exclude: Optional[int]) -> Optional[int]:
        """Sample a random index from the same class (for MixUp)."""
        indices = self._class_indices.get(label, [])
        if not indices or (len(indices) == 1 and exclude in indices):
            return None
        candidates = [i for i in indices if i != exclude]
        if not candidates:
            return None
        return int(np.random.choice(candidates))

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def class_counts(self) -> Counter:
        return Counter(self.labels.tolist())

    @property
    def class_tiers(self) -> Dict[int, int]:
        return dict(self.tier_map)

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute per-class inverse-frequency weights.
        Useful for WeightedRandomSampler or as focal-loss alpha.
        """
        counts = self.class_counts
        n_classes = int(self.labels.max()) + 1
        weights = torch.zeros(n_classes, dtype=torch.float32)
        for cls, cnt in counts.items():
            weights[cls] = 1.0 / cnt
        # normalize so that mean weight = 1
        weights = weights / weights[weights > 0].mean()
        return weights


# ---------------------------------------------------------------------------
# Weighted Random Sampler helper
# ---------------------------------------------------------------------------

def make_weighted_sampler(
    dataset: SkeletonFeeder,
    replacement: bool = True,
    num_samples: Optional[int] = None,
) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler that draws class-balanced mini-batches.

    Each sample is weighted inversely to its class frequency, so rare classes
    appear as often as majority classes despite having fewer samples.

    Parameters
    ----------
    dataset      : SkeletonFeeder instance (needs ``.labels``)
    replacement  : sample with replacement (required when upsampling)
    num_samples  : total samples per epoch; defaults to len(dataset)

    Returns
    -------
    WeightedRandomSampler — pass as ``sampler=`` to DataLoader

    Example
    -------
        sampler = make_weighted_sampler(train_ds)
        loader  = DataLoader(train_ds, batch_size=32, sampler=sampler)
    """
    class_weights = dataset.get_class_weights()          # (num_classes,)
    sample_weights = class_weights[dataset.labels]       # (N,)

    n = num_samples if num_samples is not None else len(dataset)
    return WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=n,
        replacement=replacement,
    )


# ---------------------------------------------------------------------------
# Quick factory for common usage patterns
# ---------------------------------------------------------------------------

def build_feeder_pair(
    train_data:   np.ndarray,
    train_labels: np.ndarray,
    val_data:     np.ndarray,
    val_labels:   np.ndarray,
    *,
    train_bone:   Optional[np.ndarray] = None,
    val_bone:     Optional[np.ndarray] = None,
    include_bone: bool = False,
    flip_pairs:   Optional[List[Tuple[int, int]]] = None,
    window_size:  int = -1,
    mixup_prob:   float = 0.2,
    custom_policy: Optional[Dict[int, dict]] = None,
    verbose:      bool = True,
) -> Tuple[SkeletonFeeder, SkeletonFeeder]:
    """
    Convenience factory: create a (train_feeder, val_feeder) pair.

    Training feeder has augmentation enabled; validation feeder does not.

    Example
    -------
        train_ds, val_ds = build_feeder_pair(
            train_data, train_labels, val_data, val_labels,
            include_bone=True, train_bone=train_bone, val_bone=val_bone,
            flip_pairs=SkeletonFeeder.PENN14_FLIP_PAIRS,
        )
        sampler  = make_weighted_sampler(train_ds)
        train_dl = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=4)
        val_dl   = DataLoader(val_ds,   batch_size=64, shuffle=False,   num_workers=2)
    """
    train_ds = SkeletonFeeder(
        data=train_data,
        labels=train_labels,
        augment=True,
        bone_data=train_bone,
        include_bone=include_bone,
        flip_pairs=flip_pairs,
        window_size=window_size,
        mixup_prob=mixup_prob,
        custom_policy=custom_policy,
        verbose=verbose,
    )
    val_ds = SkeletonFeeder(
        data=val_data,
        labels=val_labels,
        augment=False,
        bone_data=val_bone,
        include_bone=include_bone,
        flip_pairs=flip_pairs,
        window_size=window_size,
        mixup_prob=0.0,
        verbose=False,
    )
    return train_ds, val_ds
