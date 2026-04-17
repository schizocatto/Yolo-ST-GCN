"""
augmentation.py
Skeleton-specific data augmentation primitives for ST-GCN input tensors.

All functions operate on numpy arrays with shape (C, T, V, M) where:
    C = channels (x, y coordinates)
    T = temporal frames
    V = joints / vertices
    M = number of persons (usually 1)

These are the *tools* layer — stateless functions with no side effects.
The feeder layer (feeder.py) calls these based on per-class augmentation policy.
"""

from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Temporal augmentations
# ---------------------------------------------------------------------------

def random_choose(data: np.ndarray, window_size: int, auto_pad: bool = True) -> np.ndarray:
    """
    Randomly crop a contiguous window of `window_size` frames.

    If the sequence is shorter than window_size and auto_pad=True, it is
    padded with zeros on both sides to reach window_size.
    """
    C, T, V, M = data.shape
    if T == window_size:
        return data
    if T < window_size:
        if not auto_pad:
            return data
        return auto_pad_seq(data, window_size)
    start = np.random.randint(0, T - window_size)
    return data[:, start:start + window_size, :, :]


def auto_pad_seq(data: np.ndarray, target_size: int) -> np.ndarray:
    """Zero-pad a sequence symmetrically to reach target_size frames."""
    C, T, V, M = data.shape
    if T >= target_size:
        return data
    pad_total = target_size - T
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    pad = ((0, 0), (pad_left, pad_right), (0, 0), (0, 0))
    return np.pad(data, pad, mode='constant', constant_values=0)


def random_shift(data: np.ndarray) -> np.ndarray:
    """
    Random temporal translation: roll the sequence along the time axis by a
    random offset in [-T//4, +T//4].  Frames that wrap-around are zeroed out.
    """
    C, T, V, M = data.shape
    max_shift = max(1, T // 4)
    shift = np.random.randint(-max_shift, max_shift + 1)
    if shift == 0:
        return data
    out = np.zeros_like(data)
    if shift > 0:
        out[:, shift:, :, :] = data[:, :T - shift, :, :]
    else:
        out[:, :T + shift, :, :] = data[:, -shift:, :, :]
    return out


def temporal_reverse(data: np.ndarray) -> np.ndarray:
    """Reverse the temporal order of the sequence."""
    return data[:, ::-1, :, :].copy()


def temporal_subsample(data: np.ndarray, factor: float = 0.8) -> np.ndarray:
    """
    Drop frames by sampling every 1/factor frames, then stretch back with
    linear interpolation to preserve the original length.
    factor < 1.0  → faster motion (more frames dropped)
    factor > 1.0  → slower motion (frames duplicated)
    """
    C, T, V, M = data.shape
    new_T = max(2, int(round(T * factor)))
    src_indices = np.linspace(0, T - 1, new_T)
    # linear interpolation along time axis
    out = np.zeros((C, T, V, M), dtype=data.dtype)
    dst_indices = np.linspace(0, new_T - 1, T)
    for t, di in enumerate(dst_indices):
        lo = int(di)
        hi = min(lo + 1, new_T - 1)
        alpha = di - lo
        src_lo = int(src_indices[lo])
        src_hi = int(src_indices[hi])
        out[:, t, :, :] = (1 - alpha) * data[:, src_lo, :, :] + alpha * data[:, src_hi, :, :]
    return out


# ---------------------------------------------------------------------------
# Spatial augmentations
# ---------------------------------------------------------------------------

def random_move(data: np.ndarray,
                angle_range: float = 10.0,
                scale_range: float = 0.1,
                shift_range: float = 0.05) -> np.ndarray:
    """
    Apply a smoothly varying (low-frequency) 2-D affine transform to each frame.

    The transform parameters (angle, scale, shift) change slowly across time,
    simulating continuous camera or viewpoint drift.

    Parameters
    ----------
    angle_range  : max rotation in degrees
    scale_range  : max scale deviation from 1.0
    shift_range  : max translation as fraction of coordinate range
    """
    C, T, V, M = data.shape

    # generate smooth random parameters via linear interpolation
    def _smooth(lo, hi, n):
        start = np.random.uniform(lo, hi)
        end   = np.random.uniform(lo, hi)
        return np.linspace(start, end, n)

    angles = np.deg2rad(_smooth(-angle_range, angle_range, T))
    scales = _smooth(1.0 - scale_range, 1.0 + scale_range, T)
    dx     = _smooth(-shift_range, shift_range, T)
    dy     = _smooth(-shift_range, shift_range, T)

    out = data.copy()
    for t in range(T):
        cos_a = np.cos(angles[t]) * scales[t]
        sin_a = np.sin(angles[t]) * scales[t]
        x = out[0, t, :, :].copy()
        y = out[1, t, :, :].copy() if C > 1 else np.zeros_like(x)
        out[0, t, :, :] = cos_a * x - sin_a * y + dx[t]
        if C > 1:
            out[1, t, :, :] = sin_a * x + cos_a * y + dy[t]
    return out


def horizontal_flip(data: np.ndarray,
                    flip_pairs: Optional[list] = None) -> np.ndarray:
    """
    Mirror the skeleton horizontally (flip x-axis, swap left/right joints).

    Parameters
    ----------
    flip_pairs : list of (left_idx, right_idx) joint pairs to swap after flip.
                 If None, only the x coordinate is negated (no joint swapping).
    """
    out = data.copy()
    out[0, :, :, :] = -out[0, :, :, :]        # negate x
    if flip_pairs:
        for left, right in flip_pairs:
            out[:, :, left, :], out[:, :, right, :] = (
                out[:, :, right, :].copy(), out[:, :, left, :].copy()
            )
    return out


def random_scale(data: np.ndarray, scale_range: Tuple[float, float] = (0.85, 1.15)) -> np.ndarray:
    """
    Uniformly scale coordinates by a random factor in [scale_range[0], scale_range[1]].
    """
    factor = np.random.uniform(*scale_range)
    return data * factor


def random_noise(data: np.ndarray, std: float = 0.01) -> np.ndarray:
    """Add small zero-mean Gaussian noise to joint coordinates."""
    noise = np.random.normal(0.0, std, size=data.shape).astype(data.dtype)
    return data + noise


def joint_dropout(data: np.ndarray, drop_prob: float = 0.05) -> np.ndarray:
    """
    Randomly zero out individual joints for all frames (simulates occluded keypoints).
    Each joint is zeroed independently with probability `drop_prob`.
    """
    V = data.shape[2]
    mask = (np.random.rand(V) > drop_prob).astype(data.dtype)  # (V,)
    out = data.copy()
    out[:, :, :, :] *= mask[np.newaxis, np.newaxis, :, np.newaxis]
    return out


def random_translate(data: np.ndarray, max_shift: float = 0.05) -> np.ndarray:
    """
    Uniformly translate all joints by a random (dx, dy) offset.
    Simulates subject not being centered in frame.
    """
    out = data.copy()
    dx = np.random.uniform(-max_shift, max_shift)
    out[0] += dx
    if data.shape[0] > 1:
        dy = np.random.uniform(-max_shift, max_shift)
        out[1] += dy
    return out


# ---------------------------------------------------------------------------
# Skeleton-graph mix augmentations
# ---------------------------------------------------------------------------

def skeleton_mixup(data_a: np.ndarray, data_b: np.ndarray,
                   alpha: float = 0.3) -> np.ndarray:
    """
    Interpolate between two skeleton sequences (joint-level MixUp).

    λ is sampled from Beta(alpha, alpha); smaller alpha → weaker mixing.
    Only meaningful when called with the SAME class pair (intra-class MixUp)
    to avoid introducing ambiguous labels.
    """
    lam = np.random.beta(alpha, alpha)
    return lam * data_a + (1.0 - lam) * data_b


# ---------------------------------------------------------------------------
# Composite policy helper
# ---------------------------------------------------------------------------

def apply_augmentation_policy(
    data: np.ndarray,
    *,
    random_choose: bool = False,
    window_size: int = -1,
    random_shift: bool = False,
    random_move: bool = False,
    move_angle: float = 10.0,
    move_scale: float = 0.1,
    move_trans: float = 0.05,
    horizontal_flip_prob: float = 0.0,
    flip_pairs: Optional[list] = None,
    scale_prob: float = 0.0,
    scale_range: Tuple[float, float] = (0.85, 1.15),
    noise_std: float = 0.0,
    joint_drop_prob: float = 0.0,
    temporal_reverse_prob: float = 0.0,
    subsample_prob: float = 0.0,
    subsample_factor_range: Tuple[float, float] = (0.8, 1.2),
) -> np.ndarray:
    """
    Apply a stochastic augmentation policy described by keyword flags/probabilities.

    This is the single call site used by the feeder — it keeps the feeder code clean
    while all augmentation math lives here.
    """
    out = data

    # Temporal
    if random_choose and window_size > 0:
        out = _module_random_choose(out, window_size)
    elif window_size > 0:
        out = auto_pad_seq(out, window_size)

    if random_shift and np.random.rand() < 0.5:
        out = _module_random_shift(out)

    if temporal_reverse_prob > 0 and np.random.rand() < temporal_reverse_prob:
        out = temporal_reverse(out)

    if subsample_prob > 0 and np.random.rand() < subsample_prob:
        factor = np.random.uniform(*subsample_factor_range)
        out = temporal_subsample(out, factor)

    # Spatial
    if random_move:
        out = _apply_random_move(out, move_angle, move_scale, move_trans)

    if horizontal_flip_prob > 0 and np.random.rand() < horizontal_flip_prob:
        out = horizontal_flip(out, flip_pairs)

    if scale_prob > 0 and np.random.rand() < scale_prob:
        out = random_scale(out, scale_range)

    if noise_std > 0:
        out = random_noise(out, noise_std)

    if joint_drop_prob > 0:
        out = joint_dropout(out, joint_drop_prob)

    return out


# Internal aliases to avoid name collisions with parameter names in
# apply_augmentation_policy (which shadows built-in names locally).
_module_random_choose = random_choose
_module_random_shift  = random_shift


def _apply_random_move(data, angle, scale, trans):
    """Internal helper calling the module-level random_move."""
    return random_move(data, angle_range=angle, scale_range=scale, shift_range=trans)
