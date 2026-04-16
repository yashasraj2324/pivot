"""
P.I.V.O.T. Core Module Tests
Test Kinematic Guardrail components
"""

from __future__ import annotations

import numpy as np
import pytest

from core.kinematic_guardrail import (
    COCO_BONES,
    bone_length_invariance_loss,
    compute_bone_lengths,
)


def _make_base_pose(num_keypoints: int = 17) -> np.ndarray:
    """Create a deterministic 2D skeleton for tests."""
    xs = np.linspace(0.0, 1.0, num_keypoints, dtype=np.float32)
    ys = np.linspace(1.0, 2.0, num_keypoints, dtype=np.float32)
    return np.stack([xs, ys], axis=-1)


class TestBoneLengthInvariance:
    def test_compute_bone_lengths_output_shape(self):
        base = _make_base_pose()
        pose = np.stack([base, base + 0.1, base + 0.2], axis=0)  # [T, K, 2]
        pose = pose[None, ...]  # [B, T, K, 2]

        bone_lengths = compute_bone_lengths(pose)

        assert bone_lengths.shape == (1, 3, len(COCO_BONES))
        assert bone_lengths.dtype == np.float32

    def test_bone_length_invariance_loss_zero_for_rigid_translation(self):
        base = _make_base_pose()
        frames = []
        for t in range(4):
            # Translation preserves pairwise distances, so L_bone should be ~0.
            frames.append(base + np.array([0.25 * t, -0.15 * t], dtype=np.float32))

        pose = np.stack(frames, axis=0)[None, ...]  # [1, T, K, 2]

        _, loss = bone_length_invariance_loss(pose)
        assert loss == pytest.approx(0.0, abs=1e-7)

    def test_bone_length_invariance_loss_positive_when_bone_changes(self):
        base = _make_base_pose()
        frame0 = base.copy()
        frame1 = base.copy()
        frame2 = base.copy()

        # Stretch one keypoint in frame2 to change multiple bone lengths.
        frame2[7, 0] += 0.75

        pose = np.stack([frame0, frame1, frame2], axis=0)[None, ...]

        _, loss = bone_length_invariance_loss(pose)
        assert loss > 0.0

    def test_bone_length_invariance_loss_single_frame_is_zero(self):
        base = _make_base_pose()
        pose = base[None, None, ...]  # [1, 1, K, 2]

        bone_lengths, loss = bone_length_invariance_loss(pose)
        assert bone_lengths.shape == (1, 1, len(COCO_BONES))
        assert loss == 0.0

    def test_accepts_unbatched_pose_input(self):
        base = _make_base_pose()
        pose = np.stack([base, base + 0.1], axis=0)  # [T, K, 2]

        bone_lengths, loss = bone_length_invariance_loss(pose)

        assert bone_lengths.shape == (1, 2, len(COCO_BONES))
        assert isinstance(loss, float)

    def test_raises_for_invalid_pose_shape(self):
        bad_pose = np.zeros((17, 2), dtype=np.float32)

        with pytest.raises(ValueError, match="shape"):
            bone_length_invariance_loss(bad_pose)

    def test_raises_for_invalid_bone_pair_indices(self):
        base = _make_base_pose()
        pose = np.stack([base, base], axis=0)[None, ...]

        with pytest.raises(ValueError, match="out of range"):
            bone_length_invariance_loss(pose, bone_pairs=[(0, 999)])
