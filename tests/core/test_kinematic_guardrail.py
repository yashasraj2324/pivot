"""
P.I.V.O.T. Core Module Tests
Test Kinematic Guardrail components
"""

from __future__ import annotations

import numpy as np
import pytest

from core.kinematic_guardrail import (
    COCO_BONES,
    JOINT_ANGLE_LIMITS,
    RIGID_REGIONS,
    V_MAX_DEFAULT,
    bone_length_invariance_loss,
    compute_bone_lengths,
    compute_rom_loss,
    compute_rigid_topology_loss,
    compute_velocity_loss,
    compute_l_physics,
)


class TestVelocityLoss:
    @pytest.fixture
    def base_pose(self):
        """Create base pose for velocity tests."""
        xs = np.linspace(0.0, 1.0, 17, dtype=np.float32)
        ys = np.linspace(1.0, 2.0, 17, dtype=np.float32)
        return np.stack([xs, ys], axis=-1)

    def test_compute_velocity_loss_zero_for_stationary(self, base_pose):
        """L_velocity should be ~0 for stationary keypoints."""
        pose = np.stack([base_pose, base_pose, base_pose], axis=0)[None, ...]
        _, loss = compute_velocity_loss(pose, v_max=2.0)
        assert loss == pytest.approx(0.0, abs=1e-7)

    def test_compute_velocity_loss_zero_below_v_max(self, base_pose):
        """L_velocity should be ~0 when velocities below v_max."""
        frame0 = base_pose
        frame1 = base_pose + np.array([0.5, 0.5], dtype=np.float32)
        frame2 = base_pose + np.array([1.0, 1.0], dtype=np.float32)
        pose = np.stack([frame0, frame1, frame2], axis=0)[None, ...]
        _, loss = compute_velocity_loss(pose, v_max=2.0)
        assert loss == pytest.approx(0.0, abs=1e-7)

    def test_compute_velocity_loss_penalizes_exceedance(self, base_pose):
        """L_velocity should penalize velocities exceeding v_max."""
        frame0 = base_pose
        frame1 = base_pose + np.array([3.0, 0.0], dtype=np.float32)  # velocity = 3.0 > v_max
        pose = np.stack([frame0, frame1], axis=0)[None, ...]
        _, loss = compute_velocity_loss(pose, v_max=2.0)
        assert loss > 0.0

    def test_compute_velocity_loss_single_frame_returns_zero(self, base_pose):
        """L_velocity should be 0 when only one frame."""
        pose = base_pose[None, None, ...]
        _, loss = compute_velocity_loss(pose, v_max=2.0)
        assert loss == 0.0

    def test_compute_velocity_loss_custom_v_max(self, base_pose):
        """compute_velocity_loss should accept custom v_max."""
        frame0 = base_pose
        frame1 = base_pose + np.array([2.5, 0.0], dtype=np.float32)  # velocity = 2.5
        pose = np.stack([frame0, frame1], axis=0)[None, ...]
        _, loss_low = compute_velocity_loss(pose, v_max=2.0)  # exceeds
        _, loss_high = compute_velocity_loss(pose, v_max=3.0)  # within
        assert loss_low > 0.0
        assert loss_high == pytest.approx(0.0, abs=1e-7)

    def test_compute_velocity_loss_returns_velocity_array(self, base_pose):
        """compute_velocity_loss should return velocity magnitudes."""
        frame0 = base_pose
        frame1 = base_pose + np.array([1.0, 0.0], dtype=np.float32)
        frame2 = base_pose + np.array([2.0, 0.0], dtype=np.float32)
        pose = np.stack([frame0, frame1, frame2], axis=0)[None, ...]
        velocities, _ = compute_velocity_loss(pose, v_max=2.0)
        assert velocities.shape == (1, 2, 17)  # [B, T-1, K]

    def test_compute_velocity_loss_accepts_unbatched(self, base_pose):
        """compute_velocity_loss should accept unbatched [T, K, 2] input."""
        pose = np.stack([base_pose, base_pose + 0.5], axis=0)
        velocities, loss = compute_velocity_loss(pose, v_max=2.0)
        assert velocities.shape == (1, 1, 17)  # [B, T-1, K]
        assert isinstance(loss, float)


class TestLPhysics:
    @pytest.fixture
    def base_pose(self):
        """Create base pose for L_physics tests."""
        xs = np.linspace(0.0, 1.0, 17, dtype=np.float32)
        ys = np.linspace(1.0, 2.0, 17, dtype=np.float32)
        return np.stack([xs, ys], axis=-1)

    def test_compute_l_physics_returns_dict(self, base_pose):
        """compute_l_physics should return dict with expected keys."""
        pose = np.stack([base_pose, base_pose + 0.5], axis=0)[None, ...]
        result = compute_l_physics(pose)
        assert "bone_loss" in result
        assert "rom_loss" in result
        assert "velocity_loss" in result
        assert "topology_loss" in result
        assert "total_loss" in result

    def test_compute_l_physics_combines_weights(self, base_pose):
        """compute_l_physics should combine losses with weights."""
        frame0 = base_pose
        frame1 = base_pose + np.array([3.0, 0.0], dtype=np.float32)  # velocity violation
        pose = np.stack([frame0, frame1], axis=0)[None, ...]
        result_default = compute_l_physics(pose, bone_weight=1.0, rom_weight=1.0, velocity_weight=1.0)
        result_custom = compute_l_physics(pose, bone_weight=0.5, rom_weight=0.25, velocity_weight=2.0)
        assert result_default["total_loss"] != result_custom["total_loss"]

    def test_compute_v_max_default(self):
        """V_MAX_DEFAULT should be 2.0."""
        assert V_MAX_DEFAULT == 2.0

    def test_compute_l_physics_rom_weight_affects_total(self, base_pose):
        """ROM weight should affect total loss when ROM violations exist."""
        frame0 = _make_human_like_pose()
        frame1 = frame0.copy()
        frame1[15] += np.array([0.8, -0.1], dtype=np.float32)  # perturb left ankle -> knee angle shift
        pose = np.stack([frame0, frame1], axis=0)[None, ...]

        result_low_rom = compute_l_physics(
            pose,
            bone_weight=1.0,
            rom_weight=0.2,
            velocity_weight=1.0,
        )
        result_high_rom = compute_l_physics(
            pose,
            bone_weight=1.0,
            rom_weight=2.0,
            velocity_weight=1.0,
        )

        assert result_high_rom["total_loss"] > result_low_rom["total_loss"]


class TestROMLoss:
    def test_compute_rom_loss_returns_expected_shape(self):
        base = _make_human_like_pose()
        pose = np.stack([base, base + np.array([0.1, -0.1], dtype=np.float32)], axis=0)[None, ...]

        joint_angles, rom_loss = compute_rom_loss(pose)

        assert joint_angles.shape == (1, 2, len(JOINT_ANGLE_LIMITS))
        assert isinstance(rom_loss, float)

    def test_compute_rom_loss_zero_for_pose_within_limits(self):
        base = _make_human_like_pose()
        pose = np.stack([base, base], axis=0)[None, ...]

        joint_angles, _ = compute_rom_loss(pose)
        limits = {
            joint_idx: (float(joint_angles[0, 0, j] - 1.0), float(joint_angles[0, 0, j] + 1.0))
            for j, joint_idx in enumerate(sorted(JOINT_ANGLE_LIMITS.keys()))
        }

        _, rom_loss = compute_rom_loss(pose, joint_limits=limits)
        assert rom_loss == pytest.approx(0.0, abs=1e-7)

    def test_compute_rom_loss_positive_when_limit_violated(self):
        base = _make_human_like_pose()
        deformed = base.copy()
        deformed[15] += np.array([1.5, 0.0], dtype=np.float32)  # force strong left knee bend
        pose = np.stack([base, deformed], axis=0)[None, ...]

        _, rom_loss = compute_rom_loss(pose)
        assert rom_loss > 0.0

    def test_compute_rom_loss_accepts_unbatched(self):
        base = _make_human_like_pose()
        pose = np.stack([base, base + np.array([0.2, 0.0], dtype=np.float32)], axis=0)

        joint_angles, rom_loss = compute_rom_loss(pose)
        assert joint_angles.shape == (1, 2, len(JOINT_ANGLE_LIMITS))
        assert isinstance(rom_loss, float)


def _make_human_like_pose() -> np.ndarray:
    """Create a stable non-collinear 17-keypoint pose for topology tests."""
    return np.array(
        [
            [0.00, 1.90],   # 0 nose
            [-0.10, 2.00],  # 1 left_eye
            [0.10, 2.00],   # 2 right_eye
            [-0.22, 1.98],  # 3 left_ear
            [0.22, 1.98],   # 4 right_ear
            [-0.30, 1.55],  # 5 left_shoulder
            [0.30, 1.55],   # 6 right_shoulder
            [-0.48, 1.20],  # 7 left_elbow
            [0.48, 1.20],   # 8 right_elbow
            [-0.58, 0.90],  # 9 left_wrist
            [0.58, 0.90],   # 10 right_wrist
            [-0.20, 1.00],  # 11 left_hip
            [0.20, 1.00],   # 12 right_hip
            [-0.20, 0.55],  # 13 left_knee
            [0.20, 0.55],   # 14 right_knee
            [-0.20, 0.10],  # 15 left_ankle
            [0.20, 0.10],   # 16 right_ankle
        ],
        dtype=np.float32,
    )


class TestRigidTopologyLoss:
    def test_returns_expected_shapes(self):
        base = _make_human_like_pose()
        pose = np.stack([base, base + np.array([0.1, -0.2], dtype=np.float32)], axis=0)[None, ...]

        region_ssim, loss = compute_rigid_topology_loss(pose)

        assert set(region_ssim.keys()) == set(RIGID_REGIONS.keys())
        for values in region_ssim.values():
            assert values.shape == (1, 1)
        assert isinstance(loss, float)

    def test_loss_near_zero_for_rigid_translation(self):
        base = _make_human_like_pose()
        frame0 = base
        frame1 = base + np.array([0.35, -0.27], dtype=np.float32)
        frame2 = base + np.array([0.70, -0.54], dtype=np.float32)
        pose = np.stack([frame0, frame1, frame2], axis=0)[None, ...]

        _, loss = compute_rigid_topology_loss(pose)
        assert loss == pytest.approx(0.0, abs=1e-4)

    def test_loss_positive_for_region_deformation(self):
        base = _make_human_like_pose()
        deformed = base.copy()
        deformed[6] += np.array([0.55, -0.35], dtype=np.float32)  # right shoulder shift
        pose = np.stack([base, deformed], axis=0)[None, ...]

        _, loss = compute_rigid_topology_loss(pose)
        assert loss > 0.0

    def test_single_frame_returns_zero_loss(self):
        base = _make_human_like_pose()
        pose = base[None, None, ...]

        region_ssim, loss = compute_rigid_topology_loss(pose)

        for values in region_ssim.values():
            assert values.shape == (1, 0)
        assert loss == 0.0


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
