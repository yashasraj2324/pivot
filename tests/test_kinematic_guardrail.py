"""
PIVOT Tests — Kinematic Guardrail
Unit tests for biomechanical range of motion limits (T2.3).
"""
import numpy as np
import pytest
from core.kinematic_guardrail import (
    COCO_KEYPOINTS,
    COCO_BONES,
    JOINT_ANGLE_LIMITS,
    get_joint_limits,
    PoseEstimator,
)


class TestJointAngleLimits:
    """Test suite for joint angle limits configuration."""

    def test_joint_angle_limits_defaults(self):
        """JOINT_ANGLE_LIMITS should have all expected joints."""
        assert len(JOINT_ANGLE_LIMITS) == 8
        assert 5 in JOINT_ANGLE_LIMITS  # left_shoulder
        assert 6 in JOINT_ANGLE_LIMITS  # right_shoulder
        assert 7 in JOINT_ANGLE_LIMITS  # left_elbow
        assert 8 in JOINT_ANGLE_LIMITS  # right_elbow
        assert 11 in JOINT_ANGLE_LIMITS  # left_hip
        assert 12 in JOINT_ANGLE_LIMITS  # right_hip
        assert 13 in JOINT_ANGLE_LIMITS  # left_knee
        assert 14 in JOINT_ANGLE_LIMITS  # right_knee

    def test_joint_angle_limits_values(self):
        """Joint limits should have physiological values."""
        assert JOINT_ANGLE_LIMITS[5] == (0, 180)   # shoulder
        assert JOINT_ANGLE_LIMITS[7] == (0, 145)   # elbow
        assert JOINT_ANGLE_LIMITS[11] == (0, 120)  # hip
        assert JOINT_ANGLE_LIMITS[13] == (0, 150)  # knee

    def test_get_joint_limits_no_custom(self):
        """get_joint_limits should return defaults when no custom provided."""
        limits = get_joint_limits()
        assert limits == JOINT_ANGLE_LIMITS

    def test_get_joint_limits_with_custom(self):
        """get_joint_limits should allow custom overrides."""
        custom = {5: (0, 90), 7: (0, 120)}
        limits = get_joint_limits(custom)
        
        assert limits[5] == (0, 90)
        assert limits[7] == (0, 120)
        assert limits[6] == JOINT_ANGLE_LIMITS[6]  # unchanged

    def test_get_joint_limits_does_not_mutate_original(self):
        """get_joint_limits should not modify original JOINT_ANGLE_LIMITS."""
        custom = {5: (0, 90)}
        limits = get_joint_limits(custom)
        
        assert JOINT_ANGLE_LIMITS[5] == (0, 180)


class TestROMLoss:
    """Test suite for compute_rom_loss functionality."""

    @pytest.fixture
    def pose_estimator(self):
        """Create pose estimator without loading detector."""
        return PoseEstimator(device="cpu")

    def test_rom_loss_valid_pose(self, pose_estimator):
        """L_ROM should be ~0 for valid pose within limits."""
        keypoints = np.array([
            [256, 50, 1.0], [270, 45, 1.0], [242, 45, 1.0], [235, 40, 1.0], [277, 40, 1.0],
            [280, 80, 1.0], [232, 80, 1.0], [300, 120, 1.0], [212, 120, 1.0],
            [320, 150, 1.0], [192, 150, 1.0], [256, 200, 1.0], [256, 200, 1.0],
            [256, 280, 1.0], [256, 280, 1.0], [256, 350, 1.0], [256, 350, 1.0],
        ], dtype=np.float32)
        
        rom_loss = pose_estimator.compute_rom_loss(keypoints)
        
        assert rom_loss < 0.2

    def test_rom_loss_upper_violation(self, pose_estimator):
        """L_ROM should penalize angle exceeding max limit."""
        keypoints = np.array([
            [256, 50, 1.0], [270, 45, 1.0], [242, 45, 1.0], [235, 40, 1.0], [277, 40, 1.0],
            [280, 80, 1.0], [232, 80, 1.0], [400, 300, 1.0], [112, 300, 1.0],
            [420, 350, 1.0], [92, 350, 1.0], [256, 200, 1.0], [256, 200, 1.0],
            [256, 280, 1.0], [256, 280, 1.0], [256, 350, 1.0], [256, 350, 1.0],
        ], dtype=np.float32)
        
        rom_loss = pose_estimator.compute_rom_loss(keypoints)
        
        assert rom_loss > 0

    def test_rom_loss_custom_limits(self, pose_estimator):
        """compute_rom_loss should accept custom limits."""
        keypoints = np.array([
            [256, 50, 1.0], [270, 45, 1.0], [242, 45, 1.0], [235, 40, 1.0], [277, 40, 1.0],
            [280, 80, 1.0], [232, 80, 1.0], [300, 120, 1.0], [212, 120, 1.0],
            [320, 150, 1.0], [192, 150, 1.0], [256, 200, 1.0], [256, 200, 1.0],
            [256, 280, 1.0], [256, 280, 1.0], [256, 350, 1.0], [256, 350, 1.0],
        ], dtype=np.float32)
        
        tight_limits = {5: (0, 20), 6: (0, 20)}
        rom_loss = pose_estimator.compute_rom_loss(keypoints, limits=tight_limits)
        
        assert rom_loss > 0


class TestClampJointAngles:
    """Test suite for clamp_joint_angles functionality."""

    @pytest.fixture
    def pose_estimator(self):
        """Create pose estimator."""
        return PoseEstimator(device="cpu")

    def test_clamp_no_violations(self, pose_estimator):
        """clamp_joint_angles should return keypoints and violations list."""
        keypoints = np.array([
            [256, 50, 1.0], [270, 45, 1.0], [242, 45, 1.0], [235, 40, 1.0], [277, 40, 1.0],
            [280, 80, 1.0], [232, 80, 1.0], [300, 120, 1.0], [212, 120, 1.0],
            [320, 150, 1.0], [192, 150, 1.0], [256, 200, 1.0], [256, 200, 1.0],
            [256, 280, 1.0], [256, 280, 1.0], [256, 350, 1.0], [256, 350, 1.0],
        ], dtype=np.float32)
        
        corrected, violations = pose_estimator.clamp_joint_angles(keypoints)
        
        assert corrected is not None
        assert isinstance(violations, list)

    def test_clamp_detects_violations(self, pose_estimator):
        """clamp_joint_angles should detect violations."""
        keypoints = np.array([
            [256, 50, 1.0], [270, 45, 1.0], [242, 45, 1.0], [235, 40, 1.0], [277, 40, 1.0],
            [280, 80, 1.0], [232, 80, 1.0], [450, 350, 1.0], [62, 350, 1.0],
            [480, 400, 1.0], [32, 400, 1.0], [256, 200, 1.0], [256, 200, 1.0],
            [256, 280, 1.0], [256, 280, 1.0], [256, 350, 1.0], [256, 350, 1.0],
        ], dtype=np.float32)
        
        corrected, violations = pose_estimator.clamp_joint_angles(keypoints)
        
        assert len(violations) > 0
        assert all("joint_name" in v for v in violations)
        assert all("angle" in v for v in violations)
        assert all("limit" in v for v in violations)

    def test_clamp_violation_types(self, pose_estimator):
        """clamp_joint_angles should identify violation types."""
        keypoints = np.array([
            [256, 50, 1.0], [270, 45, 1.0], [242, 45, 1.0], [235, 40, 1.0], [277, 40, 1.0],
            [280, 80, 1.0], [232, 80, 1.0], [300, 120, 1.0], [212, 120, 1.0],
            [320, 150, 1.0], [192, 150, 1.0], [256, 200, 1.0], [256, 200, 1.0],
            [256, 280, 1.0], [256, 280, 1.0], [256, 350, 1.0], [256, 350, 1.0],
        ], dtype=np.float32)
        
        tight_limits = {7: (0, 10), 13: (0, 10)}
        _, violations = pose_estimator.clamp_joint_angles(keypoints, limits=tight_limits)
        
        assert len(violations) > 0
        for v in violations:
            assert v["violation_type"] in ["hyperextension", "over_extension"]


class TestPoseEstimatorROM:
    """Test suite for PoseEstimator ROM integration."""

    def test_pose_estimator_has_rom_methods(self):
        """PoseEstimator should have ROM-related methods."""
        estimator = PoseEstimator(device="cpu")
        assert hasattr(estimator, "compute_rom_loss")
        assert hasattr(estimator, "clamp_joint_angles")

    def test_pose_estimator_has_joint_limits(self):
        """PoseEstimator should have access to JOINT_ANGLE_LIMITS."""
        from core.kinematic_guardrail import JOINT_ANGLE_LIMITS
        estimator = PoseEstimator(device="cpu")
        assert estimator.get_joint_angles is not None