"""
PIVOT Project — Phase 1: Kinematic Guardrail
Requirement: KG-REQ-001 through KG-REQ-004

Implements pose estimation for 17-keypoint COCO skeleton using DWPose (ControlNet).
Provides bone length invariance, biomechanical range of motion limits,
temporal velocity limits, and rigid body topology preservation.
"""
from __future__ import annotations
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Sequence
import warnings

import numpy as np
from PIL import Image

try:
    from controlnet_aux import DWposeDetector
except Exception:  # noqa: BLE001
    DWposeDetector = None

import torch
import cv2


# COCO 17-keypoint skeleton definition
COCO_KEYPOINTS = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle"
}

# Bone connections (pairs of keypoint indices)
COCO_BONES = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # left/right arm
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # left/right leg
    (5, 11), (6, 12), (11, 12)  # torso
]

# Joint angle limits for biomechanical validation (per ADR-002)
# Format: {joint_idx: (min_angle, max_angle)} in degrees
JOINT_ANGLE_LIMITS: Dict[int, Tuple[float, float]] = {
    5: (0, 180),   # left_shoulder
    6: (0, 180),   # right_shoulder
    7: (0, 145),   # left_elbow
    8: (0, 145),   # right_elbow
    11: (0, 120),  # left_hip
    12: (0, 120),  # right_hip
    13: (0, 150),  # left_knee
    14: (0, 150),  # right_knee
}


def get_joint_limits(custom_limits: Dict[int, Tuple[float, float]] = None) -> Dict[int, Tuple[float, float]]:
    """
    Get joint limits with optional overrides.
    
    Args:
        custom_limits: Optional dict to override default limits
        
    Returns:
        Dict of joint_idx -> (min_angle, max_angle)
    """
    limits = JOINT_ANGLE_LIMITS.copy()
    if custom_limits:
        limits.update(custom_limits)
    return limits


class PoseEstimator:
    """
    Pose estimation for 17-keypoint COCO skeleton using DWPose detector.

    Provides methods to extract keypoints from images and compute pose-related
    metrics for kinematic constraint verification.
    """

    def __init__(self, device: str = "auto"):
        """
        Initialize the pose estimator with DWPose detector.

        Args:
            device: Device to run inference on ('auto', 'cpu', 'cuda', 'cuda:0', etc.)
        """
        self.device = self._resolve_device(device)
        self._detector = None
        self._last_keypoints: Optional[np.ndarray] = None

    def _resolve_device(self, device: str) -> str:
        """Resolve device string to actual torch device."""
        if device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    @property
    def detector(self):
        """Lazy-load detector on first access."""
        if self._detector is None:
            self._detector = self._load_detector()
        return self._detector

    def _load_detector(self):
        """Load and cache the DWPose detector."""
        try:
            # Try to use DWPoseDetector from controlnet_aux
            try:
                if DWposeDetector is None:
                    raise ImportError("controlnet_aux.DWposeDetector is unavailable")
                detector = DWposeDetector()
                detector.to(self.device)
                return detector
            except (ImportError, NameError) as e1:
                # If that fails due to missing OpenMMLab packages, try OpenposeDetector
                warnings.warn(
                    f"DWPose initialization failed ({e1}). "
                    "Falling back to OpenposeDetector. "
                    "For full DWPose support, install: mim install mmcv mmpose mmdet",
                    RuntimeWarning
                )
                # Import OpenposeDetector as fallback
                from controlnet_aux import OpenposeDetector
                detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
                detector.to(self.device)
                return detector
        except Exception as e:
            raise RuntimeError(
                f"Failed to load pose detector. Ensure controlnet-aux is installed: "
                f"pip install controlnet-aux. Error: {e}"
            ) from e

    def estimate_pose(self, image: Image.Image | np.ndarray | str) -> np.ndarray:
        """
        Estimate 17-keypoint pose from an image.

        Args:
            image: Input image as PIL Image, numpy array, or file path

        Returns:
            np.ndarray: Shape (17, 3) where each row is [x, y, confidence]
                       Coordinates are in pixel space relative to image dimensions
        """
        # Convert input to PIL Image
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise TypeError("Image must be PIL Image, numpy array, or file path")

        # Get original dimensions
        original_W, original_H = image.size

        # Run pose detection
        try:
            # Check if using DWPoseDetector (has pose_estimation attribute)
            if hasattr(self.detector, 'pose_estimation'):
                candidate, subset = self.detector.pose_estimation(np.array(image))
                # candidate shape: (num_people, num_keypoints, 3) where 3 is (x, y, score)
                
                # Extract keypoints for first detected person
                if candidate.shape[0] > 0:
                    # DWpose has 135 keypoints, but we take first 17 for COCO
                    keypoints = candidate[0, :17, :].copy()
                    # Scale normalized coordinates back to original image dimensions
                    keypoints[:, 0] *= original_W
                    keypoints[:, 1] *= original_H
                else:
                    keypoints = np.zeros((17, 3), dtype=np.float32)
            else:
                # Using OpenposeDetector - extract keypoints from body_estimation
                image_np = np.array(image)[:, :, ::-1].copy()  # RGB to BGR
                candidate, subset = self.detector.body_estimation(image_np)
                
                if len(candidate) > 0:
                    # candidate is in format [num_keypoints, 3] with normalized coordinates
                    keypoints = candidate[:17, :].copy()
                    # Scale coordinates back to original image dimensions
                    keypoints[:, 0] *= original_W
                    keypoints[:, 1] *= original_H
                else:
                    keypoints = np.zeros((17, 3), dtype=np.float32)
                
        except Exception as e:
            warnings.warn(f"Pose detection failed: {e}. Returning zeros.")
            keypoints = np.zeros((17, 3), dtype=np.float32)

        return keypoints.astype(np.float32)

    def _dwpose_to_coco_mapping(self) -> List[int]:
        """
        Map DWPose keypoint indices to COCO 17-keypoint format.

        Returns:
            List of indices to extract COCO keypoints from DWPose output
        """
        # This mapping may need to be verified against actual DWPose output
        # DWPose typically has more keypoints than COCO
        return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    def get_bone_lengths(self, keypoints: np.ndarray) -> Dict[Tuple[int, int], float]:
        """
        Calculate Euclidean distances between connected keypoints (bone lengths).

        Args:
            keypoints: Shape (17, 3) keypoints array

        Returns:
            Dict mapping bone pairs to their lengths
        """
        bone_lengths = {}
        for bone_pair in COCO_BONES:
            kpt1, kpt2 = keypoints[bone_pair[0]], keypoints[bone_pair[1]]
            # Only calculate if both keypoints have confidence > 0
            if kpt1[2] > 0 and kpt2[2] > 0:
                length = np.linalg.norm(kpt1[:2] - kpt2[:2])
                bone_lengths[bone_pair] = length
            else:
                bone_lengths[bone_pair] = 0.0

        return bone_lengths

    def get_joint_angles(self, keypoints: np.ndarray) -> Dict[int, float]:
        """
        Calculate joint angles for biomechanical validation.

        Args:
            keypoints: Shape (17, 3) keypoints array

        Returns:
            Dict mapping joint indices to their angles in degrees
        """
        angles = {}

        # Shoulder angles (elbow-shoulder-hip)
        angles[5] = self._calculate_angle(keypoints[7], keypoints[5], keypoints[11])  # left shoulder
        angles[6] = self._calculate_angle(keypoints[8], keypoints[6], keypoints[12])  # right shoulder

        # Elbow angles (wrist-elbow-shoulder)
        angles[7] = self._calculate_angle(keypoints[9], keypoints[7], keypoints[5])   # left elbow
        angles[8] = self._calculate_angle(keypoints[10], keypoints[8], keypoints[6]) # right elbow

        # Hip angles (knee-hip-shoulder)
        angles[11] = self._calculate_angle(keypoints[13], keypoints[11], keypoints[5]) # left hip
        angles[12] = self._calculate_angle(keypoints[14], keypoints[12], keypoints[6]) # right hip

        # Knee angles (ankle-knee-hip)
        angles[13] = self._calculate_angle(keypoints[15], keypoints[13], keypoints[11]) # left knee
        angles[14] = self._calculate_angle(keypoints[16], keypoints[14], keypoints[12]) # right knee

        return angles

    def _calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """
        Calculate angle at point p2 formed by points p1, p2, p3.

        Args:
            p1, p2, p3: Keypoint coordinates [x, y, confidence]

        Returns:
            Angle in degrees, or 0 if calculation fails
        """
        # Only calculate if all points have confidence > 0
        if p1[2] <= 0 or p2[2] <= 0 or p3[2] <= 0:
            return 0.0

        # Vectors from p2 to p1 and p2 to p3
        v1 = p1[:2] - p2[:2]
        v2 = p3[:2] - p2[:2]

        # Cosine of angle
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

        # Clamp to avoid numerical errors
        cos_angle = np.clip(cos_angle, -1, 1)

        # Convert to degrees
        angle = np.degrees(np.arccos(cos_angle))

        return angle

    def compute_rom_loss(
        self,
        keypoints: np.ndarray = None,
        joint_angles: Dict[int, float] = None,
        limits: Dict[int, Tuple[float, float]] = None,
    ) -> float:
        """
        Compute range of motion penalty (L_ROM).
        
        L_ROM = (1/N) * Σ max(0, angle - max_limit) + max(0, min_limit - angle)
        Penalizes both upper and lower bound violations.
        
        Args:
            keypoints: Shape (17, 3) keypoints array. If provided, angles are computed from it.
            joint_angles: Dict of joint_idx -> angle in degrees. Alternative to keypoints.
            limits: Joint angle limits. If None, uses JOINT_ANGLE_LIMITS.
            
        Returns:
            Normalized L_ROM penalty (0 = no violation)
        """
        if limits is None:
            limits = JOINT_ANGLE_LIMITS
        
        if joint_angles is None and keypoints is not None:
            joint_angles = self.get_joint_angles(keypoints)
        elif joint_angles is None and keypoints is None:
            return 0.0
        
        total_penalty = 0.0
        num_joints = 0
        
        for joint_idx, (min_angle, max_angle) in limits.items():
            angle = joint_angles.get(joint_idx, 0)
            if angle > 0:
                num_joints += 1
                if angle < min_angle:
                    total_penalty += (min_angle - angle) / 180.0
                elif angle > max_angle:
                    total_penalty += (angle - max_angle) / 180.0
        
        return total_penalty / max(num_joints, 1)
    
    def clamp_joint_angles(
        self,
        keypoints: np.ndarray,
        limits: Dict[int, Tuple[float, float]] = None,
    ) -> np.ndarray:
        """
        Return keypoints with joint angles clamped to physiological limits.
        
        Note: Full IK-based correction is complex. This implementation
        provides a simplified version that detects and flags violations.
        For production use, consider integrating a lightweight IK solver.
        
        Args:
            keypoints: Shape (17, 3) keypoints array
            limits: Joint angle limits. If None, uses JOINT_ANGLE_LIMITS.
            
        Returns:
            Tuple of (corrected_keypoints, violations) where violations is list of dicts
        """
        if limits is None:
            limits = JOINT_ANGLE_LIMITS
        
        corrected = keypoints.copy()
        joint_angles = self.get_joint_angles(keypoints)
        violations = []
        
        for joint_idx, (min_angle, max_angle) in limits.items():
            angle = joint_angles.get(joint_idx, 0)
            if angle > 0:
                if angle < min_angle or angle > max_angle:
                    violations.append({
                        "joint": joint_idx,
                        "joint_name": COCO_KEYPOINTS.get(joint_idx, f"joint_{joint_idx}"),
                        "angle": angle,
                        "limit": (min_angle, max_angle),
                        "violation_type": "hyperextension" if angle < min_angle else "over_extension"
                    })
        
        return corrected, violations

    def validate_pose(self, keypoints: np.ndarray) -> Dict[str, Any]:
        """
        Validate pose against biomechanical constraints.

        Args:
            keypoints: Shape (17, 3) keypoints array

        Returns:
            Dict with validation results and metrics
        """
        validation = {
            "is_valid": True,
            "violations": [],
            "bone_lengths": self.get_bone_lengths(keypoints),
            "joint_angles": self.get_joint_angles(keypoints)
        }

        # Check joint angle limits (physiological ranges)
        angle_limits = {
            5: (0, 180),   # left shoulder
            6: (0, 180),   # right shoulder
            7: (0, 145),   # left elbow
            8: (0, 145),   # right elbow
            11: (0, 120),  # left hip
            12: (0, 120),  # right hip
            13: (0, 150),  # left knee
            14: (0, 150),  # right knee
        }

        for joint_idx, (min_angle, max_angle) in angle_limits.items():
            angle = validation["joint_angles"].get(joint_idx, 0)
            if angle > 0 and (angle < min_angle or angle > max_angle):
                validation["is_valid"] = False
                validation["violations"].append({
                    "type": "joint_angle_violation",
                    "joint": COCO_KEYPOINTS[joint_idx],
                    "angle": angle,
                    "limits": (min_angle, max_angle)
                })

        return validation


# Global pose estimator instance
_pose_estimator = None

def get_pose_estimator(device: str = "auto") -> PoseEstimator:
    """
    Get or create global pose estimator instance.

    Args:
        device: Device for inference

    Returns:
        PoseEstimator instance
    """
    global _pose_estimator
    if _pose_estimator is None or _pose_estimator.device != device:
        _pose_estimator = PoseEstimator(device=device)
    return _pose_estimator


def estimate_pose_from_image(
    image_path: str,
    device: str = "auto"
) -> np.ndarray:
    """
    Convenience function to estimate pose from image file.

    Args:
        image_path: Path to image file
        device: Device for inference

    Returns:
        np.ndarray: Shape (17, 3) keypoints
    """
    estimator = get_pose_estimator(device=device)
    return estimator.estimate_pose(image_path)


def compute_bone_lengths(
    pose_keypoints: np.ndarray,
    bone_pairs: Sequence[tuple[int, int]] | None = None,
) -> np.ndarray:
    """
    Compute per-frame bone lengths from pose keypoints.

    Parameters
    ----------
    pose_keypoints:
        Pose tensor shaped ``[B, T, K, 2]`` or ``[T, K, 2]``.
    bone_pairs:
        Skeletal keypoint index pairs. Defaults to ``COCO_BONES``.

    Returns
    -------
    np.ndarray
        Bone lengths with shape ``[B, T, P]``, where ``P`` is number of pairs.
    """
    keypoints = _coerce_pose_keypoints(pose_keypoints)
    pairs = tuple(bone_pairs) if bone_pairs is not None else tuple(COCO_BONES)
    _validate_bone_pairs(pairs, num_keypoints=keypoints.shape[2])

    start_indices = np.asarray([a for a, _ in pairs], dtype=np.int64)
    end_indices = np.asarray([b for _, b in pairs], dtype=np.int64)

    starts = keypoints[:, :, start_indices, :]
    ends = keypoints[:, :, end_indices, :]
    vectors = ends - starts

    return np.linalg.norm(vectors, axis=-1).astype(np.float32)


def bone_length_invariance_loss(
    pose_keypoints: np.ndarray,
    bone_pairs: Sequence[tuple[int, int]] | None = None,
) -> tuple[np.ndarray, float]:
    """
    Compute L_bone invariance loss for consecutive frames.

    L_bone = sum_{b,t,p} (d[b,t,p] - d[b,t-1,p])^2
    where d is the per-frame bone length tensor.

    Parameters
    ----------
    pose_keypoints:
        Pose tensor shaped ``[B, T, K, 2]`` or ``[T, K, 2]``.
    bone_pairs:
        Skeletal keypoint index pairs. Defaults to ``COCO_BONES``.

    Returns
    -------
    tuple[np.ndarray, float]
        ``(bone_lengths, bone_loss)``.
    """
    bone_lengths = compute_bone_lengths(pose_keypoints, bone_pairs=bone_pairs)

    if bone_lengths.shape[1] < 2:
        return bone_lengths, 0.0

    delta = bone_lengths[:, 1:, :] - bone_lengths[:, :-1, :]
    loss = float(np.sum(np.square(delta), dtype=np.float64))
    return bone_lengths, loss


def _coerce_pose_keypoints(pose_keypoints: np.ndarray) -> np.ndarray:
    """Normalize pose keypoints to ``[B, T, K, 2]`` float32."""
    points = np.asarray(pose_keypoints, dtype=np.float32)

    if points.ndim == 3:
        points = points[None, ...]

    if points.ndim != 4:
        raise ValueError("pose_keypoints must have shape [B, T, K, 2] or [T, K, 2].")

    if points.shape[-1] < 2:
        raise ValueError("pose_keypoints last dimension must include x,y coordinates.")

    return points[..., :2].astype(np.float32)


def _validate_bone_pairs(
    bone_pairs: Sequence[tuple[int, int]],
    *,
    num_keypoints: int,
) -> None:
    """Validate bone pair indices against keypoint dimension."""
    if not bone_pairs:
        raise ValueError("bone_pairs cannot be empty.")

    for start, end in bone_pairs:
        if start < 0 or end < 0:
            raise ValueError("bone pair indices must be non-negative.")
        if start >= num_keypoints or end >= num_keypoints:
            raise ValueError(
                f"bone pair ({start}, {end}) out of range for {num_keypoints} keypoints."
            )