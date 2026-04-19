"""
PIVOT Core — Verification Daemon
Requirement: ADR-003

Implements the verification daemon that orchestrates identity checking
and correction loops for the PIVOT generation pipeline.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Callable, Sequence
import numpy as np

from core.cosine_similarity_gate import CosineSimilarityGate, IdentityGateResult
from core.kinematic_guardrail import (
    V_MAX_DEFAULT,
    compute_l_physics,
    compute_bone_lengths,
    compute_velocity_loss,
    COCO_BONES,
)


@dataclass
class KinematicResult:
    """Result of kinematic constraint verification."""
    passed: bool
    bone_loss: float = 0.0
    velocity_loss: float = 0.0
    total_loss: float = 0.0
    v_max: float = V_MAX_DEFAULT
    max_velocity: float = 0.0


@dataclass
class VerificationResult:
    """Result of verification daemon execution."""
    passed: bool
    identity_result: Optional[IdentityGateResult] = None
    kinematic_result: Optional[KinematicResult] = None
    retry_count: int = 0
    max_retries: int = 5
    error_message: Optional[str] = None
    latent_rewind_count: int = 0
    final_similarity: Optional[float] = None


class CorrectionTrigger:
    """Base class for correction triggers."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
    
    def __call__(self, result: VerificationResult) -> bool:
        """Return True if correction should be triggered."""
        return self.enabled and not result.passed and result.retry_count < result.max_retries


class LatentRewindTrigger(CorrectionTrigger):
    """Trigger latent rewind on verification failure."""
    rewound_latents: list = field(default_factory=list)
    
    def __init__(self, enabled: bool = True):
        super().__init__(enabled)
        self.rewound_latents = []
    
    def __call__(self, result: VerificationResult) -> Optional[np.ndarray]:
        if not self.enabled or result.passed or result.retry_count >= result.max_retries:
            return None
        
        if self.rewound_latents and len(self.rewound_latents) > result.retry_count:
            return self.rewound_latents[result.retry_count]
        return None


class LocalizedInpaintingTrigger(CorrectionTrigger):
    """Trigger localized inpainting on verification failure."""
    mask: Optional[np.ndarray] = None
    
    def __init__(self, enabled: bool = True):
        super().__init__(enabled)
        self.mask = None
    
    def __call__(self, result: VerificationResult) -> Optional[np.ndarray]:
        if not self.enabled or result.passed:
            return None
        return self.mask


class IdentityWeightIncreaseTrigger(CorrectionTrigger):
    """Increase identity conditioning weight on retry."""
    base_weight: float = 0.7
    increment: float = 0.1
    max_weight: float = 1.0
    
    def __init__(self, enabled: bool = True, base_weight: float = 0.7, increment: float = 0.1, max_weight: float = 1.0):
        super().__init__(enabled)
        self.base_weight = base_weight
        self.increment = increment
        self.max_weight = max_weight
    
    def __call__(self, result: VerificationResult) -> float:
        if not self.enabled or result.passed:
            return self.base_weight
        
        new_weight = min(self.base_weight + (result.retry_count * self.increment), self.max_weight)
        return new_weight


class VerificationDaemon:
    """
    Verification daemon for PIVOT generation pipeline.

    Per ADR-003:
    - Sequential checking with immediate halt on failure
    - Structured correction loop with max 5 retries
    - Identity check: Cosine similarity ≥ 0.90
    - Kinematic check: L_velocity < threshold (velocity constraints)

    The daemon runs identity and kinematic verification and orchestrates correction
    triggers when verification fails.
    """

    DEFAULT_MAX_RETRIES = 5
    IDENTITY_THRESHOLD = 0.90
    DEFAULT_V_MAX = V_MAX_DEFAULT
    DEFAULT_KINEMATIC_THRESHOLD = 1.0

    def __init__(
        self,
        identity_threshold: float = IDENTITY_THRESHOLD,
        max_retries: int = DEFAULT_MAX_RETRIES,
        enable_logging: bool = True,
        latent_rewind_fn: Optional[Callable[[], np.ndarray]] = None,
        inpainting_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        v_max: float = DEFAULT_V_MAX,
        kinematic_threshold: float = DEFAULT_KINEMATIC_THRESHOLD,
        enable_kinematic: bool = True,
    ):
        """
        Initialize verification daemon.

        Args:
            identity_threshold: Cosine similarity threshold (default 0.90)
            max_retries: Maximum correction attempts (default 5)
            enable_logging: Enable console logging
            latent_rewind_fn: Optional function to retrieve previous latents
            inpainting_fn: Optional function for localized inpainting
            v_max: Maximum velocity threshold (default 2.0 units/frame)
            kinematic_threshold: Maximum allowed kinematic loss before fail (default 1.0)
            enable_kinematic: Enable kinematic constraint checking (default True)
        """
        self.identity_gate = CosineSimilarityGate(
            threshold=identity_threshold,
            enable_logging=enable_logging,
        )
        self.max_retries = max_retries
        self.enable_logging = enable_logging
        self.latent_rewind_fn = latent_rewind_fn
        self.inpainting_fn = inpainting_fn
        self.v_max = v_max
        self.kinematic_threshold = kinematic_threshold
        self.enable_kinematic = enable_kinematic

        self._correction_triggers: list[CorrectionTrigger] = []
        self._identity_weight_trigger = IdentityWeightIncreaseTrigger()
        
    def register_correction_trigger(self, trigger: CorrectionTrigger) -> None:
        """Register a correction trigger to be invoked on failure."""
        self._correction_triggers.append(trigger)
    
    def verify_identity(
        self,
        reference_embedding: np.ndarray,
        generated_embedding: np.ndarray,
    ) -> IdentityGateResult:
        """
        Run identity verification gate.

        Args:
            reference_embedding: Reference ArcFace embedding
            generated_embedding: Generated face embedding

        Returns:
            IdentityGateResult with pass/fail status
        """
        return self.identity_gate(reference_embedding, generated_embedding)

    def verify_kinematic(
        self,
        pose_keypoints: np.ndarray,
    ) -> KinematicResult:
        """
        Run kinematic constraint verification.

        Args:
            pose_keypoints: Pose keypoints shaped [B, T, K, 2] or [T, K, 2]

        Returns:
            KinematicResult with pass/fail status
        """
        if not self.enable_kinematic:
            return KinematicResult(passed=True)

        result = compute_l_physics(pose_keypoints, v_max=self.v_max)

        velocities, _ = compute_velocity_loss(pose_keypoints, v_max=self.v_max)
        max_velocity = float(np.max(velocities)) if velocities.size > 0 else 0.0

        passed = result["total_loss"] < self.kinematic_threshold

        if self.enable_logging:
            print(f"[VerificationDaemon] Kinematic: loss={result['total_loss']:.4f}, "
                  f"bone={result['bone_loss']:.4f}, velocity={result['velocity_loss']:.4f}, "
                  f"max_v={max_velocity:.2f}, v_max={self.v_max}, passed={passed}")

        return KinematicResult(
            passed=passed,
            bone_loss=result["bone_loss"],
            velocity_loss=result["velocity_loss"],
            total_loss=result["total_loss"],
            v_max=self.v_max,
            max_velocity=max_velocity,
        )
    
    def run(
        self,
        reference_embedding: np.ndarray,
        generated_embedding: np.ndarray,
        pose_keypoints: Optional[np.ndarray] = None,
        generation_fn: Optional[Callable[[float], np.ndarray]] = None,
    ) -> VerificationResult:
        """
        Run verification with correction loop.

        Args:
            reference_embedding: Reference ArcFace embedding (512-d)
            generated_embedding: Initial generated face embedding
            pose_keypoints: Optional pose keypoints [B, T, K, 2] for kinematic check
            generation_fn: Optional function to regenerate with adjusted weight

        Returns:
            VerificationResult with final status
        """
        retry_count = 0
        latent_rewind_count = 0

        kinematic_result = None
        if pose_keypoints is not None and self.enable_kinematic:
            kinematic_result = self.verify_kinematic(pose_keypoints)
            if not kinematic_result.passed and self.enable_logging:
                print(f"[VerificationDaemon] Kinematic check failed, halting early")
                return VerificationResult(
                    passed=False,
                    identity_result=None,
                    kinematic_result=kinematic_result,
                    retry_count=0,
                    max_retries=self.max_retries,
                )

        identity_result = self.verify_identity(reference_embedding, generated_embedding)

        if self.enable_logging:
            print(f"[VerificationDaemon] Initial check: identity={'PASS' if identity_result.passed else 'FAIL'}")

        while not identity_result.passed and retry_count < self.max_retries:
            retry_count += 1

            if self.enable_logging:
                print(f"[VerificationDaemon] Retry {retry_count}/{self.max_retries}")

            if self.latent_rewind_fn is not None:
                rewound = self.latent_rewind_fn()
                if rewound is not None:
                    latent_rewind_count += 1

            new_weight = self._identity_weight_trigger(
                VerificationResult(passed=False, retry_count=retry_count)
            )

            if generation_fn is not None:
                if self.enable_logging:
                    print(f"[VerificationDaemon] Regenerating with identity_weight={new_weight:.2f}")
                generated_embedding = generation_fn(new_weight)

            identity_result = self.verify_identity(reference_embedding, generated_embedding)

            if self.enable_logging:
                print(f"[VerificationDaemon] Retry {retry_count} result: {'PASS' if identity_result.passed else 'FAIL'}")

        return VerificationResult(
            passed=identity_result.passed,
            identity_result=identity_result,
            kinematic_result=kinematic_result,
            retry_count=retry_count,
            max_retries=self.max_retries,
            latent_rewind_count=latent_rewind_count,
            final_similarity=identity_result.similarity_score if identity_result else None,
        )
    
    def verify_single_pass(
        self,
        reference_embedding: np.ndarray,
        generated_embedding: np.ndarray,
        pose_keypoints: Optional[np.ndarray] = None,
    ) -> VerificationResult:
        """
        Single-pass verification without correction loop.

        Use this when you only need a pass/fail check without retries.

        Args:
            reference_embedding: Reference ArcFace embedding
            generated_embedding: Generated face embedding
            pose_keypoints: Optional pose keypoints [B, T, K, 2] for kinematic check

        Returns:
            VerificationResult
        """
        kinematic_result = None
        if pose_keypoints is not None and self.enable_kinematic:
            kinematic_result = self.verify_kinematic(pose_keypoints)
            if not kinematic_result.passed:
                return VerificationResult(
                    passed=False,
                    identity_result=None,
                    kinematic_result=kinematic_result,
                    retry_count=0,
                    max_retries=self.max_retries,
                )

        identity_result = self.verify_identity(reference_embedding, generated_embedding)

        return VerificationResult(
            passed=identity_result.passed,
            identity_result=identity_result,
            kinematic_result=kinematic_result,
            retry_count=0,
            max_retries=self.max_retries,
            final_similarity=identity_result.similarity_score if identity_result else None,
        )
    
    def set_identity_threshold(self, threshold: float) -> None:
        """Update the identity threshold."""
        self.identity_gate.set_threshold(threshold)
    
    def get_identity_similarity(
        self,
        reference_embedding: np.ndarray,
        generated_embedding: np.ndarray,
    ) -> float:
        """Get similarity score without threshold evaluation."""
        return self.identity_gate.get_similarity(reference_embedding, generated_embedding)


def create_verification_daemon(
    identity_threshold: float = 0.90,
    max_retries: int = 5,
    enable_logging: bool = True,
    v_max: float = V_MAX_DEFAULT,
    kinematic_threshold: float = 1.0,
    enable_kinematic: bool = True,
) -> VerificationDaemon:
    """
    Factory function to create a configured verification daemon.

    Args:
        identity_threshold: Cosine similarity threshold
        max_retries: Maximum correction attempts
        enable_logging: Enable console logging
        v_max: Maximum velocity threshold (units/frame)
        kinematic_threshold: Maximum kinematic loss before failure
        enable_kinematic: Enable kinematic constraint checking

    Returns:
        Configured VerificationDaemon instance
    """
    return VerificationDaemon(
        identity_threshold=identity_threshold,
        max_retries=max_retries,
        enable_logging=enable_logging,
        v_max=v_max,
        kinematic_threshold=kinematic_threshold,
        enable_kinematic=enable_kinematic,
    )

class LatentStateManager:
    """
    Tracks denoising latent states and enables rewind to t-1.

    Per ADR-003: On constraint violation, the daemon rewinds the
    denoising process to the last valid latent state (t-1).

    Usage:
        manager = LatentStateManager(max_history=5)
        manager.push(latent_t0)   # after each denoising step
        manager.push(latent_t1)
        prev = manager.rewind()   # returns latent_t0, discards latent_t1
    """

    def __init__(self, max_history: int = 5):
        """
        Args:
            max_history: Maximum number of latent states to keep.
                         Older states beyond this are discarded.
        """
        if max_history < 1:
            raise ValueError("max_history must be >= 1")
        self.max_history = max_history
        self._history: list[np.ndarray] = []

    def push(self, latent: np.ndarray) -> None:
        """
        Store a latent state after a denoising step.

        Args:
            latent: Latent tensor from the current denoising step.
        """
        self._history.append(latent.copy())
        if len(self._history) > self.max_history:
            self._history.pop(0)

    def rewind(self) -> Optional[np.ndarray]:
        """
        Rewind to the previous valid latent state (t-1).

        Discards the most recent (failing) state and returns
        the one before it. Returns None if no history exists.

        Returns:
            The t-1 latent state, or None if history is empty.
        """
        if not self._history:
            return None
        # discard the current (failing) state
        self._history.pop()
        if not self._history:
            return None
        return self._history[-1].copy()

    def current(self) -> Optional[np.ndarray]:
        """Return the most recent latent without modifying history."""
        if not self._history:
            return None
        return self._history[-1].copy()

    def reset(self) -> None:
        """Clear all stored latent states."""
        self._history.clear()

    @property
    def depth(self) -> int:
        """Number of latent states currently stored."""
        return len(self._history)

    def as_rewind_fn(self) -> Callable[[], Optional[np.ndarray]]:
        """
        Return a callable compatible with VerificationDaemon.latent_rewind_fn.

        This lets you pass the manager directly into the daemon:
            manager = LatentStateManager()
            daemon = VerificationDaemon(latent_rewind_fn=manager.as_rewind_fn())
        """
        return self.rewind