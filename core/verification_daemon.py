"""
PIVOT Core — Verification Daemon
Requirement: ADR-003

Implements the verification daemon that orchestrates identity checking
and correction loops for the PIVOT generation pipeline.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Callable
import numpy as np

from core.cosine_similarity_gate import CosineSimilarityGate, IdentityGateResult


@dataclass
class VerificationResult:
    """Result of verification daemon execution."""
    passed: bool
    identity_result: Optional[IdentityGateResult] = None
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
    
    The daemon runs identity verification and orchestrates correction
    triggers when verification fails.
    """
    
    DEFAULT_MAX_RETRIES = 5
    IDENTITY_THRESHOLD = 0.90
    
    def __init__(
        self,
        identity_threshold: float = IDENTITY_THRESHOLD,
        max_retries: int = DEFAULT_MAX_RETRIES,
        enable_logging: bool = True,
        latent_rewind_fn: Optional[Callable[[], np.ndarray]] = None,
        inpainting_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        """
        Initialize verification daemon.
        
        Args:
            identity_threshold: Cosine similarity threshold (default 0.90)
            max_retries: Maximum correction attempts (default 5)
            enable_logging: Enable console logging
            latent_rewind_fn: Optional function to retrieve previous latents
            inpainting_fn: Optional function for localized inpainting
        """
        self.identity_gate = CosineSimilarityGate(
            threshold=identity_threshold,
            enable_logging=enable_logging,
        )
        self.max_retries = max_retries
        self.enable_logging = enable_logging
        self.latent_rewind_fn = latent_rewind_fn
        self.inpainting_fn = inpainting_fn
        
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
    
    def run(
        self,
        reference_embedding: np.ndarray,
        generated_embedding: np.ndarray,
        generation_fn: Optional[Callable[[float], np.ndarray]] = None,
    ) -> VerificationResult:
        """
        Run verification with correction loop.
        
        Args:
            reference_embedding: Reference ArcFace embedding (512-d)
            generated_embedding: Initial generated face embedding
            generation_fn: Optional function to regenerate with adjusted weight
            
        Returns:
            VerificationResult with final status
        """
        retry_count = 0
        latent_rewind_count = 0
        
        identity_result = self.verify_identity(reference_embedding, generated_embedding)
        
        if self.enable_logging:
            print(f"[VerificationDaemon] Initial check: {'PASS' if identity_result.passed else 'FAIL'}")
        
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
            retry_count=retry_count,
            max_retries=self.max_retries,
            latent_rewind_count=latent_rewind_count,
            final_similarity=identity_result.similarity_score,
        )
    
    def verify_single_pass(
        self,
        reference_embedding: np.ndarray,
        generated_embedding: np.ndarray,
    ) -> VerificationResult:
        """
        Single-pass verification without correction loop.
        
        Use this when you only need a pass/fail check without retries.
        
        Args:
            reference_embedding: Reference ArcFace embedding
            generated_embedding: Generated face embedding
            
        Returns:
            VerificationResult
        """
        identity_result = self.verify_identity(reference_embedding, generated_embedding)
        
        return VerificationResult(
            passed=identity_result.passed,
            identity_result=identity_result,
            retry_count=0,
            max_retries=self.max_retries,
            final_similarity=identity_result.similarity_score,
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
) -> VerificationDaemon:
    """
    Factory function to create a configured verification daemon.
    
    Args:
        identity_threshold: Cosine similarity threshold
        max_retries: Maximum correction attempts
        enable_logging: Enable console logging
        
    Returns:
        Configured VerificationDaemon instance
    """
    return VerificationDaemon(
        identity_threshold=identity_threshold,
        max_retries=max_retries,
        enable_logging=enable_logging,
    )