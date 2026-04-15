"""
PIVOT Core — Identity Cosine Similarity Gate
Requirement: IR-REQ-001, ADR-003

Implements the identity verification gate for the verification daemon.
Computes cosine similarity between reference and generated embeddings
and compares against the 0.90 threshold specified in ADR-003.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np

from core.identity_router import cosine_similarity


@dataclass
class IdentityGateResult:
    """Result of identity gate evaluation."""
    passed: bool
    similarity_score: float
    threshold: float
    reference_embedding: Optional[np.ndarray] = None
    generated_embedding: Optional[np.ndarray] = None


class CosineSimilarityGate:
    """
    Identity verification gate using cosine similarity.
    
    Compares reference and generated ArcFace embeddings against a
    threshold to determine if identity preservation is acceptable.
    
    Per ADR-003:
    - Threshold: 0.90 (well-established for ArcFace)
    - Sequential checking with immediate halt on failure
    - Structured correction loop with max 5 retries
    """
    
    DEFAULT_THRESHOLD: float = 0.90
    
    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        enable_logging: bool = True,
    ):
        """
        Initialize the identity gate.
        
        Args:
            threshold: Similarity threshold for pass (default 0.90)
            enable_logging: Whether to log similarity scores
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be in [0.0, 1.0], got {threshold}")
        
        self.threshold = threshold
        self.enable_logging = enable_logging
        
    def __call__(
        self,
        reference_embedding: np.ndarray,
        generated_embedding: np.ndarray,
    ) -> IdentityGateResult:
        """
        Evaluate identity preservation.
        
        Args:
            reference_embedding: Reference ArcFace embedding (512-d)
            generated_embedding: Generated face embedding (512-d)
            
        Returns:
            IdentityGateResult with pass/fail and similarity score
        """
        similarity = cosine_similarity(reference_embedding, generated_embedding)
        
        passed = similarity >= self.threshold
        
        if self.enable_logging:
            status = "PASS" if passed else "FAIL"
            print(
                f"[IdentityGate] {status}: "
                f"similarity={similarity:.4f}, threshold={self.threshold:.2f}"
            )
        
        return IdentityGateResult(
            passed=passed,
            similarity_score=similarity,
            threshold=self.threshold,
            reference_embedding=reference_embedding,
            generated_embedding=generated_embedding,
        )
    
    def evaluate(
        self,
        reference_embedding: np.ndarray,
        generated_embedding: np.ndarray,
    ) -> bool:
        """
        Simple boolean evaluation for quick pass/fail checking.
        
        Args:
            reference_embedding: Reference ArcFace embedding
            generated_embedding: Generated face embedding
            
        Returns:
            True if similarity >= threshold
        """
        similarity = cosine_similarity(reference_embedding, generated_embedding)
        return similarity >= self.threshold
    
    def get_similarity(
        self,
        reference_embedding: np.ndarray,
        generated_embedding: np.ndarray,
    ) -> float:
        """
        Get similarity score without threshold evaluation.
        
        Args:
            reference_embedding: Reference ArcFace embedding
            generated_embedding: Generated face embedding
            
        Returns:
            Cosine similarity in [-1, 1]
        """
        return cosine_similarity(reference_embedding, generated_embedding)
    
    def set_threshold(self, threshold: float) -> None:
        """Update the similarity threshold."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be in [0.0, 1.0], got {threshold}")
        self.threshold = threshold


def create_identity_gate(
    threshold: float = 0.90,
    enable_logging: bool = True,
) -> CosineSimilarityGate:
    """
    Factory function to create a configured identity gate.
    
    Args:
        threshold: Similarity threshold (default 0.90)
        enable_logging: Enable console logging
        
    Returns:
        Configured CosineSimilarityGate instance
    """
    return CosineSimilarityGate(
        threshold=threshold,
        enable_logging=enable_logging,
    )