"""
PIVOT Tests — Cosine Similarity Gate
Unit tests for the identity verification gate.
"""
import numpy as np
import pytest
from core.cosine_similarity_gate import (
    CosineSimilarityGate,
    IdentityGateResult,
    create_identity_gate,
)


class TestCosineSimilarityGate:
    """Test suite for CosineSimilarityGate."""
    
    def test_identical_embeddings_pass(self):
        """Identical embeddings should have similarity 1.0 and pass."""
        gate = CosineSimilarityGate(threshold=0.90)
        embedding = np.random.randn(512).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        
        result = gate(embedding, embedding)
        
        assert result.passed is True
        assert result.similarity_score == pytest.approx(1.0, abs=1e-5)
        assert result.threshold == 0.90
    
    def test_orthogonal_embeddings_fail(self):
        """Orthogonal embeddings should have near-zero similarity and fail."""
        gate = CosineSimilarityGate(threshold=0.90)
        
        emb_a = np.random.randn(512).astype(np.float32)
        emb_a = emb_a / np.linalg.norm(emb_a)
        
        emb_b = np.random.randn(512).astype(np.float32)
        emb_b = emb_b / np.linalg.norm(emb_b)
        
        result = gate(emb_a, emb_b)
        
        assert result.passed is False
        assert result.similarity_score < 0.90
    
    def test_high_similarity_passes(self):
        """Similarity above threshold should pass."""
        gate = CosineSimilarityGate(threshold=0.90)
        
        ref = np.random.randn(512).astype(np.float32)
        ref = ref / np.linalg.norm(ref)
        
        np.random.seed(42)
        noise = np.random.randn(512).astype(np.float32) * 0.01
        gen = ref + noise
        gen = gen / np.linalg.norm(gen)
        
        result = gate(ref, gen)
        
        assert result.similarity_score >= 0.90, f"Similarity {result.similarity_score:.4f} below threshold"
    
    def test_low_similarity_fails(self):
        """Similarity below threshold should fail."""
        gate = CosineSimilarityGate(threshold=0.90)
        
        ref = np.random.randn(512).astype(np.float32)
        ref = ref / np.linalg.norm(ref)
        
        gen = np.random.randn(512).astype(np.float32)
        gen = gen / np.linalg.norm(gen)
        
        result = gate(ref, gen)
        
        assert result.passed is False
    
    def test_evaluate_returns_boolean(self):
        """evaluate() should return a simple boolean."""
        gate = CosineSimilarityGate(threshold=0.90)
        embedding = np.random.randn(512).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        
        passed = gate.evaluate(embedding, embedding)
        
        assert passed is True
    
    def test_get_similarity_returns_score(self):
        """get_similarity() should return raw similarity score."""
        gate = CosineSimilarityGate()
        
        emb_a = np.ones(512, dtype=np.float32)
        emb_b = np.ones(512, dtype=np.float32)
        
        similarity = gate.get_similarity(emb_a, emb_b)
        
        assert similarity == pytest.approx(1.0, abs=1e-5)
    
    def test_set_threshold_updates(self):
        """set_threshold() should update the threshold."""
        gate = CosineSimilarityGate(threshold=0.90)
        gate.set_threshold(0.80)
        
        assert gate.threshold == 0.80
    
    def test_invalid_threshold_raises(self):
        """Invalid threshold values should raise ValueError."""
        with pytest.raises(ValueError):
            CosineSimilarityGate(threshold=1.5)
        
        with pytest.raises(ValueError):
            CosineSimilarityGate(threshold=-0.1)
    
    def test_factory_creates_gate(self):
        """Factory function should create configured gate."""
        gate = create_identity_gate(threshold=0.85, enable_logging=False)
        
        assert isinstance(gate, CosineSimilarityGate)
        assert gate.threshold == 0.85
        assert gate.enable_logging is False
    
    def test_result_contains_embeddings(self):
        """Result should contain embeddings for debugging."""
        gate = CosineSimilarityGate()
        ref = np.random.randn(512).astype(np.float32)
        gen = np.random.randn(512).astype(np.float32)
        
        result = gate(ref, gen)
        
        assert result.reference_embedding is not None
        assert result.generated_embedding is not None
        assert result.reference_embedding.shape == (512,)
        assert result.generated_embedding.shape == (512,)
    
    def test_logging_can_be_disabled(self):
        """Logging can be disabled via constructor."""
        gate = CosineSimilarityGate(enable_logging=False)
        embedding = np.random.randn(512).astype(np.float32)
        
        result = gate(embedding, embedding)
        
        assert result.passed is True


class TestIdentityGateResult:
    """Test suite for IdentityGateResult dataclass."""
    
    def test_dataclass_creation(self):
        """IdentityGateResult should be created correctly."""
        result = IdentityGateResult(
            passed=True,
            similarity_score=0.95,
            threshold=0.90,
        )
        
        assert result.passed is True
        assert result.similarity_score == 0.95
        assert result.threshold == 0.90
    
    def test_dataclass_with_embeddings(self):
        """Dataclass should accept optional embeddings."""
        ref = np.random.randn(512).astype(np.float32)
        gen = np.random.randn(512).astype(np.float32)
        
        result = IdentityGateResult(
            passed=True,
            similarity_score=0.95,
            threshold=0.90,
            reference_embedding=ref,
            generated_embedding=gen,
        )
        
        assert np.array_equal(result.reference_embedding, ref)
        assert np.array_equal(result.generated_embedding, gen)