"""
PIVOT Tests — Verification Daemon
Unit tests for the verification daemon.
"""
import numpy as np
import pytest
from core.verification_daemon import (
    VerificationDaemon,
    VerificationResult,
    CorrectionTrigger,
    LatentRewindTrigger,
    LocalizedInpaintingTrigger,
    IdentityWeightIncreaseTrigger,
    create_verification_daemon,
)


class TestVerificationDaemon:
    """Test suite for VerificationDaemon."""
    
    def test_initialization(self):
        """Daemon should initialize with default parameters."""
        daemon = VerificationDaemon()
        
        assert daemon.identity_gate.threshold == 0.90
        assert daemon.max_retries == 5
    
    def test_custom_parameters(self):
        """Daemon should accept custom parameters."""
        daemon = VerificationDaemon(
            identity_threshold=0.85,
            max_retries=3,
            enable_logging=False,
        )
        
        assert daemon.identity_gate.threshold == 0.85
        assert daemon.max_retries == 3
    
    def test_verify_identity_passes_on_match(self):
        """Verify identity should pass when embeddings match."""
        daemon = VerificationDaemon(enable_logging=False)
        
        embedding = np.random.randn(512).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        
        result = daemon.verify_single_pass(embedding, embedding)
        
        assert result.passed is True
        assert result.identity_result.similarity_score == pytest.approx(1.0, abs=1e-4)
    
    def test_verify_identity_fails_on_mismatch(self):
        """Verify identity should fail when embeddings differ."""
        daemon = VerificationDaemon(enable_logging=False)
        
        ref = np.random.randn(512).astype(np.float32)
        ref = ref / np.linalg.norm(ref)
        
        gen = np.random.randn(512).astype(np.float32)
        gen = gen / np.linalg.norm(gen)
        
        result = daemon.verify_single_pass(ref, gen)
        
        assert result.passed is False
    
    def test_get_identity_similarity(self):
        """get_identity_similarity should return raw score."""
        daemon = VerificationDaemon(enable_logging=False)
        
        emb_a = np.ones(512, dtype=np.float32)
        emb_b = np.ones(512, dtype=np.float32)
        
        similarity = daemon.get_identity_similarity(emb_a, emb_b)
        
        assert similarity == pytest.approx(1.0, abs=1e-4)
    
    def test_set_identity_threshold(self):
        """set_identity_threshold should update the threshold."""
        daemon = VerificationDaemon()
        daemon.set_identity_threshold(0.80)
        
        assert daemon.identity_gate.threshold == 0.80
    
    def test_run_with_correction_loop(self):
        """Run should execute correction loop on failure."""
        daemon = VerificationDaemon(
            max_retries=3,
            enable_logging=False,
        )
        
        ref = np.random.randn(512).astype(np.float32)
        ref = ref / np.linalg.norm(ref)
        
        gen_fail = np.random.randn(512).astype(np.float32)
        gen_fail = gen_fail / np.linalg.norm(gen_fail)
        
        def regenerate_fn(weight):
            return ref
        
        result = daemon.run(ref, gen_fail, generation_fn=regenerate_fn)
        
        assert result.passed is True
        assert result.retry_count == 1
    
    def test_run_stops_at_max_retries(self):
        """Run should stop after max retries."""
        daemon = VerificationDaemon(
            max_retries=2,
            enable_logging=False,
        )
        
        ref = np.random.randn(512).astype(np.float32)
        ref = ref / np.linalg.norm(ref)
        
        gen = np.random.randn(512).astype(np.float32)
        
        result = daemon.run(ref, gen)
        
        assert result.retry_count == 2
        assert result.passed is False
    
    def test_correction_trigger_registration(self):
        """Should register correction triggers."""
        daemon = VerificationDaemon(enable_logging=False)
        trigger = LatentRewindTrigger()
        
        daemon.register_correction_trigger(trigger)
        
        assert len(daemon._correction_triggers) == 1


class TestCorrectionTriggers:
    """Test suite for correction triggers."""
    
    def test_latent_rewind_trigger_disabled(self):
        """Disabled trigger should return None."""
        trigger = LatentRewindTrigger(enabled=False)
        
        result = VerificationResult(passed=False, retry_count=0)
        
        assert trigger(result) is None
    
    def test_latent_rewind_trigger_enabled(self):
        """Enabled trigger should return latent."""
        trigger = LatentRewindTrigger(enabled=True)
        trigger.rewound_latents = [np.random.randn(4, 64, 64)]
        
        result = VerificationResult(passed=False, retry_count=0)
        
        latent = trigger(result)
        assert latent is not None
    
    def test_localized_inpainting_trigger(self):
        """Inpainting trigger should return mask."""
        trigger = LocalizedInpaintingTrigger(enabled=True)
        trigger.mask = np.ones((256, 256), dtype=np.float32)
        
        result = VerificationResult(passed=False, retry_count=0)
        
        mask = trigger(result)
        assert mask is not None
        assert mask.shape == (256, 256)
    
    def test_identity_weight_increase_trigger(self):
        """Weight trigger should increase with retries."""
        trigger = IdentityWeightIncreaseTrigger(
            base_weight=0.7,
            increment=0.1,
            max_weight=1.0,
        )
        
        result0 = VerificationResult(passed=False, retry_count=0)
        result1 = VerificationResult(passed=False, retry_count=1)
        result2 = VerificationResult(passed=False, retry_count=2)
        
        assert trigger(result0) == pytest.approx(0.7, abs=1e-6)
        assert trigger(result1) == pytest.approx(0.8, abs=1e-6)
        assert trigger(result2) == pytest.approx(0.9, abs=1e-6)
    
    def test_identity_weight_caps_at_max(self):
        """Weight trigger should cap at max_weight."""
        trigger = IdentityWeightIncreaseTrigger(
            base_weight=0.7,
            increment=0.1,
            max_weight=0.9,
        )
        
        result = VerificationResult(passed=False, retry_count=10)
        
        assert trigger(result) == 0.9
    
    def test_identity_weight_returns_base_on_pass(self):
        """Weight trigger should return base when passed."""
        trigger = IdentityWeightIncreaseTrigger(base_weight=0.7)
        
        result = VerificationResult(passed=True, retry_count=5)
        
        assert trigger(result) == 0.7


class TestVerificationResult:
    """Test suite for VerificationResult dataclass."""
    
    def test_dataclass_defaults(self):
        """Should have correct defaults."""
        result = VerificationResult(passed=True)
        
        assert result.passed is True
        assert result.identity_result is None
        assert result.retry_count == 0
        assert result.max_retries == 5
        assert result.error_message is None
        assert result.latent_rewind_count == 0
        assert result.final_similarity is None
    
    def test_dataclass_full_init(self):
        """Should accept all fields."""
        from core.cosine_similarity_gate import IdentityGateResult
        
        identity = IdentityGateResult(passed=True, similarity_score=0.95, threshold=0.90)
        
        result = VerificationResult(
            passed=True,
            identity_result=identity,
            retry_count=2,
            max_retries=5,
            error_message="test error",
            latent_rewind_count=1,
            final_similarity=0.95,
        )
        
        assert result.passed is True
        assert result.retry_count == 2
        assert result.final_similarity == 0.95


class TestFactory:
    """Test suite for factory function."""
    
    def test_create_verification_daemon(self):
        """Factory should create configured daemon."""
        daemon = create_verification_daemon(
            identity_threshold=0.85,
            max_retries=3,
            enable_logging=False,
        )
        
        assert isinstance(daemon, VerificationDaemon)
        assert daemon.identity_gate.threshold == 0.85
        assert daemon.max_retries == 3