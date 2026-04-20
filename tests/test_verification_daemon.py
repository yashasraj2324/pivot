"""
PIVOT Tests — Verification Daemon
Unit tests for the verification daemon.
"""
import numpy as np
import pytest
from core.verification_daemon import (
    VerificationDaemon,
    VerificationResult,
    KinematicResult,
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

class TestLatentStateManager:
    """Test suite for LatentStateManager — T3.4."""

    def test_push_and_depth(self):
        """depth should increase with each push."""
        from core.verification_daemon import LatentStateManager
        manager = LatentStateManager(max_history=5)

        assert manager.depth == 0
        manager.push(np.zeros((4, 64, 64), dtype=np.float32))
        assert manager.depth == 1
        manager.push(np.ones((4, 64, 64), dtype=np.float32))
        assert manager.depth == 2

    def test_rewind_returns_previous_state(self):
        """rewind should return t-1 latent and discard current."""
        from core.verification_daemon import LatentStateManager
        manager = LatentStateManager(max_history=5)

        t0 = np.full((4, 64, 64), 0.0, dtype=np.float32)
        t1 = np.full((4, 64, 64), 1.0, dtype=np.float32)

        manager.push(t0)
        manager.push(t1)

        rewound = manager.rewind()

        assert rewound is not None
        assert np.allclose(rewound, t0)
        assert manager.depth == 1

    def test_rewind_empty_returns_none(self):
        """rewind on empty manager should return None."""
        from core.verification_daemon import LatentStateManager
        manager = LatentStateManager(max_history=5)

        assert manager.rewind() is None

    def test_rewind_single_entry_returns_none(self):
        """rewind with only one entry discards it and returns None."""
        from core.verification_daemon import LatentStateManager
        manager = LatentStateManager(max_history=5)

        manager.push(np.zeros((4, 64, 64), dtype=np.float32))
        result = manager.rewind()

        assert result is None
        assert manager.depth == 0

    def test_max_history_evicts_oldest(self):
        """History beyond max_history should evict oldest entries."""
        from core.verification_daemon import LatentStateManager
        manager = LatentStateManager(max_history=3)

        for i in range(5):
            manager.push(np.full((4,), float(i), dtype=np.float32))

        assert manager.depth == 3

    def test_current_does_not_modify_history(self):
        """current() should not change depth."""
        from core.verification_daemon import LatentStateManager
        manager = LatentStateManager(max_history=5)

        manager.push(np.zeros((4,), dtype=np.float32))
        _ = manager.current()

        assert manager.depth == 1

    def test_reset_clears_history(self):
        """reset() should clear all stored states."""
        from core.verification_daemon import LatentStateManager
        manager = LatentStateManager(max_history=5)

        manager.push(np.zeros((4,), dtype=np.float32))
        manager.push(np.ones((4,), dtype=np.float32))
        manager.reset()

        assert manager.depth == 0
        assert manager.current() is None

    def test_as_rewind_fn_integrates_with_daemon(self):
        """as_rewind_fn should wire manager into daemon correctly."""
        from core.verification_daemon import LatentStateManager, VerificationDaemon

        manager = LatentStateManager(max_history=5)
        manager.push(np.zeros((4, 64, 64), dtype=np.float32))
        manager.push(np.ones((4, 64, 64), dtype=np.float32))
        manager.push(np.full((4, 64, 64), 2.0, dtype=np.float32))

        daemon = VerificationDaemon(
            max_retries=2,
            enable_logging=False,
            latent_rewind_fn=manager.as_rewind_fn(),
        )

        ref = np.random.randn(512).astype(np.float32)
        ref = ref / np.linalg.norm(ref)
        bad = np.random.randn(512).astype(np.float32)
        bad = bad / np.linalg.norm(bad)

        result = daemon.run(ref, bad)

        assert result.latent_rewind_count == 2

    def test_invalid_max_history_raises(self):
        """max_history < 1 should raise ValueError."""
        from core.verification_daemon import LatentStateManager

        with pytest.raises(ValueError):
            LatentStateManager(max_history=0) 

class TestInpaintingMaskGenerator:
    """Test suite for InpaintingMaskGenerator — T3.5."""

    def test_initialization(self):
        """Should initialize with correct parameters."""
        from core.verification_daemon import InpaintingMaskGenerator
        gen = InpaintingMaskGenerator(image_height=512, image_width=512)
        assert gen.image_height == 512
        assert gen.image_width == 512

    def test_invalid_dimensions_raises(self):
        """image_height/width < 1 should raise ValueError."""
        from core.verification_daemon import InpaintingMaskGenerator
        with pytest.raises(ValueError):
            InpaintingMaskGenerator(image_height=0, image_width=512)

    def test_invalid_dilation_raises(self):
        """Negative dilation should raise ValueError."""
        from core.verification_daemon import InpaintingMaskGenerator
        with pytest.raises(ValueError):
            InpaintingMaskGenerator(dilation_px=-1)

    def test_face_mask_shape(self):
        """Face mask should match image dimensions."""
        from core.verification_daemon import InpaintingMaskGenerator
        gen = InpaintingMaskGenerator(image_height=256, image_width=256)
        mask = gen.generate_face_mask()
        assert mask.shape == (256, 256)
        assert mask.dtype == np.float32

    def test_face_mask_values_clamped(self):
        """Face mask values must be in [0, 1]."""
        from core.verification_daemon import InpaintingMaskGenerator
        gen = InpaintingMaskGenerator(image_height=256, image_width=256)
        mask = gen.generate_face_mask()
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_face_mask_with_pose_keypoints(self):
        """Face mask should use pose keypoints when provided."""
        from core.verification_daemon import InpaintingMaskGenerator
        gen = InpaintingMaskGenerator(image_height=256, image_width=256, dilation_px=10)
        keypoints = np.zeros((17, 3), dtype=np.float32)
        keypoints[0] = [128, 50, 1.0]  # nose
        keypoints[1] = [120, 45, 1.0]  # left_eye
        keypoints[2] = [136, 45, 1.0]  # right_eye
        mask = gen.generate_face_mask(pose_keypoints=keypoints)
        assert mask.sum() > 0

    def test_face_mask_with_explicit_bbox(self):
        """Face mask should use explicit bbox when provided."""
        from core.verification_daemon import InpaintingMaskGenerator
        gen = InpaintingMaskGenerator(image_height=256, image_width=256, dilation_px=0)
        mask = gen.generate_face_mask(face_bbox=(50, 50, 150, 150))
        assert mask[100, 100] == 1.0
        assert mask[0, 0] == 0.0

    def test_joint_mask_shape(self):
        """Joint mask should match image dimensions."""
        from core.verification_daemon import InpaintingMaskGenerator
        gen = InpaintingMaskGenerator(image_height=256, image_width=256)
        keypoints = np.zeros((17, 2), dtype=np.float32)
        keypoints[5] = [100, 100]
        keypoints[7] = [120, 150]
        mask = gen.generate_joint_mask(keypoints, violating_joints=[5, 7])
        assert mask.shape == (256, 256)

    def test_joint_mask_covers_violating_region(self):
        """Joint mask should cover the violating joint area."""
        from core.verification_daemon import InpaintingMaskGenerator
        gen = InpaintingMaskGenerator(image_height=256, image_width=256, dilation_px=5)
        keypoints = np.zeros((17, 2), dtype=np.float32)
        keypoints[7] = [128, 128]  # left elbow
        mask = gen.generate_joint_mask(keypoints, violating_joints=[7])
        assert mask[128, 128] == 1.0

    def test_generate_from_identity_failure(self):
        """Identity failure should produce face mask."""
        from core.verification_daemon import InpaintingMaskGenerator
        from core.cosine_similarity_gate import IdentityGateResult
        gen = InpaintingMaskGenerator(image_height=256, image_width=256)
        result = VerificationResult(
            passed=False,
            identity_result=IdentityGateResult(
                passed=False, similarity_score=0.5, threshold=0.9
            ),
        )
        mask = gen.generate_from_verification_result(result)
        assert mask.shape == (256, 256)
        assert mask.sum() > 0

    def test_generate_from_kinematic_failure(self):
        """Kinematic failure with keypoints should produce joint mask."""
        from core.verification_daemon import InpaintingMaskGenerator
        gen = InpaintingMaskGenerator(image_height=256, image_width=256)
        kinematic = KinematicResult(passed=False, total_loss=2.0)
        result = VerificationResult(passed=False, kinematic_result=kinematic)
        keypoints = np.zeros((17, 2), dtype=np.float32)
        keypoints[5] = [100, 100]
        mask = gen.generate_from_verification_result(result, pose_keypoints=keypoints)
        assert mask.sum() > 0

    def test_generate_fallback_full_frame(self):
        """No specific failure should mask full frame."""
        from core.verification_daemon import InpaintingMaskGenerator
        gen = InpaintingMaskGenerator(image_height=64, image_width=64)
        result = VerificationResult(passed=True)
        mask = gen.generate_from_verification_result(result)
        assert mask.sum() == 64 * 64    

class TestConstrainedRegenerationEngine:
    """Test suite for ConstrainedRegenerationEngine — T3.6."""

    def test_initialization(self):
        """Should initialize with correct parameters."""
        from core.verification_daemon import ConstrainedRegenerationEngine
        engine = ConstrainedRegenerationEngine()
        assert engine.base_identity_weight == 0.7
        assert engine.base_kinematic_weight == 0.7

    def test_invalid_base_weight_raises(self):
        """Invalid base weight should raise ValueError."""
        from core.verification_daemon import ConstrainedRegenerationEngine
        with pytest.raises(ValueError):
            ConstrainedRegenerationEngine(base_identity_weight=1.5)

    def test_invalid_increment_raises(self):
        """Negative increment should raise ValueError."""
        from core.verification_daemon import ConstrainedRegenerationEngine
        with pytest.raises(ValueError):
            ConstrainedRegenerationEngine(identity_increment=-0.1)

    def test_no_failure_returns_base_weights(self):
        """No failure should return base weights."""
        from core.verification_daemon import ConstrainedRegenerationEngine
        engine = ConstrainedRegenerationEngine(
            base_identity_weight=0.7,
            base_kinematic_weight=0.7,
        )
        result = VerificationResult(passed=True)
        config = engine.get_config(result, retry_count=0)
        assert config.identity_weight == pytest.approx(0.7)
        assert config.kinematic_weight == pytest.approx(0.7)
        assert config.violated_constraint == "none"

    def test_identity_failure_increases_identity_weight(self):
        """Identity failure should increase identity weight."""
        from core.verification_daemon import ConstrainedRegenerationEngine
        from core.cosine_similarity_gate import IdentityGateResult
        engine = ConstrainedRegenerationEngine(
            base_identity_weight=0.7,
            identity_increment=0.1,
        )
        result = VerificationResult(
            passed=False,
            identity_result=IdentityGateResult(
                passed=False, similarity_score=0.5, threshold=0.9
            ),
        )
        config = engine.get_config(result, retry_count=2)
        assert config.identity_weight == pytest.approx(0.9)
        assert config.kinematic_weight == pytest.approx(0.7)
        assert config.violated_constraint == "identity"

    def test_kinematic_failure_increases_kinematic_weight(self):
        """Kinematic failure should increase kinematic weight."""
        from core.verification_daemon import ConstrainedRegenerationEngine
        engine = ConstrainedRegenerationEngine(
            base_kinematic_weight=0.7,
            kinematic_increment=0.1,
        )
        result = VerificationResult(
            passed=False,
            kinematic_result=KinematicResult(passed=False, total_loss=2.0),
        )
        config = engine.get_config(result, retry_count=1)
        assert config.kinematic_weight == pytest.approx(0.8)
        assert config.identity_weight == pytest.approx(0.7)
        assert config.violated_constraint == "kinematic"

    def test_both_failures_increases_both_weights(self):
        """Both failures should increase both weights."""
        from core.verification_daemon import ConstrainedRegenerationEngine
        from core.cosine_similarity_gate import IdentityGateResult
        engine = ConstrainedRegenerationEngine(
            base_identity_weight=0.7,
            base_kinematic_weight=0.7,
            identity_increment=0.1,
            kinematic_increment=0.1,
        )
        result = VerificationResult(
            passed=False,
            identity_result=IdentityGateResult(
                passed=False, similarity_score=0.5, threshold=0.9
            ),
            kinematic_result=KinematicResult(passed=False, total_loss=2.0),
        )
        config = engine.get_config(result, retry_count=1)
        assert config.identity_weight == pytest.approx(0.8)
        assert config.kinematic_weight == pytest.approx(0.8)
        assert config.violated_constraint == "both"

    def test_weight_caps_at_max(self):
        """Weight should not exceed max."""
        from core.verification_daemon import ConstrainedRegenerationEngine
        from core.cosine_similarity_gate import IdentityGateResult
        engine = ConstrainedRegenerationEngine(
            base_identity_weight=0.7,
            identity_increment=0.1,
            max_identity_weight=0.9,
        )
        result = VerificationResult(
            passed=False,
            identity_result=IdentityGateResult(
                passed=False, similarity_score=0.5, threshold=0.9
            ),
        )
        config = engine.get_config(result, retry_count=10)
        assert config.identity_weight == pytest.approx(0.9)

    def test_get_identity_weight_shortcut(self):
        """get_identity_weight should return correct weight."""
        from core.verification_daemon import ConstrainedRegenerationEngine
        from core.cosine_similarity_gate import IdentityGateResult
        engine = ConstrainedRegenerationEngine(
            base_identity_weight=0.7,
            identity_increment=0.1,
        )
        result = VerificationResult(
            passed=False,
            identity_result=IdentityGateResult(
                passed=False, similarity_score=0.5, threshold=0.9
            ),
        )
        weight = engine.get_identity_weight(result, retry_count=1)
        assert weight == pytest.approx(0.8)

    def test_config_includes_retry_count(self):
        """Config should store retry count."""
        from core.verification_daemon import ConstrainedRegenerationEngine
        engine = ConstrainedRegenerationEngine()
        result = VerificationResult(passed=True)
        config = engine.get_config(result, retry_count=3)
        assert config.retry_count == 3    

class TestMaxRetryDepthWithFallback:
    """Test suite for T3.7 — max retry depth with fallback to best candidate."""

    def test_default_max_retries_is_five(self):
        """Daemon default max retries must be 5."""
        daemon = VerificationDaemon(enable_logging=False)
        assert daemon.max_retries == 5

    def test_stops_at_max_retries(self):
        """Daemon must stop exactly at max_retries."""
        daemon = VerificationDaemon(max_retries=5, enable_logging=False)
        ref = np.random.randn(512).astype(np.float32)
        ref = ref / np.linalg.norm(ref)
        bad = np.random.randn(512).astype(np.float32)
        bad = bad / np.linalg.norm(bad)
        result = daemon.run(ref, bad)
        assert result.retry_count == 5

    def test_fallback_returns_best_candidate(self):
        """After exhausting retries fallback must return highest similarity."""
        daemon = VerificationDaemon(max_retries=3, enable_logging=False)
        ref = np.random.randn(512).astype(np.float32)
        ref = ref / np.linalg.norm(ref)

        # always return a bad embedding — never passes threshold
        def generation_fn(weight):
            bad = np.random.randn(512).astype(np.float32)
            return bad / np.linalg.norm(bad)

        bad = np.random.randn(512).astype(np.float32)
        bad = bad / np.linalg.norm(bad)

        result = daemon.run(ref, bad, generation_fn=generation_fn)
    
        assert result.retry_count == 3
        assert result.passed is False
        assert result.final_similarity is not None
    
    def test_fallback_not_triggered_on_pass(self):
        """Fallback must not trigger if identity passes."""
        daemon = VerificationDaemon(max_retries=5, enable_logging=False)
        ref = np.random.randn(512).astype(np.float32)
        ref = ref / np.linalg.norm(ref)
        result = daemon.run(ref, ref)
        assert result.passed is True
        assert result.retry_count == 0      
        
class TestT41IdentityRouterToDaemonIntegration:
    """T4.1: Identity Router connected to Verification Daemon."""

    def test_imports_work_together(self):
        """Identity Router and Daemon must be importable together."""
        from core.identity_router import cosine_similarity, extract_arcface_embedding
        from core.verification_daemon import VerificationDaemon
        assert callable(cosine_similarity)
        assert callable(extract_arcface_embedding)
        assert VerificationDaemon is not None

    def test_daemon_uses_cosine_gate(self):
        """Daemon identity gate must be CosineSimilarityGate instance."""
        from core.cosine_similarity_gate import CosineSimilarityGate
        daemon = VerificationDaemon(enable_logging=False)
        assert isinstance(daemon.identity_gate, CosineSimilarityGate)

    def test_full_identity_verification_pass(self):
        """Matching embeddings must pass through full pipeline."""
        daemon = VerificationDaemon(enable_logging=False)
        ref = np.random.randn(512).astype(np.float32)
        ref = ref / np.linalg.norm(ref)
        result = daemon.verify_single_pass(ref, ref)
        assert result.passed is True
        assert result.final_similarity >= 0.90

    def test_full_identity_verification_fail(self):
        """Non-matching embeddings must fail through full pipeline."""
        daemon = VerificationDaemon(enable_logging=False)
        ref = np.random.randn(512).astype(np.float32)
        ref = ref / np.linalg.norm(ref)
        bad = np.random.randn(512).astype(np.float32)
        bad = bad / np.linalg.norm(bad)
        result = daemon.verify_single_pass(ref, bad)
        assert result.passed is False

    def test_correction_loop_with_identity_router(self):
        """Correction loop must recover using identity router embeddings."""
        daemon = VerificationDaemon(max_retries=3, enable_logging=False)
        ref = np.random.randn(512).astype(np.float32)
        ref = ref / np.linalg.norm(ref)
        bad = np.random.randn(512).astype(np.float32)
        bad = bad / np.linalg.norm(bad)

        def generation_fn(weight):
            return ref

        result = daemon.run(ref, bad, generation_fn=generation_fn)
        assert result.passed is True
        assert result.retry_count == 1

    def test_cosine_similarity_from_identity_router(self):
        """cosine_similarity from identity_router must match gate output."""
        from core.identity_router import cosine_similarity
        ref = np.random.randn(512).astype(np.float32)
        ref = ref / np.linalg.norm(ref)
        daemon = VerificationDaemon(enable_logging=False)
        direct_score = cosine_similarity(ref, ref)
        gate_score = daemon.get_identity_similarity(ref, ref)
        assert abs(direct_score - gate_score) < 1e-5   


class TestT32IdentityGateIntegration:
    """T3.2: Identity check wired from ArcFace extraction into the cosine gate."""

    def test_verify_identity_from_images_uses_arcface_embeddings(self, monkeypatch):
        """Image paths should be converted to embeddings before cosine gating."""
        from core import verification_daemon as vd

        ref = np.random.randn(512).astype(np.float32)
        ref = ref / np.linalg.norm(ref)
        gen = ref.copy()

        calls = []

        def fake_extract(image_path, **kwargs):
            calls.append(image_path)
            return ref if image_path == "ref.png" else gen

        monkeypatch.setattr(vd, "extract_arcface_embedding", fake_extract)

        daemon = VerificationDaemon(enable_logging=False)
        result = daemon.verify_identity_from_images("ref.png", "gen.png")

        assert result.passed is True
        assert result.similarity_score == pytest.approx(1.0, abs=1e-5)
        assert calls == ["ref.png", "gen.png"]

    def test_run_from_images_keeps_existing_workflow(self, monkeypatch):
        """run_from_images should feed extracted embeddings into the normal workflow."""
        from core import verification_daemon as vd

        ref = np.random.randn(512).astype(np.float32)
        ref = ref / np.linalg.norm(ref)
        bad = np.random.randn(512).astype(np.float32)
        bad = bad / np.linalg.norm(bad)

        def fake_extract(image_path, **kwargs):
            return ref if image_path == "ref.png" else bad

        monkeypatch.setattr(vd, "extract_arcface_embedding", fake_extract)

        daemon = VerificationDaemon(max_retries=0, enable_logging=False)
        result = daemon.run_from_images("ref.png", "gen.png")

        assert result.identity_result is not None
        assert result.identity_result.passed is False
        assert result.final_similarity is not None
        
class TestT42KinematicGuardrailToDaemonIntegration:
    """T4.2: Kinematic Guardrail connected to Verification Daemon."""

    def test_imports_work_together(self):
        """Kinematic Guardrail and Daemon must be importable together."""
        from core.kinematic_guardrail import compute_l_physics, compute_velocity_loss
        from core.verification_daemon import VerificationDaemon
        assert callable(compute_l_physics)
        assert callable(compute_velocity_loss)
        assert VerificationDaemon is not None

    def test_verify_kinematic_pass(self):
        """Static pose should pass kinematic verification."""
        daemon = VerificationDaemon(enable_logging=False)
        pose = np.zeros((2, 17, 2), dtype=np.float32)
        pose[1] = 1.0  # small movement within v_max
        result = daemon.verify_kinematic(pose)
        assert result.passed is True
        assert result.total_loss >= 0.0

    def test_verify_kinematic_fail_on_high_velocity(self):
        """Large displacement should fail kinematic verification."""
        daemon = VerificationDaemon(
            kinematic_threshold=0.0,
            enable_logging=False,
        )
        pose = np.zeros((2, 17, 2), dtype=np.float32)
        pose[1] = 999.0  # massive jump
        result = daemon.verify_kinematic(pose)
        assert result.passed is False
        assert result.total_loss > 0.0

    def test_kinematic_halt_before_identity(self):
        """Kinematic failure must halt pipeline before identity check."""
        daemon = VerificationDaemon(
            kinematic_threshold=0.0,
            enable_logging=False,
        )
        ref = np.random.randn(512).astype(np.float32)
        ref = ref / np.linalg.norm(ref)
        pose = np.zeros((2, 17, 2), dtype=np.float32)
        pose[1] = 999.0
        result = daemon.run(ref, ref, pose_keypoints=pose)
        assert result.passed is False
        assert result.kinematic_result is not None
        assert result.identity_result is None

    def test_kinematic_pass_proceeds_to_identity(self):
        """Kinematic pass must allow identity check to proceed."""
        daemon = VerificationDaemon(enable_logging=False)
        ref = np.random.randn(512).astype(np.float32)
        ref = ref / np.linalg.norm(ref)
        pose = np.zeros((2, 17, 2), dtype=np.float32)
        result = daemon.run(ref, ref, pose_keypoints=pose)
        assert result.identity_result is not None
        assert result.passed is True

    def test_kinematic_result_contains_loss_components(self):
        """KinematicResult must contain all loss components."""
        daemon = VerificationDaemon(enable_logging=False)
        pose = np.zeros((2, 17, 2), dtype=np.float32)
        result = daemon.verify_kinematic(pose)
        assert hasattr(result, 'bone_loss')
        assert hasattr(result, 'rom_loss')
        assert hasattr(result, 'velocity_loss')
        assert hasattr(result, 'topology_loss')
        assert hasattr(result, 'total_loss')
        assert hasattr(result, 'max_velocity')   
        
class TestT43EndToEndPipelineIntegration:
    """T4.3: End-to-end pipeline integration — Identity Router + Kinematic Guardrail + Daemon."""

    def test_all_components_importable(self):
        """All three components must be importable together."""
        from core.identity_router import cosine_similarity, extract_arcface_embedding
        from core.kinematic_guardrail import compute_l_physics, compute_velocity_loss
        from core.verification_daemon import VerificationDaemon, create_verification_daemon
        assert all([
            callable(cosine_similarity),
            callable(extract_arcface_embedding),
            callable(compute_l_physics),
            callable(compute_velocity_loss),
            VerificationDaemon is not None,
        ])

    def test_single_pass_identity_and_kinematic_both_pass(self):
        """Both identity and kinematic must pass in single pass."""
        daemon = create_verification_daemon(enable_logging=False)
        ref = np.random.randn(512).astype(np.float32)
        ref = ref / np.linalg.norm(ref)
        pose = np.zeros((2, 17, 2), dtype=np.float32)
        pose[1] = 0.5  # within v_max

        result = daemon.verify_single_pass(ref, ref, pose_keypoints=pose)

        assert result.passed is True
        assert result.identity_result is not None
        assert result.identity_result.passed is True
        assert result.kinematic_result is not None
        assert result.kinematic_result.passed is True

    def test_single_pass_identity_fails_kinematic_passes(self):
        """Identity failure with kinematic pass."""
        daemon = create_verification_daemon(enable_logging=False)
        ref = np.random.randn(512).astype(np.float32)
        ref = ref / np.linalg.norm(ref)
        bad = np.random.randn(512).astype(np.float32)
        bad = bad / np.linalg.norm(bad)
        pose = np.zeros((2, 17, 2), dtype=np.float32)

        result = daemon.verify_single_pass(ref, bad, pose_keypoints=pose)

        assert result.passed is False
        assert result.kinematic_result.passed is True
        assert result.identity_result.passed is False

    def test_single_pass_kinematic_fails_halts_identity(self):
        """Kinematic failure must halt before identity check."""
        daemon = create_verification_daemon(
            kinematic_threshold=0.0,
            enable_logging=False,
        )
        ref = np.random.randn(512).astype(np.float32)
        ref = ref / np.linalg.norm(ref)
        pose = np.zeros((2, 17, 2), dtype=np.float32)
        pose[1] = 999.0

        result = daemon.verify_single_pass(ref, ref, pose_keypoints=pose)

        assert result.passed is False
        assert result.kinematic_result.passed is False
        assert result.identity_result is None

    def test_full_pipeline_no_corrections(self):
        """Full pipeline run without corrections — matching embeddings, valid pose."""
        daemon = create_verification_daemon(enable_logging=False)
        ref = np.random.randn(512).astype(np.float32)
        ref = ref / np.linalg.norm(ref)
        pose = np.zeros((2, 17, 2), dtype=np.float32)

        result = daemon.run(ref, ref, pose_keypoints=pose)

        assert result.passed is True
        assert result.retry_count == 0
        assert result.kinematic_result.passed is True
        assert result.identity_result.passed is True

    def test_pipeline_result_contains_all_fields(self):
        """Result must contain all expected fields."""
        daemon = create_verification_daemon(enable_logging=False)
        ref = np.random.randn(512).astype(np.float32)
        ref = ref / np.linalg.norm(ref)
        pose = np.zeros((2, 17, 2), dtype=np.float32)

        result = daemon.run(ref, ref, pose_keypoints=pose)

        assert result.passed is not None
        assert result.identity_result is not None
        assert result.kinematic_result is not None
        assert result.retry_count == 0
        assert result.max_retries == 5
        assert result.final_similarity is not None
        assert result.latent_rewind_count == 0                                       


class TestT34ToT37UnifiedCorrectionLoop:
    """Validate unified correction flow for rewind/inpaint/regenerate/retry/fallback."""

    def test_kinematic_failure_can_recover_via_generation_fn(self):
        """Kinematic failures should retry and recover when generation returns better pose."""
        daemon = VerificationDaemon(
            max_retries=2,
            kinematic_threshold=0.01,
            enable_logging=False,
        )

        ref = np.random.randn(512).astype(np.float32)
        ref = ref / np.linalg.norm(ref)

        bad_embedding = np.random.randn(512).astype(np.float32)
        bad_embedding = bad_embedding / np.linalg.norm(bad_embedding)

        bad_pose = np.zeros((2, 17, 2), dtype=np.float32)
        bad_pose[1] = 999.0  # force kinematic failure

        good_pose = np.zeros((2, 17, 2), dtype=np.float32)
        good_pose[1] = 0.1

        def generation_fn(config):
            # Return embedding + pose in the richer format supported by the daemon.
            return {
                "embedding": ref,
                "pose_keypoints": good_pose,
            }

        result = daemon.run(
            ref,
            bad_embedding,
            pose_keypoints=bad_pose,
            generation_fn=generation_fn,
        )

        assert result.passed is True
        assert result.retry_count == 1
        assert result.identity_result is not None and result.identity_result.passed
        assert result.kinematic_result is not None and result.kinematic_result.passed

    def test_inpainting_hook_called_during_retry(self):
        """Inpainting callback should be invoked with a generated mask on failure retries."""
        calls = []

        def inpainting_fn(mask):
            calls.append(mask)
            return mask

        daemon = VerificationDaemon(
            max_retries=1,
            kinematic_threshold=0.0,
            enable_logging=False,
            inpainting_fn=inpainting_fn,
        )

        ref = np.random.randn(512).astype(np.float32)
        ref = ref / np.linalg.norm(ref)
        bad = np.random.randn(512).astype(np.float32)
        bad = bad / np.linalg.norm(bad)

        pose = np.zeros((2, 17, 2), dtype=np.float32)
        pose[1] = 999.0

        def generation_fn(weight):
            # Keep it bad to force retry path completion.
            return bad

        _ = daemon.run(ref, bad, pose_keypoints=pose, generation_fn=generation_fn)

        assert len(calls) >= 1
        assert calls[0].ndim == 2

    def test_fallback_prefers_better_kinematic_candidate_when_identity_tied(self):
        """Fallback should pick lower-kinematic-loss candidate when identity scores are equal."""
        daemon = VerificationDaemon(
            max_retries=2,
            kinematic_threshold=0.0,  # force failures
            enable_logging=False,
        )

        ref = np.random.randn(512).astype(np.float32)
        ref = ref / np.linalg.norm(ref)
        # Keep identity score constant and poor in all attempts.
        bad = np.random.randn(512).astype(np.float32)
        bad = bad / np.linalg.norm(bad)

        pose_bad = np.zeros((2, 17, 2), dtype=np.float32)
        pose_bad[1] = 1000.0
        pose_less_bad = np.zeros((2, 17, 2), dtype=np.float32)
        pose_less_bad[1] = 300.0
        pose_better = np.zeros((2, 17, 2), dtype=np.float32)
        pose_better[1] = 50.0

        poses = [pose_less_bad, pose_better]

        def generation_fn(config):
            idx = min(config.retry_count - 1, len(poses) - 1)
            return {
                "embedding": bad,
                "pose_keypoints": poses[idx],
            }

        result = daemon.run(ref, bad, pose_keypoints=pose_bad, generation_fn=generation_fn)

        assert result.passed is False
        assert result.kinematic_result is not None
        # Best fallback should prefer the smallest available loss among failed candidates.
        assert result.kinematic_result.total_loss <= daemon.verify_kinematic(pose_bad).total_loss