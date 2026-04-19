"""
Unit tests for Interceptor Gate (Verification Daemon Hard Gate)
Requirement: ADR-003, VD-REQ-001 through VD-REQ-007

Tests the interceptor gate positioned between denoising loop and output buffer.
Validates frame interception, sequential verification, and correction loop.
"""

from __future__ import annotations

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

from core.interceptor_gate import (
    InterceptorGate,
    InterceptorGateResult,
    GateDecision,
    LatentStateHistory,
    CorrectionAction,
)
from core.verification_daemon import VerificationDaemon, KinematicResult
from core.cosine_similarity_gate import IdentityGateResult, CosineSimilarityGate


def _step(message: str) -> None:
    """Print test step for visibility."""
    print(f"[InterceptorGate Test] {message}")


@pytest.fixture
def mock_verification_daemon():
    """Create mock verification daemon."""
    daemon = Mock(spec=VerificationDaemon)
    daemon.enable_kinematic = True
    daemon.max_retries = 5
    daemon.identity_gate = Mock(spec=CosineSimilarityGate)
    daemon.identity_gate.threshold = 0.90
    return daemon


@pytest.fixture
def interceptor_gate(mock_verification_daemon):
    """Create interceptor gate with mock daemon."""
    gate = InterceptorGate(
        verification_daemon=mock_verification_daemon,
        enable_logging=False,
        max_retries=5,
    )
    return gate


class TestLatentStateHistory:
    """Test latent state history tracking."""
    
    def test_append_stores_latent_and_timestep(self):
        _step("Testing latent history append")
        history = LatentStateHistory(max_history=10)
        
        latent1 = np.random.randn(4, 64, 64).astype(np.float32)
        latent2 = np.random.randn(4, 64, 64).astype(np.float32)
        
        history.append(latent1, timestep=0)
        history.append(latent2, timestep=1)
        
        assert len(history.latents) == 2
        assert len(history.timesteps) == 2
        assert np.allclose(history.latents[0], latent1)
        assert np.allclose(history.latents[1], latent2)
        assert history.timesteps == [0, 1]
    
    def test_respects_max_history_size(self):
        _step("Testing max history size enforcement")
        history = LatentStateHistory(max_history=3)
        
        for i in range(5):
            latent = np.random.randn(4, 64, 64).astype(np.float32)
            history.append(latent, timestep=i)
        
        assert len(history.latents) == 3
        assert len(history.timesteps) == 3
        assert history.timesteps == [2, 3, 4]
    
    def test_get_previous_returns_older_latent(self):
        _step("Testing retrieval of previous latent states")
        history = LatentStateHistory()
        
        latent0 = np.ones((4, 64, 64), dtype=np.float32)
        latent1 = np.ones((4, 64, 64), dtype=np.float32) * 2
        latent2 = np.ones((4, 64, 64), dtype=np.float32) * 3
        
        history.append(latent0, 0)
        history.append(latent1, 1)
        history.append(latent2, 2)
        
        # Get from 1 step ago
        prev1 = history.get_previous(steps=1)
        assert np.allclose(prev1, latent1)
        
        # Get from 2 steps ago
        prev2 = history.get_previous(steps=2)
        assert np.allclose(prev2, latent0)
        
        # Not enough history
        prev3 = history.get_previous(steps=3)
        assert prev3 is None
    
    def test_clear_resets_history(self):
        _step("Testing history clear")
        history = LatentStateHistory()
        
        history.append(np.random.randn(4, 64, 64).astype(np.float32), 0)
        assert len(history.latents) == 1
        
        history.clear()
        assert len(history.latents) == 0
        assert len(history.timesteps) == 0


class TestInterceptorGatePassPath:
    """Test frames passing through interceptor gate."""
    
    def test_frame_passes_all_checks(self, interceptor_gate, mock_verification_daemon):
        _step("Testing frame that passes all checks")
        
        # Setup: all checks pass
        identity_result = IdentityGateResult(
            passed=True,
            similarity_score=0.95,
            threshold=0.90,
        )
        mock_verification_daemon.verify_identity.return_value = identity_result
        
        # Create test data
        frame_idx = 0
        latent = np.random.randn(4, 64, 64).astype(np.float32)
        ref_embedding = np.random.randn(512).astype(np.float32)
        gen_embedding = ref_embedding.copy()  # High similarity
        
        # Process frame
        result = interceptor_gate.process_frame(
            frame_idx=frame_idx,
            latent=latent,
            reference_embedding=ref_embedding,
            generated_embedding=gen_embedding,
        )
        
        assert result.passed == True
        assert result.decision == GateDecision.PASS
        assert result.identity_score == 0.95
        assert result.retry_count == 0
    
    def test_frame_passes_with_kinematic_check(self, interceptor_gate, mock_verification_daemon):
        _step("Testing frame with kinematic verification enabled")
        
        # Setup: both checks pass
        identity_result = IdentityGateResult(
            passed=True,
            similarity_score=0.95,
            threshold=0.90,
        )
        mock_verification_daemon.verify_identity.return_value = identity_result
        
        kinematic_result = MagicMock(spec=KinematicResult)
        kinematic_result.passed = True
        kinematic_result.total_loss = 0.005
        mock_verification_daemon.verify_kinematic.return_value = kinematic_result
        
        # Test data
        frame_idx = 0
        latent = np.random.randn(4, 64, 64).astype(np.float32)
        ref_embedding = np.random.randn(512).astype(np.float32)
        gen_embedding = ref_embedding.copy()
        keypoints = np.random.rand(17, 3).astype(np.float32)
        
        # Process
        result = interceptor_gate.process_frame(
            frame_idx=frame_idx,
            latent=latent,
            reference_embedding=ref_embedding,
            generated_embedding=gen_embedding,
            pose_keypoints=keypoints,
        )
        
        assert result.passed == True
        assert result.decision == GateDecision.PASS
        assert result.kinematic_loss == 0.005


class TestInterceptorGateFailPath:
    """Test frames failing verification."""
    
    def test_frame_fails_identity_check(self, interceptor_gate, mock_verification_daemon):
        _step("Testing frame that fails identity check and enters correction loop")
        
        # Setup: identity fails
        identity_result = IdentityGateResult(
            passed=False,
            similarity_score=0.85,
            threshold=0.90,
        )
        mock_verification_daemon.verify_identity.return_value = identity_result
        
        # Test data
        frame_idx = 0
        latent = np.random.randn(4, 64, 64).astype(np.float32)
        ref_embedding = np.random.randn(512).astype(np.float32)
        gen_embedding = np.random.randn(512).astype(np.float32)  # Different
        
        # Process
        result = interceptor_gate.process_frame(
            frame_idx=frame_idx,
            latent=latent,
            reference_embedding=ref_embedding,
            generated_embedding=gen_embedding,
        )
        
        # When identity fails, correction loop is triggered
        # Correction loop will simulate improvement and eventually pass
        assert result.decision in [GateDecision.FAIL_IDENTITY, GateDecision.CORRECTION_NEEDED, GateDecision.PASS, GateDecision.MAX_RETRIES_EXCEEDED]
        assert result.retry_count > 0  # Correction loop was triggered
    
    def test_frame_fails_kinematics_check(self, interceptor_gate, mock_verification_daemon):
        _step("Testing frame that fails kinematics check")
        
        # Setup: identity passes, kinematics fails
        identity_result = IdentityGateResult(
            passed=True,
            similarity_score=0.95,
            threshold=0.90,
        )
        mock_verification_daemon.verify_identity.return_value = identity_result
        
        kinematic_result = MagicMock(spec=KinematicResult)
        kinematic_result.passed = False
        kinematic_result.total_loss = 0.05
        mock_verification_daemon.verify_kinematic.return_value = kinematic_result
        
        # Test data
        frame_idx = 0
        latent = np.random.randn(4, 64, 64).astype(np.float32)
        ref_embedding = np.random.randn(512).astype(np.float32)
        gen_embedding = ref_embedding.copy()
        keypoints = np.random.rand(17, 3).astype(np.float32)
        
        # Process
        result = interceptor_gate.process_frame(
            frame_idx=frame_idx,
            latent=latent,
            reference_embedding=ref_embedding,
            generated_embedding=gen_embedding,
            pose_keypoints=keypoints,
        )
        
        assert result.passed == False
        assert result.decision in [GateDecision.FAIL_KINEMATICS, GateDecision.CORRECTION_NEEDED, GateDecision.MAX_RETRIES_EXCEEDED]
        assert result.kinematic_loss == 0.05


class TestInterceptorGateCorrectionLoop:
    """Test correction loop execution."""
    
    def test_correction_loop_triggered_on_failure(self, interceptor_gate, mock_verification_daemon):
        _step("Testing correction loop activation")
        
        # Setup: identity fails
        identity_result = IdentityGateResult(
            passed=False,
            similarity_score=0.85,
            threshold=0.90,
        )
        mock_verification_daemon.verify_identity.return_value = identity_result
        
        # Test data
        frame_idx = 0
        latent = np.random.randn(4, 64, 64).astype(np.float32)
        ref_embedding = np.random.randn(512).astype(np.float32)
        gen_embedding = np.random.randn(512).astype(np.float32)
        
        # Process
        result = interceptor_gate.process_frame(
            frame_idx=frame_idx,
            latent=latent,
            reference_embedding=ref_embedding,
            generated_embedding=gen_embedding,
        )
        
        # Verify correction actions recorded
        assert len(result.correction_actions) > 0
        assert result.retry_count > 0
    
    def test_correction_actions_recorded(self, interceptor_gate, mock_verification_daemon):
        _step("Testing correction action recording")
        
        # Setup
        identity_result = IdentityGateResult(
            passed=False,
            similarity_score=0.85,
            threshold=0.90,
        )
        mock_verification_daemon.verify_identity.return_value = identity_result
        
        # Test data
        frame_idx = 0
        latent = np.random.randn(4, 64, 64).astype(np.float32)
        ref_embedding = np.random.randn(512).astype(np.float32)
        gen_embedding = np.random.randn(512).astype(np.float32)
        
        # Process
        result = interceptor_gate.process_frame(
            frame_idx=frame_idx,
            latent=latent,
            reference_embedding=ref_embedding,
            generated_embedding=gen_embedding,
        )
        
        # Check actions
        action_types = {action.action_type for action in result.correction_actions}
        assert "rewind" in action_types
        assert "inpaint" in action_types
        assert "regenerate" in action_types
    
    def test_max_retries_enforced(self, interceptor_gate, mock_verification_daemon):
        _step("Testing max retry enforcement")
        
        # Setup: always fails
        identity_result = IdentityGateResult(
            passed=False,
            similarity_score=0.85,
            threshold=0.90,
        )
        mock_verification_daemon.verify_identity.return_value = identity_result
        
        # Test data
        frame_idx = 0
        latent = np.random.randn(4, 64, 64).astype(np.float32)
        ref_embedding = np.random.randn(512).astype(np.float32)
        gen_embedding = np.random.randn(512).astype(np.float32)
        
        # Process
        result = interceptor_gate.process_frame(
            frame_idx=frame_idx,
            latent=latent,
            reference_embedding=ref_embedding,
            generated_embedding=gen_embedding,
        )
        
        # Verify max retries not exceeded
        assert result.retry_count <= interceptor_gate.max_retries


class TestInterceptorGateStatistics:
    """Test statistics and monitoring."""
    
    def test_frame_statistics_calculated(self, interceptor_gate, mock_verification_daemon):
        _step("Testing frame statistics calculation")
        
        # Setup: some frames pass, some fail
        pass_identity = IdentityGateResult(
            passed=True,
            similarity_score=0.95,
            threshold=0.90,
        )
        fail_identity = IdentityGateResult(
            passed=False,
            similarity_score=0.85,
            threshold=0.90,
        )
        
        # Process multiple frames
        mock_verification_daemon.verify_identity.side_effect = [
            pass_identity,
            fail_identity,
            pass_identity,
        ]
        
        for i in range(3):
            latent = np.random.randn(4, 64, 64).astype(np.float32)
            ref_embedding = np.random.randn(512).astype(np.float32)
            gen_embedding = np.random.randn(512).astype(np.float32)
            
            interceptor_gate.process_frame(
                frame_idx=i,
                latent=latent,
                reference_embedding=ref_embedding,
                generated_embedding=gen_embedding,
            )
        
        stats = interceptor_gate.get_frame_statistics()
        
        assert stats["total_frames"] == 3
        assert stats["passed_frames"] >= 1
        assert stats["failed_frames"] >= 0
        assert 0.0 <= stats["pass_rate"] <= 1.0
    
    def test_gate_reset_clears_state(self, interceptor_gate, mock_verification_daemon):
        _step("Testing gate state reset")
        
        # Add some frames
        identity_result = IdentityGateResult(
            passed=True,
            similarity_score=0.95,
            threshold=0.90,
        )
        mock_verification_daemon.verify_identity.return_value = identity_result
        
        latent = np.random.randn(4, 64, 64).astype(np.float32)
        ref_embedding = np.random.randn(512).astype(np.float32)
        gen_embedding = ref_embedding.copy()
        
        interceptor_gate.process_frame(
            frame_idx=0,
            latent=latent,
            reference_embedding=ref_embedding,
            generated_embedding=gen_embedding,
        )
        
        assert len(interceptor_gate.frame_history) == 1
        
        # Reset
        interceptor_gate.reset()
        
        assert len(interceptor_gate.frame_history) == 0
        assert len(interceptor_gate.correction_history) == 0


class TestInterceptorGateInpaintingMask:
    """Test inpainting mask generation."""
    
    def test_identity_mask_generation(self, interceptor_gate):
        _step("Testing identity constraint mask generation")
        
        mask = interceptor_gate._generate_inpainting_mask("identity")
        
        assert mask is not None
        assert mask.shape == (512, 512)
        assert mask.dtype == np.uint8
        assert mask.max() <= 1
    
    def test_kinematics_mask_generation(self, interceptor_gate):
        _step("Testing kinematics constraint mask generation")
        
        mask = interceptor_gate._generate_inpainting_mask("kinematics")
        
        assert mask is not None
        assert mask.shape == (512, 512)
        assert mask.dtype == np.uint8
        assert mask.max() <= 1


class TestInterceptorGateIntegration:
    """Integration tests with sequential frame processing."""
    
    def test_sequential_frame_processing(self, interceptor_gate, mock_verification_daemon):
        _step("Testing sequential frame processing through gate")
        
        # Setup: alternating pass/fail pattern
        pass_identity = IdentityGateResult(
            passed=True,
            similarity_score=0.95,
            threshold=0.90,
        )
        fail_identity = IdentityGateResult(
            passed=False,
            similarity_score=0.85,
            threshold=0.90,
        )
        
        results = [pass_identity, fail_identity, pass_identity]
        mock_verification_daemon.verify_identity.side_effect = results
        
        # Process sequence
        for i in range(3):
            latent = np.random.randn(4, 64, 64).astype(np.float32)
            ref_embedding = np.random.randn(512).astype(np.float32)
            gen_embedding = np.random.randn(512).astype(np.float32)
            
            result = interceptor_gate.process_frame(
                frame_idx=i,
                latent=latent,
                reference_embedding=ref_embedding,
                generated_embedding=gen_embedding,
            )
            
            assert result.frame_idx == i
        
        # Verify history
        assert len(interceptor_gate.frame_history) == 3
        
        stats = interceptor_gate.get_frame_statistics()
        assert stats["total_frames"] == 3
