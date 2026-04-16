"""
PIVOT Tests — Identity Router (T1.1-T1.5)
Unit tests for Identity Router components including:
- T1.1: ArcFace embedding extraction
- T1.2: IP-Adapter conditioning injection  
- T1.3: Localized masking (SAM/DWPose)
- T1.4: Cosine similarity gate
- T1.5: Unit tests for Identity Router components
"""
import numpy as np
import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path


class TestIdentityRouterImports:
    """Test that all Identity Router components are importable."""

    def test_import_extract_arcface_embedding(self):
        """Should import extract_arcface_embedding."""
        from core.identity_router import extract_arcface_embedding
        assert callable(extract_arcface_embedding)

    def test_import_cosine_similarity(self):
        """Should import cosine_similarity function."""
        from core.identity_router import cosine_similarity
        assert callable(cosine_similarity)

    def test_import_build_localized_identity_mask(self):
        """Should import build_localized_identity_mask."""
        from core.identity_router import build_localized_identity_mask
        assert callable(build_localized_identity_mask)

    def test_import_ip_adapter_components(self):
        """Should import IP-Adapter components."""
        from core.adapters.ip_adapter import (
            IPAdapterProjection,
            IPAdapterConditioning,
            CrossAttention2x2IPAdapter,
            IdentityToIPAdapterBridge,
        )
        assert IPAdapterProjection is not None
        assert IPAdapterConditioning is not None
        assert CrossAttention2x2IPAdapter is not None
        assert IdentityToIPAdapterBridge is not None


class TestCosineSimilarityFunction:
    """Test the cosine_similarity utility function."""

    def test_identical_vectors(self):
        """Identical vectors should return 1.0."""
        from core.identity_router import cosine_similarity
        vec = np.array([1.0, 0.0, 0.0])
        assert cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should return 0.0."""
        from core.identity_router import cosine_similarity
        vec_a = np.array([1.0, 0.0, 0.0])
        vec_b = np.array([0.0, 1.0, 0.0])
        assert cosine_similarity(vec_a, vec_b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        """Opposite vectors should return -1.0."""
        from core.identity_router import cosine_similarity
        vec_a = np.array([1.0, 0.0, 0.0])
        vec_b = np.array([-1.0, 0.0, 0.0])
        assert cosine_similarity(vec_a, vec_b) == pytest.approx(-1.0)

    def test_unnormalized_vectors(self):
        """Should handle unnormalized vectors."""
        from core.identity_router import cosine_similarity
        vec_a = np.array([2.0, 0.0, 0.0])
        vec_b = np.array([1.0, 0.0, 0.0])
        result = cosine_similarity(vec_a, vec_b)
        assert 0.99 <= result <= 1.01

    def test_zero_vector(self):
        """Zero vector should return 0.0."""
        from core.identity_router import cosine_similarity
        vec_a = np.array([0.0, 0.0, 0.0])
        vec_b = np.array([1.0, 0.0, 0.0])
        assert cosine_similarity(vec_a, vec_b) == 0.0


class TestIPAdapterProjection:
    """Test T1.2: IP-Adapter projection layers."""

    def test_projection_initialization(self):
        """Should initialize with correct dimensions."""
        from core.adapters.ip_adapter import IPAdapterProjection
        proj = IPAdapterProjection(
            image_embed_dim=512,
            cross_attention_dim=768,
            num_tokens=4,
        )
        assert proj.image_embed_dim == 512
        assert proj.cross_attention_dim == 768
        assert proj.num_tokens == 4

    def test_projection_forward_global(self):
        """Should project global embeddings correctly."""
        from core.adapters.ip_adapter import IPAdapterProjection
        proj = IPAdapterProjection(
            image_embed_dim=512,
            cross_attention_dim=768,
            num_tokens=4,
        )
        batch_size = 2
        x = torch.randn(batch_size, 512)
        output = proj(x)
        assert output.shape == (batch_size, 4, 768)

    def test_projection_forward_tokens(self):
        """Should project token embeddings correctly."""
        from core.adapters.ip_adapter import IPAdapterProjection
        proj = IPAdapterProjection(
            image_embed_dim=512,
            cross_attention_dim=768,
            num_tokens=4,
        )
        batch_size = 2
        x = torch.randn(batch_size, 4, 512)
        output = proj(x)
        assert output.shape == (batch_size, 4, 768)


class TestIdentityToIPAdapterBridge:
    """Test T1.2: Identity to IP-Adapter bridge module."""

    def test_bridge_initialization(self):
        """Should initialize with correct dimensions."""
        from core.adapters.ip_adapter import IdentityToIPAdapterBridge
        bridge = IdentityToIPAdapterBridge(
            identity_embed_dim=512,
            ip_adapter_embed_dim=768,
        )
        assert bridge.identity_embed_dim == 512
        assert bridge.ip_adapter_embed_dim == 768

    def test_bridge_forward(self):
        """Should convert identity embedding to IP-Adapter format."""
        from core.adapters.ip_adapter import IdentityToIPAdapterBridge
        bridge = IdentityToIPAdapterBridge(
            identity_embed_dim=512,
            ip_adapter_embed_dim=768,
        )
        batch_size = 2
        identity_emb = torch.randn(batch_size, 512)
        output = bridge(identity_emb)
        assert output.shape == (batch_size, 4, 768)

    def test_bridge_output_range(self):
        """Output should be normalized and in reasonable range."""
        from core.adapters.ip_adapter import IdentityToIPAdapterBridge
        bridge = IdentityToIPAdapterBridge(
            identity_embed_dim=512,
            ip_adapter_embed_dim=768,
        )
        identity_emb = torch.randn(1, 512)
        output = bridge(identity_emb)
        assert output.abs().mean() < 10.0


class TestCrossAttention2x2IPAdapter:
    """Test T1.2: Cross-attention with IP-Adapter support."""

    def test_attention_initialization(self):
        """Should initialize cross-attention with IP-Adapter support."""
        from core.adapters.ip_adapter import CrossAttention2x2IPAdapter
        attn = CrossAttention2x2IPAdapter(
            query_dim=768,
            cross_attention_dim=768,
            ip_adapter_dim=768,
        )
        assert attn.ip_adapter_dim == 768
        assert attn.use_ip_adapter is False

    def test_set_ip_adapter(self):
        """Should enable/disable IP-Adapter conditioning."""
        from core.adapters.ip_adapter import CrossAttention2x2IPAdapter
        attn = CrossAttention2x2IPAdapter(
            query_dim=768,
            cross_attention_dim=768,
            ip_adapter_dim=768,
        )
        attn.set_ip_adapter(True, 0.8)
        assert attn.use_ip_adapter is True

    def test_forward_without_ip_adapter(self):
        """Should work without IP-Adapter embeddings."""
        from core.adapters.ip_adapter import CrossAttention2x2IPAdapter
        attn = CrossAttention2x2IPAdapter(
            query_dim=768,
            cross_attention_dim=768,
            ip_adapter_dim=768,
        )
        batch = 2
        seq_len = 16
        query = torch.randn(batch, seq_len, 768)
        context = torch.randn(batch, seq_len, 768)
        output = attn(query, context)
        assert output.shape == query.shape


class TestBuildLocalizedIdentityMask:
    """Test T1.3: Localized masking with DWPose/SAM."""

    def test_mask_creation_with_face_fallback(self):
        """Should create mask when no pose model available."""
        import tempfile
        from core.identity_router import build_localized_identity_mask
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img = np.zeros((256, 256, 3), dtype=np.uint8)
            import cv2
            cv2.imwrite(f.name, img)
            
            with patch('core.identity_router._get_app') as mock_app:
                mock_face = Mock()
                mock_face.bbox = np.array([100, 100, 200, 200])
                mock_app.return_value.get.return_value = [mock_face]
                
                mask = build_localized_identity_mask(f.name)
                
                assert mask.shape == (256, 256)
                assert mask.dtype == np.float32
                assert mask.max() <= 1.0

    def test_mask_returns_valid_shape(self):
        """Should return mask with same dimensions as input image."""
        import tempfile
        from core.identity_router import build_localized_identity_mask
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            height, width = 512, 384
            img = np.zeros((height, width, 3), dtype=np.uint8)
            import cv2
            cv2.imwrite(f.name, img)
            
            with patch('core.identity_router._get_app') as mock_app:
                mock_face = Mock()
                mock_face.bbox = np.array([100, 100, 300, 400])
                mock_app.return_value.get.return_value = [mock_face]
                
                mask = build_localized_identity_mask(f.name)
                
                assert mask.shape == (height, width)

    def test_mask_values_clamped(self):
        """Mask values should be clamped between 0 and 1."""
        import tempfile
        from core.identity_router import build_localized_identity_mask
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            import cv2
            cv2.imwrite(f.name, img)
            
            with patch('core.identity_router._get_app') as mock_app:
                mock_face = Mock()
                mock_face.bbox = np.array([20, 20, 80, 80])
                mock_app.return_value.get.return_value = [mock_face]
                
                mask = build_localized_identity_mask(f.name)
                
                assert mask.min() >= 0.0
                assert mask.max() <= 1.0


class TestLocalizedMaskWithPose:
    """Test T1.3: Localized masking with DWPose keypoints."""

    def test_mask_with_pose_keypoints(self):
        """Should use provided pose keypoints for masking."""
        import tempfile
        from core.identity_router import build_localized_identity_mask
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img = np.zeros((256, 256, 3), dtype=np.uint8)
            import cv2
            cv2.imwrite(f.name, img)
            
            pose_keypoints = np.zeros((17, 3), dtype=np.float32)
            pose_keypoints[0] = [128, 50, 1.0]   # nose
            pose_keypoints[5] = [100, 100, 1.0]  # left_shoulder
            pose_keypoints[6] = [156, 100, 1.0]  # right_shoulder
            pose_keypoints[11] = [110, 180, 1.0] # left_hip
            pose_keypoints[12] = [146, 180, 1.0] # right_hip
            
            mask = build_localized_identity_mask(f.name, pose_keypoints=pose_keypoints)
            
            assert mask.shape == (256, 256)
            assert mask.sum() > 0

    def test_mask_with_custom_parameters(self):
        """Should accept custom mask parameters."""
        import tempfile
        from core.identity_router import build_localized_identity_mask
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img = np.zeros((256, 256, 3), dtype=np.uint8)
            import cv2
            cv2.imwrite(f.name, img)
            
            with patch('core.identity_router._get_app') as mock_app:
                mock_face = Mock()
                mock_face.bbox = np.array([100, 100, 200, 200])
                mock_app.return_value.get.return_value = [mock_face]
                
                mask = build_localized_identity_mask(
                    f.name,
                    mask_dilation=11,
                    bbox_expansion=0.2,
                )
                
                assert mask.shape == (256, 256)


class TestIPAdapterConditioning:
    """Test T1.2: IP-Adapter conditioning module."""

    def test_conditioning_initialization(self):
        """Should initialize with correct parameters."""
        from core.adapters.ip_adapter import IPAdapterConditioning
        cond = IPAdapterConditioning(
            image_encoder_name="clip_vit_l_14",
            cross_attention_dim=768,
            conditioning_scale=0.8,
        )
        assert cond.conditioning_scale == 0.8

    def test_conditioning_forward(self):
        """Should process image embeddings."""
        from core.adapters.ip_adapter import IPAdapterConditioning
        cond = IPAdapterConditioning(
            image_encoder_name="clip_vit_l_14",
            cross_attention_dim=768,
            conditioning_scale=0.8,
        )
        batch = 2
        image_emb = torch.randn(batch, 768)
        output = cond(image_emb)
        assert output.shape[0] == batch


class TestInjectIPAdapter:
    """Test T1.2: IP-Adapter injection function."""

    def test_inject_function_exists(self):
        """inject_ip_adapter_into_unet should be importable."""
        from core.adapters.ip_adapter import inject_ip_adapter_into_unet
        assert callable(inject_ip_adapter_into_unet)


class TestLoadIPAdapter:
    """Test T1.2: IP-Adapter weight loading."""

    def test_load_function_exists(self):
        """load_ip_adapter should be importable."""
        from core.adapters.ip_adapter import load_ip_adapter
        assert callable(load_ip_adapter)


class TestIdentityRouterIntegration:
    """Integration tests for Identity Router pipeline."""

    def test_arcface_to_ip_adapter_bridge(self):
        """Should convert ArcFace embedding to IP-Adapter format."""
        from core.adapters.ip_adapter import IdentityToIPAdapterBridge
        
        bridge = IdentityToIPAdapterBridge(
            identity_embed_dim=512,
            ip_adapter_embed_dim=768,
        )
        
        arcface_embedding = torch.randn(1, 512)
        ip_adapter_emb = bridge(arcface_embedding)
        
        assert ip_adapter_emb.shape == (1, 4, 768)
        assert not torch.isnan(ip_adapter_emb).any()

    def test_identity_gate_with_embedding(self):
        """Should work with identity embeddings."""
        from core.cosine_similarity_gate import CosineSimilarityGate
        
        gate = CosineSimilarityGate(threshold=0.90)
        
        ref_embedding = np.random.randn(512).astype(np.float32)
        ref_embedding = ref_embedding / np.linalg.norm(ref_embedding)
        
        result = gate(ref_embedding, ref_embedding)
        
        assert result.passed is True
        assert result.similarity_score >= 0.90

    def test_mask_generation_pipeline(self):
        """Should generate masks through full pipeline."""
        import tempfile
        from core.identity_router import build_localized_identity_mask
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img = np.ones((256, 256, 3), dtype=np.uint8) * 128
            import cv2
            cv2.imwrite(f.name, img)
            
            with patch('core.identity_router._get_app') as mock_app:
                mock_face = Mock()
                mock_face.bbox = np.array([80, 60, 176, 196])
                mock_app.return_value.get.return_value = [mock_face]
                
                mask = build_localized_identity_mask(f.name)
                
                assert mask is not None
                assert mask.shape == (256, 256)
                assert 0.0 <= mask.min() <= mask.max() <= 1.0