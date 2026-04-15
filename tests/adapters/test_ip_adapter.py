"""
P.I.V.O.T. Adapters Module Tests
Test IP-Adapter conditioning injection
"""
import torch
import pytest
from core.adapters import (
    IPAdapterProjection,
    IPAdapterConditioning,
    CrossAttention2x2IPAdapter,
    IdentityToIPAdapterBridge,
    replace_cross_attention_with_ipadapter,
)


class TestIPAdapterProjection:
    def test_projection_output_shape(self):
        proj = IPAdapterProjection(
            image_embed_dim=768,
            cross_attention_dim=768,
            num_tokens=4,
        )
        x = torch.randn(2, 4, 768)
        out = proj(x)
        assert out.shape == (2, 4, 768)
        
    def test_projection_preserves_batch(self):
        proj = IPAdapterProjection(num_tokens=8)
        x = torch.randn(1, 8, 768)
        out = proj(x)
        assert out.shape[0] == 1
        
    def test_projection_different_dims(self):
        proj = IPAdapterProjection(
            image_embed_dim=512,
            cross_attention_dim=1024,
            num_tokens=4,
        )
        x = torch.randn(2, 4, 512)
        out = proj(x)
        assert out.shape == (2, 4, 1024)


class TestIPAdapterConditioning:
    def test_conditioning_scale_default(self):
        cond = IPAdapterConditioning(cross_attention_dim=768)
        assert cond.conditioning_scale == 0.7
        
    def test_conditioning_scale_override(self):
        cond = IPAdapterConditioning(cross_attention_dim=768)
        x = torch.randn(2, 4, 768)
        out = cond(x, scale=0.5)
        assert out.shape == x.shape
        
    def test_set_conditioning_scale(self):
        cond = IPAdapterConditioning(cross_attention_dim=768)
        cond.set_conditioning_scale(0.3)
        assert cond.conditioning_scale == 0.3


class TestCrossAttentionWithIPAdapter:
    def test_basic_forward(self):
        attn = CrossAttention2x2IPAdapter(
            query_dim=320,
            heads=8,
            dim_head=40,
            cross_attention_dim=768,
            ip_adapter_dim=768,
        )
        attn.set_ip_adapter(False)
        x = torch.randn(2, 16, 320)
        context = torch.randn(2, 77, 768)
        out = attn(x, context=context)
        assert out.shape == x.shape
        
    def test_with_ip_adapter(self):
        attn = CrossAttention2x2IPAdapter(
            query_dim=320,
            heads=8,
            dim_head=40,
            cross_attention_dim=768,
            ip_adapter_dim=768,
        )
        attn.set_ip_adapter(True, 0.7)
        x = torch.randn(2, 16, 320)
        context = torch.randn(2, 77, 768)
        ip_emb = torch.randn(2, 4, 768)
        out = attn(x, context=context, ip_embeddings=ip_emb)
        assert out.shape == x.shape
        
    def test_ip_scale(self):
        attn = CrossAttention2x2IPAdapter(
            query_dim=320,
            cross_attention_dim=768,
            ip_adapter_dim=768,
        )
        attn.set_ip_adapter(True, 0.5)
        assert attn.ip_scale == 0.5


class TestCrossAttention2x2IPAdapter:
    def test_basic_forward(self):
        attn = CrossAttention2x2IPAdapter(
            query_dim=320,
            heads=8,
            dim_head=40,
            cross_attention_dim=768,
            ip_adapter_dim=768,
        )
        x = torch.randn(2, 16, 320)
        context = torch.randn(2, 77, 768)
        out = attn(x, context=context)
        assert out.shape == x.shape
        
    def test_ip_adapter_toggle(self):
        attn = CrossAttention2x2IPAdapter(
            query_dim=320,
            cross_attention_dim=768,
            ip_adapter_dim=768,
        )
        x = torch.randn(2, 16, 320)
        context = torch.randn(2, 77, 768)
        out = attn(x, context=context)
        assert out.shape == x.shape
        
    def test_ip_adapter_toggle(self):
        attn = CrossAttention2x2IPAdapter(
            query_dim=320,
            cross_attention_dim=768,
            ip_adapter_dim=768,
        )
        attn.set_ip_adapter(True, scale=0.8)
        assert attn.use_ip_adapter == True
        assert attn.ip_scale == 0.8
        
        attn.set_ip_adapter(False)
        assert attn.use_ip_adapter == False


class TestIdentityToIPAdapterBridge:
    def test_bridge_output_shape(self):
        bridge = IdentityToIPAdapterBridge(
            identity_embed_dim=512,
            ip_adapter_embed_dim=768,
            num_tokens=4,
        )
        x = torch.randn(2, 512)
        out = bridge(x)
        assert out.shape == (2, 4, 768)
        
    def test_single_embedding(self):
        bridge = IdentityToIPAdapterBridge()
        x = torch.randn(512)
        out = bridge(x)
        assert out.shape == (1, 4, 768)
        
    def test_mlp_vs_linear(self):
        bridge_mlp = IdentityToIPAdapterBridge(use_mlp=True)
        bridge_linear = IdentityToIPAdapterBridge(use_mlp=False)
        
        x = torch.randn(2, 512)
        
        out_mlp = bridge_mlp(x)
        out_linear = bridge_linear(x)
        
        assert out_mlp.shape == out_linear.shape


class TestReplaceCrossAttention:
    def test_module_replacement(self):
        import torch.nn as nn
        
        class DummyAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = torch.nn.Linear(320, 320)
                self.to_k = torch.nn.Linear(768, 320)
                self.to_v = torch.nn.Linear(768, 320)
                self.to_out = torch.nn.Linear(320, 320)
                self.heads = 8
                
        class DummyUNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn1 = DummyAttention()
                
        unet = DummyUNet()
        modified = replace_cross_attention_with_ipadapter(unet, ip_adapter_dim=768)
        
        assert hasattr(modified.attn1, 'ip_proj_k')