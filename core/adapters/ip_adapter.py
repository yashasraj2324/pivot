"""
P.I.V.O.T. — IP-Adapter Implementation
Task: T1.2 - Implement IP-Adapter conditioning injection into U-Net cross-attention layers

This module provides IP-Adapter conditioning for injecting reference image embeddings
into the U-Net's cross-attention layers. Based on the ip-adapter_plus approach for
Stable Diffusion-based video models.

Reference: https://huggingface.co/h94/IP-Adapter
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class IPAdapterProjection(nn.Module):
    """
    Projection layers that map IP-Adapter image embeddings to cross-attention dimensions.
    
    The IP-Adapter uses a lightweight image encoder followed by projection layers that
    transform image features into the same dimension space as the cross-attention
    keys and values.
    """
    
    def __init__(
        self,
        image_embed_dim: int = 768,
        cross_attention_dim: int = 768,
        num_tokens: int = 4,
    ):
        """
        Initialize IP-Adapter projection.
        
        Args:
            image_embed_dim: Dimension of the image encoder output embeddings
            cross_attention_dim: Dimension of U-Net cross-attention space
            num_tokens: Number of tokens for IP-Adapter (typically 4 for plus variant)
        """
        super().__init__()
        
        self.image_embed_dim = image_embed_dim
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        
        self.proj_global = nn.Linear(image_embed_dim, num_tokens * cross_attention_dim)
        self.proj_token = nn.Linear(image_embed_dim, cross_attention_dim)
        self.norm = nn.LayerNorm(cross_attention_dim)
        
    def forward(self, image_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Project image embeddings to cross-attention dimensions.
        
        Supports two input modes:
        - Global/pooled: (batch, image_embed_dim) -> expands to (batch, num_tokens, cross_attention_dim)
        - Patch features: (batch, num_tokens, image_embed_dim) -> projects to (batch, num_tokens, cross_attention_dim)
        
        Args:
            image_embeddings: Shape (batch_size, image_embed_dim) or (batch_size, num_tokens, image_embed_dim)
            
        Returns:
            Projected embeddings: Shape (batch_size, num_tokens, cross_attention_dim)
        """
        if image_embeddings.dim() == 2:
            x = self.proj_global(image_embeddings)
            x = x.reshape(x.shape[0], self.num_tokens, -1)
            x = x.reshape(-1, self.cross_attention_dim)
            x = self.norm(x)
            x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        else:
            batch_size, num_tokens, embed_dim = image_embeddings.shape
            x = image_embeddings.reshape(batch_size * num_tokens, embed_dim)
            x = self.proj_token(x)
            x = x.reshape(batch_size, num_tokens, -1)
            x = x.reshape(-1, self.cross_attention_dim)
            x = self.norm(x)
            x = x.reshape(batch_size, num_tokens, -1)
        
        return x


class IPAdapterConditioning(nn.Module):
    """
    IP-Adapter conditioning module for injecting reference image features
    into the U-Net cross-attention layers.
    
    This implements the ip-adapter_plus approach which uses:
    - Image encoder for extracting features from reference image
    - Projection layers to match cross-attention dimensions
    - Skip connection with configurable scaling factor
    """
    
    def __init__(
        self,
        image_encoder_name: str = "clip_vit_l_14",
        cross_attention_dim: int = 768,
        conditioning_scale: float = 0.7,
        num_tokens: int = 4,
    ):
        """
        Initialize IP-Adapter conditioning.
        
        Args:
            image_encoder_name: Name of the image encoder (clip_vit_l_14, etc.)
            cross_attention_dim: Dimension of U-Net cross-attention
            conditioning_scale: Strength of IP-Adapter influence (default 0.7)
            num_tokens: Number of IP-Adapter tokens
        """
        super().__init__()
        
        self.cross_attention_dim = cross_attention_dim
        self.conditioning_scale = conditioning_scale
        self.num_tokens = num_tokens
        
        self.projection = IPAdapterProjection(
            image_embed_dim=768,
            cross_attention_dim=cross_attention_dim,
            num_tokens=num_tokens,
        )
        
    def forward(
        self,
        image_embeddings: torch.Tensor,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Process image embeddings for cross-attention injection.
        
        Args:
            image_embeddings: Shape (batch_size, num_tokens, image_embed_dim)
            scale: Optional override for conditioning scale
            
        Returns:
            Processed embeddings ready for cross-attention injection
        """
        if scale is None:
            scale = self.conditioning_scale
            
        projected = self.projection(image_embeddings)
        return projected
    
    def set_conditioning_scale(self, scale: float) -> None:
        """Update the conditioning scale."""
        self.conditioning_scale = scale


class CrossAttention2x2IPAdapter(nn.Module):
    """
    Diffusers-compatible cross-attention with IP-Adapter support.
    
    This class extends the standard diffusers CrossAttention2x2 to accept
    IP-Adapter embeddings as additional keys/values.
    
    Used for Stable Diffusion and similar architectures.
    """
    
    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = None,
        ip_adapter_dim: Optional[int] = None,
    ):
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim or query_dim
        
        super().__init__()
        
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = inner_dim
        
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=False)
        
        self.to_out = nn.Linear(inner_dim, query_dim)
        
        assert ip_adapter_dim is not None, "ip_adapter_dim required for IP-Adapter layers"
        self.ip_adapter_dim = ip_adapter_dim
        self.ip_proj_k = nn.Linear(ip_adapter_dim, inner_dim, bias=False)
        self.ip_proj_v = nn.Linear(ip_adapter_dim, inner_dim, bias=False)
        
        self.use_ip_adapter = False
        self.ip_scale = 1.0
        
    def set_ip_adapter(self, enabled: bool = True, scale: float = 1.0) -> None:
        """Enable/disable IP-Adapter conditioning."""
        self.use_ip_adapter = enabled
        self.ip_scale = scale
        
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        ip_embeddings: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional IP-Adapter conditioning.
        
        Args:
            x: Hidden states, shape (batch, seq_len, query_dim)
            context: Cross-attention context, shape (batch, context_len, cross_attention_dim)
            ip_embeddings: IP-Adapter embeddings, shape (batch, ip_tokens, ip_adapter_dim)
            mask: Optional attention mask
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        query = self.to_q(x)
        
        if context is not None:
            key = self.to_k(context)
            value = self.to_v(context)
        else:
            key = self.to_k(x)
            value = self.to_v(x)
            
        query = query.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
        
        text_attention = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
        text_attention = text_attention.transpose(1, 2).reshape(batch_size, seq_len, self.inner_dim)
        
        if ip_embeddings is not None and self.use_ip_adapter:
            ip_query = self.to_q(x)
            ip_key = self.ip_proj_k(ip_embeddings)
            ip_value = self.ip_proj_v(ip_embeddings)
            
            ip_query = ip_query.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
            ip_key = ip_key.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
            ip_value = ip_value.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
            
            ip_attention = F.scaled_dot_product_attention(ip_query, ip_key, ip_value)
            ip_attention = ip_attention.transpose(1, 2).reshape(batch_size, seq_len, self.inner_dim)
            
            text_attention = text_attention + self.ip_scale * ip_attention
        
        return self.to_out(text_attention)


def inject_ip_adapter_into_unet(
    unet: nn.Module,
    ip_adapter_state_dict: Optional[dict] = None,
    conditioning_scale: float = 0.7,
    target_submodules: Optional[list[str]] = None,
    ip_adapter_dim: int = 768,
) -> nn.Module:
    """
    Inject IP-Adapter conditioning weights into a U-Net model.
    
    NOTE: This replaces entire attention modules. For production use with
    LoRA/ControlNet, prefer diffusers' set_attn_processor() approach instead.
    
    Args:
        unet: The U-Net model to modify
        ip_adapter_state_dict: Optional state dict containing IP-Adapter weights
        conditioning_scale: Initial conditioning scale (default 0.7)
        target_submodules: List of submodule names to inject into
        ip_adapter_dim: Dimension of IP-Adapter embeddings
        
    Returns:
        Modified U-Net with IP-Adapter support
    """
    if target_submodules is None:
        target_submodules = ["attn2"]
        
    def replace_module(module: nn.Module, name: str) -> nn.Module:
        for sub_name, child in module.named_children():
            full_name = f"{name}.{sub_name}" if name else sub_name
            if any(t in full_name for t in target_submodules):
                if hasattr(child, 'to_k') and hasattr(child, 'to_v'):
                    new_module = CrossAttention2x2IPAdapter(
                        query_dim=child.to_q.in_features,
                        heads=getattr(child, 'heads', 8),
                        dim_head=child.to_q.out_features // getattr(child, 'heads', 8),
                        cross_attention_dim=child.to_k.in_features,
                        ip_adapter_dim=ip_adapter_dim,
                    )
                    new_module.load_state_dict(child.state_dict(), strict=False)
                    new_module.set_ip_adapter(True, conditioning_scale)
                    
                    if ip_adapter_state_dict is not None:
                        for key, value in ip_adapter_state_dict.items():
                            if 'ip_proj_k' in key and 'to_k_ip' in key:
                                new_module.ip_proj_k.weight.data = value
                            elif 'ip_proj_v' in key and 'to_v_ip' in key:
                                new_module.ip_proj_v.weight.data = value
                    
                    setattr(module, sub_name, new_module)
            else:
                replace_module(child, full_name)
    
    replace_module(unet, "")
    return unet


def replace_cross_attention_with_ipadapter(
    unet: nn.Module,
    ip_adapter_dim: int = 768,
) -> nn.Module:
    """
    Replace all cross-attention layers in a diffusers U-Net with IP-Adapter versions.
    
    Args:
        unet: Diffusers UNet2DConditionModel or similar
        ip_adapter_dim: Dimension of IP-Adapter embeddings
        
    Returns:
        Modified U-Net with IP-Adapter enabled cross-attention
    """
    for name, module in unet.named_modules():
        if isinstance(module, nn.Module):
            module_class = module.__class__.__name__
            if "CrossAttention" in module_class or "Attention" in module_class:
                if hasattr(module, 'to_k') and hasattr(module, 'to_v') and hasattr(module, 'to_q'):
                    if not hasattr(module, 'ip_proj_k'):
                        is_cross_attention = module.to_k.in_features != module.to_q.in_features
                        if not is_cross_attention:
                            continue
                        old_state = module.state_dict()
                        new_module = CrossAttention2x2IPAdapter(
                            query_dim=module.to_q.in_features,
                            heads=getattr(module, 'heads', 8),
                            dim_head=module.to_q.out_features // getattr(module, 'heads', 8),
                            cross_attention_dim=module.to_k.in_features,
                            ip_adapter_dim=ip_adapter_dim,
                        )
                        new_module.load_state_dict(old_state, strict=False)
                        
                        parent_name = ".".join(name.split(".")[:-1])
                        child_name = name.split(".")[-1]
                        parent = unet.get_submodule(parent_name) if parent_name else unet
                        setattr(parent, child_name, new_module)
                        
    return unet


def load_ip_adapter(
    model_id: str = "h94/IP-Adapter",
    subfolder: str = "models",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple[dict, dict, dict]:
    """
    Load pretrained IP-Adapter weights from Hugging Face.
    
    Returns three state dicts:
    - image_proj: Projection layer weights
    - ip_adapter: Per-layer IP attention weights
    - image_encoder: CLIP image encoder (separate model, not in state dict)
    
    The CLIP image encoder (openai/clip-vit-large-patch14) must be loaded separately.
    
    Args:
        model_id: Hugging Face model ID
        subfolder: Subfolder for model weights
        device: Device to load weights to
        
    Returns:
        Tuple of (image_proj_state_dict, ip_adapter_state_dict, image_encoder_config)
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError("huggingface_hub required. Install: pip install huggingface_hub")
    
    ip_adapter_path = hf_hub_download(
        repo_id=model_id,
        filename="ip-adapter_sd15.bin",
        subfolder=subfolder,
    )
    
    state_dict = torch.load(ip_adapter_path, map_location=device)
    
    image_proj_state_dict = {}
    ip_adapter_state_dict = {}
    
    for key, value in state_dict.items():
        if key.startswith("image_proj_model."):
            image_proj_state_dict[key.replace("image_proj_model.", "")] = value
        elif key.startswith("ip_adapter."):
            ip_adapter_state_dict[key] = value
    
    image_encoder_config = {
        "model_id": "openai/clip-vit-large-patch14",
        "type": "CLIPVisionModelWithProjection",
    }
    
    return image_proj_state_dict, ip_adapter_state_dict, image_encoder_config


def load_clip_image_encoder(
    model_id: str = "openai/clip-vit-large-patch14",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> nn.Module:
    """
    Load CLIP image encoder for IP-Adapter.
    
    Args:
        model_id: Hugging Face model ID for CLIP
        device: Device to load model to
        
    Returns:
        CLIPVisionModelWithProjection
    """
    try:
        from transformers import CLIPVisionModelWithProjection
    except ImportError:
        raise ImportError("transformers required. Install: pip install transformers")
    
    return CLIPVisionModelWithProjection.from_pretrained(model_id).to(device)


class IdentityToIPAdapterBridge(nn.Module):
    """
    Bridge module that converts ArcFace identity embeddings to IP-Adapter format.
    
    This module maps the 512-dimensional ArcFace embedding from the Identity Router
    to the dimensions expected by the IP-Adapter cross-attention injection.
    
    The ArcFace embedding is projected to match the CLIP image encoder dimension
    (typically 768) for compatibility with IP-Adapter conditioning.
    """
    
    def __init__(
        self,
        identity_embed_dim: int = 512,
        ip_adapter_embed_dim: int = 768,
        num_tokens: int = 4,
        use_mlp: bool = True,
    ):
        """
        Initialize the identity to IP-Adapter bridge.
        
        Args:
            identity_embed_dim: Dimension of ArcFace embeddings (default 512)
            ip_adapter_embed_dim: Target dimension for IP-Adapter (default 768)
            num_tokens: Number of tokens to expand to (default 4)
            use_mlp: Use MLP projection vs simple linear projection
        """
        super().__init__()
        
        self.identity_embed_dim = identity_embed_dim
        self.ip_adapter_embed_dim = ip_adapter_embed_dim
        self.num_tokens = num_tokens
        
        if use_mlp:
            self.projection = nn.Sequential(
                nn.Linear(identity_embed_dim, ip_adapter_embed_dim),
                nn.GELU(),
                nn.Linear(ip_adapter_embed_dim, ip_adapter_embed_dim),
            )
        else:
            self.projection = nn.Linear(identity_embed_dim, ip_adapter_embed_dim)
            
        self.output_norm = nn.LayerNorm(ip_adapter_embed_dim)
        
    def forward(self, identity_embedding: torch.Tensor) -> torch.Tensor:
        """
        Convert identity embedding to IP-Adapter format.
        
        Args:
            identity_embedding: ArcFace embedding, shape (batch, identity_embed_dim)
                                or (identity_embed_dim,) for single embedding
            
        Returns:
            IP-Adapter formatted embedding, shape (batch, num_tokens, ip_adapter_embed_dim)
        """
        if identity_embedding.dim() == 1:
            identity_embedding = identity_embedding.unsqueeze(0)
            
        batch_size = identity_embedding.shape[0]
        
        projected = self.projection(identity_embedding)
        
        projected = self.output_norm(projected)
        
        tokens = projected.unsqueeze(1).repeat(1, self.num_tokens, 1)
        
        return tokens


class IdentityConditioningPipeline:
    """
    Complete pipeline combining Identity Router with IP-Adapter.
    
    This pipeline integrates:
    1. ArcFace embedding extraction (from identity_router.py)
    2. Identity to IP-Adapter bridge
    3. IP-Adapter conditioning injection
    """
    
    def __init__(
        self,
        unet: nn.Module,
        identity_embed_dim: int = 512,
        ip_adapter_embed_dim: int = 768,
        conditioning_scale: float = 0.7,
    ):
        """
        Initialize identity conditioning pipeline.
        
        Args:
            unet: The U-Net model to inject IP-Adapter into
            identity_embed_dim: Dimension of ArcFace embeddings
            ip_adapter_embed_dim: Dimension expected by IP-Adapter
            conditioning_scale: Strength of conditioning
        """
        self.unet = unet
        self.conditioning_scale = conditioning_scale
        
        self.bridge = IdentityToIPAdapterBridge(
            identity_embed_dim=identity_embed_dim,
            ip_adapter_embed_dim=ip_adapter_embed_dim,
        )
        
        self.unet = replace_cross_attention_with_ipadapter(
            unet,
            ip_adapter_dim=ip_adapter_embed_dim,
        )
        
        self._ip_embeddings_cache = None
        
    def set_identity(self, identity_embedding: torch.Tensor) -> None:
        """
        Set the identity embedding for conditioning.
        
        Args:
            identity_embedding: ArcFace embedding tensor
        """
        self._ip_embeddings_cache = self.bridge(identity_embedding)
        
    def set_conditioning_scale(self, scale: float) -> None:
        """Update conditioning scale."""
        self.conditioning_scale = scale
        
    def __call__(
        self,
        latents: torch.Tensor,
        timestep: int,
        encoder_hidden_states: torch.Tensor,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Forward pass with identity conditioning.
        
        Args:
            latents: Input latents
            timestep: Diffusion timestep
            encoder_hidden_states: Text encoder hidden states
            guidance_scale: CFG guidance scale (1.0 = no CFG)
            
        Returns:
            Output from U-Net
        """
        ip_embeddings = self._ip_embeddings_cache
        
        for name, module in self.unet.named_modules():
            if hasattr(module, 'set_ip_adapter'):
                if ip_embeddings is not None:
                    module.set_ip_adapter(True, self.conditioning_scale)
                else:
                    module.set_ip_adapter(False)
        
        if guidance_scale == 1.0 or ip_embeddings is None:
            if ip_embeddings is not None:
                return self.unet(
                    latents,
                    timestep,
                    encoder_hidden_states,
                    ip_embeddings=ip_embeddings,
                )
            return self.unet(latents, timestep, encoder_hidden_states)
        
        null_ip_embeddings = torch.zeros_like(ip_embeddings)
        ip_embeddings_cfg = torch.cat([null_ip_embeddings, ip_embeddings], dim=0)
        
        latents_cfg = torch.cat([latents, latents], dim=0)
        encoder_hidden_states_cfg = torch.cat(
            [encoder_hidden_states, encoder_hidden_states], dim=0
        )
        
        output = self.unet(
            latents_cfg,
            timestep,
            encoder_hidden_states_cfg,
            ip_embeddings=ip_embeddings_cfg,
        )
        
        cond, uncond = output.chunk(2, dim=0)
        return uncond + guidance_scale * (cond - uncond)