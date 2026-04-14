# ADR-004: Data Flow Between Components with Specific Tensor Shapes and Interfaces
**Date**: 2026-04-14 | **Status**: Proposed
**Author**: ai-architect agent

## Context
Detailed specification of data flow and tensor interfaces between all components in the P.I.V.O.T. Phase 1 pipeline, based on architecture.md and phase1_prd.md.

## Component Interfaces

### 1. Identity Router Interface Contracts

#### Reference Extraction (InsightFace)
```python
# Input:
#   reference_image: [B, 3, H, W] uint8  — RGB reference photograph
# Output:
#   identity_vector: [B, 512] float32   — ArcFace embedding (L2 normalized)
#   face_bbox:       [B, 4] float32     — [x1, y1, x2, y2] in pixel coordinates
```

#### Identity Conditioning (IP-Adapter Plus)
```python
# Inputs to IP-Adapter:
#   identity_vector: [B, 512] float32   — From reference extraction
#   text_embeds:     [B, N, D] float32  — From prompt encoder (CLIP text)
#   unet_features:   List[Tensor]       — U-Net intermediate features from denoising
# Outputs:
#   conditioned_features: List[Tensor]  — IP-Adapter modulated U-Net features
#                       Same shapes as unet_features input
```

#### Localized Masking (SAM/DWPose)
```python
# Input:
#   latent_video:    [B, C, T, H, W] float32  — Current denoising latent
#   pose_keypoints:  [B, T, 17, 2] float32    — From DWPose (x, y coordinates)
# Output:
#   character_mask:  [B, 1, T, H, W] float32  — Binary mask (0-1) for character silhouette
```

#### Cosine Similarity Gate
```python
# Input:
#   identity_vector: [B, 512] float32   — Reference ArcFace embedding
#   frame_embeds:    [B, T, 512] float32 — Per-frame embeddings from generation
# Output:
#   similarity_scores: [B, T] float32   — Cosine similarity per frame
#   gate_decision:     [B, T] bool      — True if similarity >= 0.90
```

### 2. Kinematic Guardrail Interface Contracts

#### Pose Estimation (DWPose)
```python
# Input:
#   latent_video:    [B, C, T, H, W] float32  — Current denoising latent
# Output:
#   pose_keypoints:  [B, T, 17, 2] float32    — (x, y) coordinates for 17 keypoints
#   pose_scores:     [B, T, 17] float32       — Confidence scores per keypoint
```

#### Bone Length Invariance
```python
# Input:
#   pose_keypoints:  [B, T, 17, 2] float32    — From DWPose
#   bone_pairs:      List[Tuple[int, int]]    — Predefined connected joint pairs
# Output:
#   bone_lengths:    [B, T, P] float32        — P = number of bone pairs
#   bone_loss:       float32                  — L_bone = Σ_t || d_t − d_{t-1} ||^2
```

#### Biomechanical Range of Motion
```python
# Input:
#   pose_keypoints:  [B, T, 17, 2] float32    — From DWPose
#   joint_defs:      Dict[str, Tuple[int, int, str]] — Joint definitions (parent, child, type)
#   joint_limits:    Dict[str, Tuple[float, float]]  — Min/max angles in degrees
# Output:
#   joint_angles:    [B, T, J] float32        — J = number of joints
#   rom_violations:  [B, T, J] bool           — True if angle outside limits
#   rom_penalty:     float32                  — Sum of max(0, angle - limit) violations
```

#### Temporal Velocity Limits
```python
# Input:
#   pose_keypoints:  [B, T, 17, 2] float32    — From DWPose
#   frame_dt:        float32                  — Time between frames (default 1.0)
#   v_max:           float32                  — Maximum velocity (default 2.0)
# Output:
#   velocities:      [B, T, 17, 2] float32    — Velocity vectors per keypoint
#   velocity_mags:   [B, T, 17] float32       — Magnitude of velocity vectors
#   velocity_violations: [B, T, 17] bool      — True if magnitude > v_max
#   velocity_penalty: float32                 — Sum of max(0, ||v|| - v_max)
```

#### Rigid Body Topology
```python
# Input:
#   latent_video:    [B, C, T, H, W] float32  — Current denoising latent
#   rigid_masks:     [B, 3, T, H, W] float32  — Binary masks for head/torso/pelvis
# Output:
#   topology_score:  float32                  — SSIM-based rigid region consistency
#   topology_violation: bool                  — True if score < threshold
```

### 3. Verification Daemon Interface Contracts

#### Interceptor Gate
```python
# Input:
#   denoised_latent: [B, C, T, H, W] float32  — Output from diffusion denoising step
#   timestep:        [B] int64                — Current diffusion timestep
#   condition:       [B, N, D] float32        — Cross-attention conditioning (text + identity)
# Output:
#   verified_latent: [B, C, T, H, W] float32  — Passed verification
#   correction_needed: [B] bool               — True if any verification failed
#   failure_type:    [B] int64                — 0=none, 1=identity, 2=kinematics, 3=both
```

#### Identity Check (within Verification Daemon)
```python
# Input:
#   frame_latents:   [B, T, C, H, W] float32  — Latent frames to decode and check
#   vae_decoder:     nn.Module                — VAE decoder to get pixel frames
#   reference_embed: [B, 512] float32         — ArcFace reference embedding
# Output:
#   identity_scores: [B, T] float32           — Cosine similarity per frame
#   identity_pass:   [B, T] bool              — True if score >= 0.90
```

#### Kinematics Check (within Verification Daemon)
```python
# Input:
#   frame_latents:   [B, T, C, H, W] float32  — Latent frames to decode and check
#   vae_decoder:     nn.Module                — VAE decoder to get pixel frames
#   dwpose_model:    nn.Module                — DWPose for keypoint extraction
# Output:
#   physics_score:   float32                  — L_bone + L_ROM + L_velocity
#   kinematics_pass: bool                     — True if physics_score <= 0.01
```

#### Latent Rewind
```python
# Input:
#   current_latent:  [B, C, T, H, W] float32  — Current denoising latent (failed verification)
#   latent_history:  List[Tensor]             — [latent_t, latent_t-1, latent_t-2, ...]
#   rewind_steps:    int64                    — Number of steps to rewind (default 1)
# Output:
#   rewound_latent:  [B, C, T, H, W] float32  — Latent at t - rewind_steps
#   updated_history: List[Tensor]             — History truncated to rewind point
```

#### Localized Inpainting
```python
# Input:
#   latent_video:    [B, C, T, H, W] float32  — Current denoising latent
#   failure_mask:    [B, 1, T, H, W] float32  — Binary mask of failing regions
#   vae_encoder:     nn.Module                — VAE encoder for latent<->pixel conversion
#   failure_type:    int64                    — 1=identity, 2=kinematics (for constraint weighting)
# Output:
#   inpaint_latent:  [B, C, T, H, W] float32  — Latent with failing regions masked for regeneration
```

#### Constrained Regeneration
```python
# Input:
#   base_latent:     [B, C, T, H, W] float32  — Rewound latent state
#   inpaint_mask:    [B, 1, T, H, W] float32  — Regions to regenerate
#   text_condition:  [B, N, D] float32        — Prompt conditioning
#   identity_cond:   [B, 512] float32         — Identity conditioning vector
#   constraint_weights: Dict[str, float]      — Weights for L_identity, L_physics
#   increased_weight: float32                 — Multiplier for violated constraint (default 1.5)
# Output:
#   regenerated_latent: [B, C, T, H, W] float32 — Output of constrained denoising pass
```

### 4. Diffusion Model Integration Points

#### U-Net Forward Pass (Modified for Conditioning)
```python
# Input:
#   latent:          [B, C, T, H, W] float32  — Noisy latent video
#   timestep:        [B] int64                — Diffusion timestep
#   encoder_hidden_states: [B, N, D] float32  — Concatenated:
#                                                  - Text embeddings [77, 768 or 1024]
#                                                  - Identity vector [1, 512] (IP-Adapter)
#   added_cond_kwargs: Dict                   — For temporal unet variants:
#                                                  - time_ids: [B, 6] (fps, motion_bucket, etc.)
# Output:
#   noise_pred:      [B, C, T, H, W] float32  — Predicted noise
```

#### IP-Adapter Injection Points
```python
# Injection into U-Net cross-attention layers:
#   For each transformer block in U-Net:
#     - Key/Value projection: identity_vector [B, 512] -> [B, 1, C_kv]
#     - Query: from U-Net intermediate features [B*N, T*H*W, C_q]
#     - Attention: softmax(QK^T/sqrt(d)) * V
#     - Output projected back to U-Net feature dimension
```

### 5. Memory-Optimized Tensor Specifications for T4 GPU

#### Activation Checkpointing Strategy
```python
# Checkpoint these memory-intensive operations:
#   - VAE encode/decode (saves ~40% VRAM)
#   - IP-Adapter cross-attention layers (saves ~25% VRAM)
#   - DWPose pose estimation (if done on GPU, saves ~15% VRAM)
# Trade-off: ~20-30% increased compute time for ~50% VRAM reduction
```

#### Quantized Model Specifications
```python
# 8-bit quantization targets (bitsandbytes):
#   - U-Net: weights quantized to int8, activations fp16
#   - VAE: weights quantized to int8, activations fp16
#   - Text encoder: weights quantized to int8, activations fp16
#   - IP-Adapter: weights quantized to int8, activations fp16
#   - DWPose: weights quantized to int8, activations fp16
# Expected VRAM savings: ~50% vs fp16, ~25% vs fp32
```

## Open Questions
<!-- PM-OPEN --> Need to validate tensor shapes with actual SD video backbone (e.g., AnimateDiff, ModelScopeT2V)
<!-- PM-OPEN --> Determine optimal checkpointing strategy for T4 (16GB VRAM) vs A100 (24GB VRAM)
<!-- PM-OPEN --> Verify IP-Adapter token count compatibility with SD XL vs SD 1.5 text encoders