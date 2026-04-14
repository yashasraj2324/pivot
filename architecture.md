# P.I.V.O.T. Phase 1 Architecture

## Overview

Phase 1 focuses on Core Pipeline Validation: establishing the single-character generation pipeline with identity locking mechanism (ArcFace + IP-Adapter) and kinematic constraint verification (L_physics). The architecture implements a closed-loop verification architecture (Verification Daemon) that treats identity and biomechanical laws as hard gates between generation and output. Foundational work on emotion/FACS enforcement is noted but not implemented in Phase 1.

## System Components

### 1. Identity Router (Layer 2)
Ensures visual consistency with reference photograph throughout video sequence.

#### Subcomponents:
- **Reference Extraction**: Uses InsightFace (buffalo_l) to extract 512-dimensional ArcFace embedding
- **Identity Conditioning**: IP-Adapter Plus injection into U-Net cross-attention layers
- **Localized Masking**: SAM or DWPose-based segmentation to confine conditioning to character silhouette
- **Cosine Similarity Gate**: Per-frame evaluation with threshold ≥ 0.90

### 2. Kinematic Guardrail (Layer 1)
Ensures generated motion complies with human biomechanics.

#### Subcomponents:
- **Bone Length Invariance**: Tracks Euclidean distance between connected joint pairs
- **Biomechanical Range of Motion**: Hard clamps joint angles to physiological limits
- **Temporal Velocity Limits**: Bounds first derivative of joint positions
- **Rigid Body Topology**: Preserves geometry of skull, ribcage, and pelvis

### 3. Verification Daemon (V1)
Enforcement layer operating as hard gate between generation and output.

#### Subcomponents:
- **Interceptor Gate**: Positioned between diffusion denoising loop and output buffer
- **Identity Check**: Cosine similarity against ArcFace reference embedding
- **Kinematics Check**: Evaluates L_physics composite metric
- **Latent Rewind**: Reverts denoising process to t-1 on violation
- **Localized Inpainting**: Pixel-precise masking over failing region
- **Constrained Regeneration**: Regeneration with increased weight on violated constraint
- **Max Retry Depth**: Maximum 5 attempts before selecting highest-scoring candidate

## Data Flow

```
Reference Image → InsightFace → Identity Vector → IP-Adapter Conditioning
Text Prompt → Prompt Encoder → Text Conditioning
Identity + Text Conditioning → Diffusion Model (U-Net)
Diffusion Output → Verification Daemon (Identity + Kinematics Checks)
Verification Result → [PASS: Advance to Output Buffer] [FAIL: Correction Loop]
Correction Loop → Latent Rewind → Localized Inpainting → Constrained Regeneration → Re-verification
```

## Technical Specifications

### Core Model
- Video diffusion backbone (SD-based architecture)
- PyTorch ≥ 2.0, CUDA ≥ 11.8
- 512×512 resolution, 16-60 frames per sequence

### Identity Stack
- Face Detection: RetinaFace (buffalo_l)
- Embedding Extraction: ArcFace (insightface) → 512-d vector
- Identity Conditioning: IP-Adapter Plus
- Segmentation: SAM or DWPose

### Kinematic Stack
- Pose Estimation: DWPose (ControlNet) → 17-keypoint skeleton
- Bone Tracking: Custom implementation
- ROM Enforcement: Configurable joint limits
- Velocity Tracking: Frame delta computation

### Loss Functions
- L_identity: 1 - cos(embedding_ref, embedding_gen) ≤ 0.10
- L_bone: Σ_t || d(j_a, j_b)_t − d(j_a, j_b)_{t-1} ||^2 ≤ tolerance
- L_ROM: max(0, angle - limit) per joint = 0
- L_velocity: max(0, ||v|| - v_max) = 0
- L_physics: L_bone + L_ROM + L_velocity ≤ 0.01

### Memory Optimization
- 8-bit Quantization (bitsandbytes)
- Attention Slicing (PyTorch SDPA)
- Gradient Checkpointing (torch.checkpoint)
- CPU Offloading (Accelerated)

## Interfaces (Phase 1 Focus)
- Experimentation notebooks in `/notebooks`
- Core implementation in `/core` and `/adapters`
- Evaluation utilities in `/scripts`

*Future phases will add SDK, REST API, and ComfyUI node interfaces.*