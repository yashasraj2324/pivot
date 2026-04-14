# ClickUp Tasks for PIVOT Phase 1

Based on the project plan, here are the tasks to be created in ClickUp.

## Milestones
- M1: Identity Router Functional (EOD Day 1)
- M2: Kinematic Guardrail Functional (EOD Day 2)
- M3: Verification Daemon Operational (EOD Day 2)
- M4: Unit Tests Passing (EOD Day 3)
- M5: End-to-End Pipeline Validated (EOD Day 3)
- M6: Documentation Complete (EOD Day 3)

## Tasks

### Day 1: Foundation Components
**Identity Router (Dev 1 Lead)**
- T1.1: Implement ArcFace embedding extraction using InsightFace buffalo_l model
- T1.2: Implement IP-Adapter conditioning injection into U-Net cross-attention layers
- T1.3: Implement localized masking using SAM/DWPose for instance-level segmentation
- T1.4: Implement cosine similarity gate with threshold 0.90
- T1.5: Unit tests for Identity Router components

**Kinematic Guardrail (Dev 2 Lead)**
- T2.1: Implement DWPose/ControlNet pose estimation for 17-keypoint skeleton
- T2.2: Implement bone length invariance loss (L_bone)
- T2.3: Implement biomechanical range of motion limits (joint-specific clamps)
- T2.4: Implement temporal velocity limits (v_max = 2.0 units/frame)
- T2.5: Implement rigid body topology preservation (SSIM on head/torso/pelvis)
- T2.6: Composite L_physics loss function (L_bone + L_ROM + L_velocity)
- T2.7: Unit tests for Kinematic Guardrail components

### Day 2: Verification Daemon & Integration
**Verification Daemon (Dev 3 Lead)**
- T3.1: Implement interceptor gate between denoising loop and output buffer
- T3.2: Integrate identity check with cosine similarity gate
- T3.3: Integrate kinematics check with L_physics evaluation
- T3.4: Implement latent rewind mechanism (rewind to t-1 on violation)
- T3.5: Implement localized inpainting with segmentation mask generation
- T3.6: Implement constrained regeneration with increased weight on violated constraint
- T3.7: Implement max retry depth (5 attempts) with fallback to highest-scoring candidate

**Integration (Dev 1 & Dev 2 Support)**
- T4.1: Connect Identity Router to Verification Daemon
- T4.2: Connect Kinematic Guardrail to Verification Daemon
- T4.3: End-to-end pipeline integration test (no corrections yet)

### Day 3: Validation, Testing & Documentation
**Testing & Validation (Dev 3 & Dev 4 Lead)**
- T5.1: Develop unit test suite for Verification Daemon
- T5.2: Integration test: Identity violation → correction loop
- T5.3: Integration test: Kinematic violation → correction loop
- T5.4: End-to-end validation: Generate 60-frame sequence with known good prompt
- T5.5: Performance benchmarking and VRAM optimization validation

**Documentation & Knowledge Transfer (Dev 4 Lead)**
- T6.1: Create and commit experimentation notebooks to /notebooks
- T6.2: Document API and usage guidelines for core components
- T6.3: Code review preparation and final integration check