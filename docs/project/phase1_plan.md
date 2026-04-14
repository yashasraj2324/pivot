# PIVOT Phase 1 Project Plan
## Core Pipeline Validation (3 Days, 4 Developers)

## Overview
This plan outlines the execution of Phase 1: Core Pipeline Validation for the PIVOT project. The goal is to establish a functional single-character generation pipeline with Identity Router and Kinematic Guardrail enforcement, including the Verification Daemon for automated correction.

**Timeline:** 3 days (April 15-17, 2026)
**Team:** 4 developers
**Total Capacity:** 12 developer-days

## Milestones & Success Criteria

| Milestone | Target Completion | Success Criteria |
|-----------|-------------------|------------------|
| **M1: Identity Router Functional** | EOD Day 1 | Reference photo → ArcFace embedding → IP-Adapter conditioning pipeline working; cosine similarity gate functional |
| **M2: Kinematic Guardrail Functional** | EOD Day 2 | L_physics loss function implemented; bone length invariance, ROM limits, and velocity limits enforced |
| **M3: Verification Daemon Operational** | EOD Day 2 | Detect → Rewind → Inpaint → Verify loop operational with max 5 retries |
| **M4: Unit Tests Passing** | EOD Day 3 | All physics constraint unit tests passing (≥80% coverage) |
| **M5: End-to-End Pipeline Validated** | EOD Day 3 | Single-character generation with stable identity and kinematics; notebook experiments committed |
| **M6: Documentation Complete** | EOD Day 3 | All experiments documented in /notebooks; code commented and ready for review |

## Resource Allocation

| Developer | Primary Focus | Secondary Support | Estimated Effort |
|-----------|---------------|-------------------|------------------|
| **Dev 1** | Identity Router (IR) | Verification Daemon (VD) | 3.5 days IR, 1.5 days VD |
| **Dev 2** | Kinematic Guardrail (KG) | Verification Daemon (VD) | 3.5 days KG, 1.5 days VD |
| **Dev 3** | Verification Daemon (VD) | Testing & Integration | 2 days VD, 1 day testing |
| **Dev 4** | Integration & Testing | Documentation | 2 days integration, 1 day testing |

## Detailed Task Breakdown

### Day 1: Foundation Components
**Focus:** Establish core identity and kinematic tracking systems

#### Identity Router Tasks (Dev 1 Lead)
- **T1.1:** Implement ArcFace embedding extraction using InsightFace buffalo_l model
  - *Output:* Function that takes reference image, returns 512-d embedding
  - *Dependency:* None
  - *Estimate:* 0.5 days

- **T1.2:** Implement IP-Adapter conditioning injection into U-Net cross-attention layers
  - *Output:* Modified diffusion model that accepts identity vector conditioning
  - *Dependency:* T1.1
  - *Estimate:* 1.0 days

- **T1.3:** Implement localized masking using SAM/DWPose for instance-level segmentation
  - *Output:* Pixel-precise mask confining identity conditioning to character silhouette
  - *Dependency:* None (can work in parallel)
  - *Estimate:* 0.5 days

- **T1.4:** Implement cosine similarity gate with threshold 0.90
  - *Output:* Verification component that halts pipeline on similarity < 0.90
  - *Dependency:* T1.1, T1.2
  - *Estimate:* 0.5 days

- **T1.5:** Unit tests for Identity Router components
  - *Output:* Test suite validating embedding extraction, conditioning, masking, and gating
  - *Dependency:* T1.1-T1.4
  - *Estimate:* 0.5 days

#### Kinematic Guardrail Tasks (Dev 2 Lead)
- **T2.1:** Implement DWPose/ControlNet pose estimation for 17-keypoint skeleton
  - *Output:* Function that extracts pose keypoints from frames
  - *Dependency:* None
  - *Estimate:* 0.5 days

- **T2.2:** Implement bone length invariance loss (L_bone)
  - *Output:* Loss function that tracks joint distances across frames
  - *Dependency:* T2.1
  - *Estimate:* 0.5 days

- **T2.3:** Implement biomechanical range of motion limits (joint-specific clamps)
  - *Output:* Constraint enforcement for shoulder, elbow, hip, knee, ankle limits
  - *Dependency:* T2.1
  - *Estimate:* 0.5 days

- **T2.4:** Implement temporal velocity limits (v_max = 2.0 units/frame)
  - *Output:* Velocity calculation and clamping per joint per frame
  - *Dependency:* T2.1
  - *Estimate:* 0.5 days

- **T2.5:** Implement rigid body topology preservation (SSIM on head/torso/pelvis)
  - *Output:* Structural similarity check for rigid regions
  - *Dependency:* T2.1
  - *Estimate:* 0.5 days

- **T2.6:** Composite L_physics loss function (L_bone + L_ROM + L_velocity)
  - *Output:* Combined physics loss with configurable tolerance (0.01)
  - *Dependency:* T2.2-T2.5
  - *Estimate:* 0.5 days

- **T2.7:** Unit tests for Kinematic Guardrail components
  - *Output:* Test suite validating all kinematic constraints
  - *Dependency:* T2.1-T2.6
  - *Estimate:* 0.5 days

### Day 2: Verification Daemon & Integration
**Focus:** Build the enforcement layer and begin integration

#### Verification Daemon Tasks (Dev 3 Lead)
- **T3.1:** Implement interceptor gate between denoising loop and output buffer
  - *Output:* Hard gate that requires ALL checks to pass before frame advancement
  - *Dependency:* None (structural)
  - *Estimate:* 0.5 days

- **T3.2:** Integrate identity check (VD-REQ-002) with cosine similarity gate
  - *Output:* Frame-by-frame identity verification triggering halt on failure
  - *Dependency:* T1.4, T3.1
  - *Estimate:* 0.5 days

- **T3.3:** Integrate kinematics check (VD-REQ-003) with L_physics evaluation
  - *Output:* Frame-by-frame physics verification triggering halt on L_physics > tolerance
  - *Dependency:* T2.6, T3.1
  - *Estimate:* 0.5 days

- **T3.4:** Implement latent rewind mechanism (rewind to t-1 on violation)
  - *Output:* State preservation and rewinding capability
  - *Dependency:* T3.1-T3.3
  - *Estimate:* 0.5 days

- **T3.5:** Implement localized inpainting with segmentation mask generation
  - *Output:* Pixel-precise mask over failing region for targeted regeneration
  - *Dependency:* T1.3, T2.1 (for joint/region identification)
  - *Estimate:* 0.5 days

- **T3.6:** Implement constrained regeneration with increased weight on violated constraint
  - *Output:* Regeneration pass with 1.5x weight on failed constraint type
  - *Dependency:* T3.4, T3.5
  - *Estimate:* 0.5 days

- **T3.7:** Implement max retry depth (5 attempts) with fallback to highest-scoring candidate
  - *Output:* Graceful degradation mechanism with logging
  - *Dependency:* T3.4-T3.6
  - *Estimate:* 0.5 days

#### Integration Tasks (Dev 1 & Dev 2 Support)
- **T4.1:** Connect Identity Router to Verification Daemon
  - *Output:* Identity vector flows from reference → IP-Adapter → VD check
  - *Dependency:* T1.4, T3.2
  - *Estimate:* 0.5 days (Dev 1)

- **T4.2:** Connect Kinematic Guardrail to Verification Daemon
  - *Output:* Pose data flows from DWPose → L_physics → VD check
  - *Dependency:* T2.6, T3.3
  - *Estimate:* 0.5 days (Dev 2)

- **T4.3:** End-to-end pipeline integration test (no corrections yet)
  - *Output:* Pipeline runs from reference + prompt to unverified output
  - *Dependency:* T4.1, T4.2
  - *Estimate:* 0.5 days (Dev 1 & Dev 2)

### Day 3: Validation, Testing & Documentation
**Focus:** Verify functionality, write tests, document experiments

#### Testing & Validation Tasks (Dev 3 & Dev 4 Lead)
- **T5.1:** Develop unit test suite for Verification Daemon
  - *Output:* Tests for rewind, inpainting, regeneration, retry logic
  - *Dependency:* T3.1-T3.7
  - *Estimate:* 0.5 days (Dev 3)

- **T5.2:** Integration test: Identity violation → correction loop
  - *Output:* System detects low similarity, rewinds, inpaints, verifies
  - *Dependency:* All T1.x, T3.x
  - *Estimate:* 0.5 days (Dev 4)

- **T5.3:** Integration test: Kinematic violation → correction loop
  - *Output:* System detects L_physics violation, rewinds, inpaints, verifies
  - *Dependency:* All T2.x, T3.x
  - *Estimate:* 0.5 days (Dev 4)

- **T5.4:** End-to-end validation: Generate 60-frame sequence with known good prompt
  - *Output:* Verified output buffer filled with identity-stable, kinematically valid frames
  - *Dependency:* T5.2, T5.3
  - *Estimate:* 0.5 days (Dev 3 & Dev 4)

- **T5.5:** Performance benchmarking and VRAM optimization validation
  - *Output:* Measurements showing pipeline works within T4/A100 constraints
  - *Dependency:* T5.4
  - *Estimate:* 0.5 days (Dev 4)

#### Documentation & Knowledge Transfer (Dev 4 Lead)
- **T6.1:** Create and commit experimentation notebooks to /notebooks
  - *Output:* 3-5 Colab-ready notebooks demonstrating core pipeline functionality
  - *Dependency:* T5.4
  - *Estimate:* 0.5 days (Dev 4)

- **T6.2:** Document API and usage guidelines for core components
  - *Output:* README updates and inline code documentation
  - *Dependency:* All completed tasks
  - *Estimate:* 0.5 days (Dev 4)

- **T6.3:** Code review preparation and final integration check
  - *Output:* Ensure all components work together and are ready for next phase
  - *Dependency:* All tasks
  - *Estimate:* 0.5 days (Dev 4)

## Dependencies Summary

### Critical Path Dependencies
```
T1.1 → T1.2 → T1.4 → T3.2 → T3.1 → T4.1
T2.1 → T2.2 → T2.3 → T2.4 → T2.5 → T2.6 → T3.3 → T3.1 → T4.2
T3.1 → T3.2 → T3.3 → T3.4 → T3.5 → T3.6 → T3.7
T4.1 + T4.2 → T4.3 → T5.2 + T5.3 → T5.4 → T5.5 → T6.1
```

### Parallel Work Streams
- **Days 1-2:** Identity Router (Dev 1) and Kinematic Guardrail (Dev 2) can work in parallel
- **Day 2:** Verification Daemon (Dev 3) builds while Dev 1 & Dev 2 integrate their components
- **Day 3:** Testing (Dev 3 & Dev 4) and Documentation (Dev 4) can proceed in parallel after integration

## Risk Mitigation

| Risk | Probability | Impact | Mitigation Strategy |
|------|-------------|--------|---------------------|
| **VRAM exhaustion on T4** | High | Pipeline failure | Implement 8-bit quantization and attention slicing; validate on A100 if needed |
| **Pose estimation inaccuracies** | Medium | False positives | Tune confidence thresholds; use temporal smoothing |
| **Latent rewind instability** | Medium | Quality degradation | Implement strict state preservation checks |
| **Integration complexity** | Medium | Delayed milestones | Daily integration checks; maintain working branches |
| **Insufficient time for testing** | Low | Undetected bugs | Prioritize core functionality; use time-boxed exploration |

## Daily Standup Focus

**Day 1 Morning:**
- Confirm environment setup and dependency installation
- Review task assignments and API contracts between components

**Day 1 Evening:**
- Demo: ArcFace embedding extraction working
- Demo: IP-Adapter conditioning injection verified

**Day 2 Morning:**
- Review: Identity Router integration status
- Review: Kinematic Guardrail component completion

**Day 2 Evening:**
- Demo: Pose estimation and L_physics loss working
- Demo: Verification Daemon interceptor gate functional

**Day 3 Morning:**
- Review: Verification Daemon correction loop status
- Review: Integration test readiness

**Day 3 Evening:**
- Demo: End-to-end pipeline generating verified output
- Review: Notebooks committed and documentation complete

## Definition of Done for Phase 1
Phase 1 is complete when:
1. ✅ Single-character generation pipeline produces outputs with stable identity (cosine similarity ≥ 0.90)
2. ✅ All kinematic constraints (bone length, ROM, velocity) are enforced per frame
3. ✅ Verification Daemon autonomously detects, corrects, and verifies violations
4. ✅ Unit test suite passes for all core components (≥80% coverage)
5. ✅ Experimentation notebooks are committed to /notebooks demonstrating core functionality
6. ✅ No manual intervention required for correction loop operation

## Next Steps (Phase 2 Preparation)
Upon completion, the team will be ready for Phase 2: Emotion & Body Language Stack, which will build upon this validated core pipeline.