# ADR-002: Kinematic Guardrail Implementation Details
**Date**: 2026-04-14 | **Status**: Proposed
**Author**: ai-architect agent

## Context
The Kinematic Guardrail (Layer 1) ensures generated motion complies with human biomechanics. Based on Phase 1 PRD requirements KG-REQ-001 through KG-REQ-004 and architecture.md specifications.

## Options Considered

### Option A — DWPose + Custom Bone Tracking + Configurable ROM + Velocity Bounds
**Pros**:
- DWPose provides accurate 17-keypoint pose estimation
- Custom bone tracking allows precise L_bone calculation
- Configurable joint limits support different populations
- Frame delta computation for velocity bounds is computationally efficient
**Cons**:
- Requires custom implementation of bone tracking logic
- DWPose adds computational overhead
**Estimated compute**: ~0.4GB VRAM for DWPose + minimal CPU for tracking

### Option B — MediaPipe Pose + Predefined Joint Limits + Simple Velocity Check
**Pros**:
- MediaPipe is lightweight and fast
- Predefined joint limits reduce implementation complexity
**Cons**:
- Less accurate than DWPose for complex poses
- Limited to 33 landmarks vs DWPose's more detailed skeleton
- May miss subtle biomechanical violations

### Option C — ViTPose + Learned ROM Constraints + Optical Flow Velocity
**Pros**:
- ViTPose provides state-of-the-art pose estimation
- Learned constraints could adapt to specific motions
**Cons**:
- Significantly higher computational cost
- Learned constraints may not generalize well
- Optical flow adds complexity and potential drift

## Decision
Selected Option A: DWPose (ControlNet) for pose estimation + custom bone length tracking + configurable joint limits + frame delta velocity computation.

## Consequences
- Pose estimation using DWPose to extract 17-keypoint skeleton per frame
- Bone length invariance calculated as sum of squared differences between consecutive frames
- Joint angles clamped to physiological limits (shoulder 0-180°, elbow 0-145°, etc.)
- Velocity bounds computed as ||(p_t - p_{t-1})/Δt|| ≤ v_max (default 2.0 units/frame)
- Rigid body topology enforced via geometric constraints on skull, ribcage, pelvis regions
- L_physics composite metric combines all constraints with threshold ≤ 0.01
- Memory efficient for T4 GPU deployment

## Open Questions
<!-- PM-OPEN --> Need to validate joint angle calculation method from 2D keypoints; may require 3D lifting or confidence-based weighting