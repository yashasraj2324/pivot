# ADR-003: Verification Daemon Workflow and Decision Logic
**Date**: 2026-04-14 | **Status**: Proposed
**Author**: ai-architect agent

## Context
The Verification Daemon (V1) operates as a hard gate between generation and output, implementing detection, correction, and verification loops. Based on Phase 1 PRD requirements VD-REQ-001 through VD-REQ-007 and architecture.md specifications.

## Options Considered

### Option A — Sequential Checking with Immediate Halt and Correction Loop
**Pros**:
- Simple to implement and reason about
- Immediate response to violations minimizes wasted computation
- Clear failure handling with defined correction steps
- Matches the hard gate semantics described in PRD
**Cons**:
- Sequential checking may miss simultaneous violations
- Correction loop adds latency to pipeline
**Estimated compute**: Minimal overhead (~0.1GB VRAM for state buffers)

### Option B — Parallel Checking with Batch Correction
**Pros**:
- Better utilization of compute resources
- Can detect multiple violation types simultaneously
**Cons**:
- More complex state management
- Difficult to attribute specific failures to correction actions
- May violate hard gate semantics if not carefully implemented

### Option C — Predictive Checking with Pre-emption
**Pros**:
- Could prevent violations before they occur
- Potentially reduces correction loop frequency
**Cons**:
- Significantly more complex to implement
- Prediction accuracy uncertain
- May interfere with generative process

## Decision
Selected Option A: Sequential checking (Identity then Kinematics) with immediate halt on failure and structured correction loop (Latent Rewind → Localized Inpainting → Constrained Regeneration → Re-verification).

## Consequences
- Verification Daemon positioned between diffusion denoising loop and output buffer
- ALL checks must pass (Identity AND Kinematics); ONE failure triggers halt
- Identity Check: Cosine similarity against ArcFace reference (threshold ≥ 0.90)
- Kinematics Check: L_physics composite metric (threshold ≤ 0.01)
- On failure: Latent rewind to t-1 preserves last valid latent state
- Localized Inpainting: SAM/DWPose-based mask over failing region only
- Constrained Regeneration: Increased weight (1.5x) on violated constraint
- Max retry depth: 5 attempts before selecting highest-scoring candidate
- Designed for T4 GPU constraints with minimal state preservation overhead

## Open Questions
<!-- PM-OPEN --> Need to determine optimal rewind step size (t-1 vs t-2) based on empirical testing; may affect correction quality vs latency tradeoff