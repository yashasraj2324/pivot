# ADR-001: Identity Router Implementation Details
**Date**: 2026-04-14 | **Status**: Proposed
**Author**: ai-architect agent

## Context
The Identity Router (Layer 2) ensures visual consistency with reference photograph throughout video sequence. Based on Phase 1 PRD requirements IR-REQ-001 through IR-REQ-004 and architecture.md specifications.

## Options Considered

### Option A — InsightFace + IP-Adapter Plus + SAM Segmentation
**Pros**:
- Proven ArcFace embeddings for identity preservation
- IP-Adapter Plus provides strong conditioning without fine-tuning
- SAM provides precise segmentation for localized masking
- All components have existing implementations and community support
**Cons**:
- SAM can be computationally expensive
- Requires careful integration of multiple models
**Estimated compute**: ~2.1GB VRAM for models + ~0.8GB for activations

### Option B — Face++ + Custom Adapter + Simple Threshold Mask
**Pros**:
- Simpler implementation
- Lower computational overhead
**Cons**:
- Less accurate identity preservation
- Proprietary Face++ API (not suitable for open source)
- Poor localization of identity conditioning

### Option C — DINOv2 + CLIP-based Adapter + DWPose Segmentation
**Pros**:
- DINOv2 provides rich visual features
- DWPose is lightweight for pose-based segmentation
**Cons**:
- Less established for identity preservation
- May not provide sufficient identity specificity
- CLIP-based adapters less effective than IP-Adapter for identity

## Decision
Selected Option A: InsightFace (buffalo_l) + IP-Adapter Plus + SAM segmentation with cosine similarity gate.

## Consequences
- Identity preservation maintained through 512-d ArcFace embeddings
- Conditioning injected into U-Net cross-attention layers via IP-Adapter Plus
- Localized masking prevents identity bleed into background
- Per-frame cosine similarity threshold of 0.90 triggers verification daemon
- Memory optimizations required for T4 GPU constraints

## Open Questions
<!-- PM-OPEN --> Need to verify SAM model size for T4 GPU constraints; may need to substitute with lighter segmentation if VRAM exceeded