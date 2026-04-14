# P.I.V.O.T. Phase 1 — Core Pipeline Validation
## Product Requirements & Technical Specifications

---

## 1. Executive Summary

**Phase 1** establishes the foundational infrastructure for P.I.V.O.T.'s Core Pipeline Validation, focusing exclusively on **single-character generation** with Identity Router and Kinematic Guardrail enforcement. This phase validates that the Shapeshifter Effect can be eliminated for human subjects through a closed-loop verification architecture that enforces biomechanical constraints and identity permanence within the video diffusion process.

The phase delivers a fully functional proof-of-concept that demonstrates:
- Stable identity preservation across all frames using ArcFace embeddings and IP-Adapter conditioning
- Strict biomechanical constraint enforcement through the L_physics loss function
- Automated detection, correction, and verification without human intervention

---

## 2. Problem Statement

### 2.1 The Shapeshifter Effect in Single Characters

Current video diffusion models suffer from systematic identity and structural degradation in single-character outputs:

| Failure Mode | Description | Impact |
|--------------|-------------|--------|
| **Identity Drift** | Face gradually morphs across frames until the person at frame 1 is unrecognizable by frame 60 | Character becomes unidentifiable |
| **Topological Morphing** | Skull, joints, and torso change shape mid-sequence; arms stretch; hands grow extra fingers | Anatomically impossible outputs |
| **Kinematic Violations** | Joints hyperextend beyond physiological limits; characters teleport between positions | Physically impossible motion |
| **Structural Distortion** | Ribcage and pelvis geometry warps unpredictably | Human form becomes unrecognizable |

### 2.2 Why Phase 1 Exists

Phase 1 exists to prove that the Shapeshifter Effect can be solved before adding the complexity of multi-character support, environmental physics, and emotion enforcement. The core insight driving this phase:

> These are not generation quality problems — they are enforcement architecture problems. The solution requires a verification and correction layer that treats biomechanical and identity laws as hard constraints.

---

## 3. Product Goals (Phase 1)

| Goal ID | Goal Description | Success Metric |
|---------|------------------|----------------|
| **G1** | Single-character video generation from one reference photo and a text prompt | Video output with consistent identity |
| **G2** | Strict kinematic enforcement on every output frame | 100% of frames pass L_physics constraints |
| **G3** | Identity permanence throughout entire sequence | Cosine similarity ≥ 0.90 on all frames |
| **G4** | Automated correction loop (detect → rewind → inpaint → verify) | Zero manual intervention required |
| **G5** | Reproducible notebook-based development workflow | All experiments committed to /notebooks |

---

## 4. Product Requirements

### 4.1 Identity Router (Layer 2)

The Identity Router ensures the generated character maintains visual consistency with the reference photograph throughout the entire video sequence.

#### 4.1.1 Reference Extraction
- **Requirement ID:** IR-REQ-001
- **Description:** Extract a stable 512-dimensional facial embedding from the reference photograph using InsightFace (ArcFace)
- **Technical Specification:**
  - Model: `buffalo_l` (InsightFace variant)
  - Output: 512-dimensional embedding vector
  - Preprocessing: Face detection, alignment, and normalization
- **Validation:** Embedding extracted successfully; cosine similarity to self ≥ 0.99

#### 4.1.2 Identity Conditioning
- **Requirement ID:** IR-REQ-002
- **Description:** Inject the identity vector into the diffusion model's cross-attention layers via IP-Adapter
- **Technical Specification:**
  - IP-Adapter variant: `ip-adapter_plus` (for SD-based video models)
  - Injection point: U-Net cross-attention layers
  - Conditioning weight: Configurable, default 0.7
- **Validation:** Generated frames maintain facial features consistent with reference

#### 4.1.3 Localized Masking
- **Requirement ID:** IR-REQ-003
- **Description:** Apply instance-level segmentation masking to confine identity conditioning strictly to the character's silhouette
- **Technical Specification:**
  - Segmentation model: SAM (Segment Anything Model) or DWPose-based body segmentation
  - Mask application: Pixel-precise boundary enforcement
  - Background exclusion: Identity vector does not affect background regions
- **Validation:** No identity bleed into background elements

#### 4.1.4 Cosine Similarity Gate
- **Requirement ID:** IR-REQ-004
- **Description:** Evaluate every frame against reference embedding; halt if cosine similarity drops below threshold
- **Technical Specification:**
  - Threshold: 0.90 (configurable)
  - Evaluation: Per-frame, not batch-averaged
  - Trigger: Immediate pipeline halt on violation
- **Validation:** System halts and triggers correction when similarity < 0.90

---

### 4.2 Kinematic Guardrail (Layer 1)

The Kinematic Guardrail ensures all generated motion complies with human biomechanics.

#### 4.2.1 Bone Length Invariance
- **Requirement ID:** KG-REQ-001
- **Description:** Euclidean distance between all connected joint pairs remains constant across time, normalized for camera depth
- **Technical Specification:**
  - Formula: `L_bone = Σ_t || d(j_a, j_b)_t − d(j_a, j_b)_{t-1} ||^2`
  - Expected value: 0 (within tolerance)
  - Tolerance: ≤ 0.5% deviation per joint pair
  - Tracked joints: All skeletal connections (neck, shoulders, elbows, wrists, hips, knees, ankles, spine)
- **Validation:** No bone length deviation detected in output frames

#### 4.2.2 Biomechanical Range of Motion
- **Requirement ID:** KG-REQ-002
- **Description:** All joint angles are mathematically clamped to human physiological limits
- **Technical Specification:**
  - Joint limits (degrees):
    - Shoulder (flexion/extension): 0–180°
    - Shoulder (abduction): 0–180°
    - Elbow (flexion): 0–145°
    - Hip (flexion): 0–125°
    - Knee (flexion): 0–135°
    - Ankle (dorsiflexion): 0–20°
    - Ankle (plantarflexion): 0–50°
  - Implementation: Hard clamp in kinematic evaluation
  - Violation detection: Any joint exceeding limits triggers correction
- **Validation:** All joints remain within physiological bounds

#### 4.2.3 Temporal Velocity Limits
- **Requirement ID:** KG-REQ-003
- **Description:** First derivative of all joint positions is bounded; teleportation is architecturally impossible
- **Technical Specification:**
  - Formula: `|| (p_t − p_{t-1}) / Δt || ≤ v_max`
  - Maximum velocity: Configurable, default 2.0 units/frame (calibrated to human maximums)
  - Evaluation: Per-joint, per-frame
  - Teleportation prevention: Velocity > v_max triggers immediate halt
- **Validation:** No joint velocity exceeds v_max

#### 4.2.4 Rigid Body Topology
- **Requirement ID:** KG-REQ-004
- **Description:** Skull, ribcage, and pelvis maintain fixed geometry with no stretching or morphing
- **Technical Specification:**
  - Rigid regions: Head, torso (ribcage), pelvis
  - Topology preservation: Geometric constraints on rigid body segments
  - Detection: Structural similarity index (SSIM) on rigid region bounding boxes
- **Validation:** Rigid regions maintain constant shape across frames

---

### 4.3 Verification Daemon (V1)

The Verification Daemon is the enforcement layer that sits between generation and output, operating as a hard gate.

#### 4.3.1 Interceptor Gate
- **Requirement ID:** VD-REQ-001
- **Description:** Operates as a hard gate before writing to the output buffer; no frame passes without verification
- **Technical Specification:**
  - Position: Between diffusion denoising loop and output buffer
  - Gate logic: ALL checks must pass; ONE failure = halt
  - Async operation: Verification runs parallel to generation for minimal latency
- **Validation:** No unverified frames reach output

#### 4.3.2 Identity Check
- **Requirement ID:** VD-REQ-002
- **Description:** Evaluate cosine similarity against ArcFace reference embedding
- **Technical Specification:**
  - Threshold: 0.90
  - Evaluation frequency: Every frame
  - Trigger action: Halt on failure
- **Validation:** Similarity ≥ 0.90 on all output frames

#### 4.3.3 Kinematics Check
- **Requirement ID:** VD-REQ-003
- **Description:** Evaluate L_physics composite metric across all tracked bounds
- **Technical Specification:**
  - Components: L_bone + ROM penalties + velocity bounds
  - Tolerance: Configurable, default 0.01
  - Evaluation frequency: Every frame
  - Trigger action: Halt on failure
- **Validation:** L_physics ≤ tolerance on all output frames

#### 4.3.4 Latent Rewind
- **Requirement ID:** VD-REQ-004
- **Description:** On violation, rewind the denoising process to t-1, preserving the last fully valid latent state
- **Technical Specification:**
  - Rewind target: t-1 (previous denoising step)
  - State preservation: Last valid latent tensor
  - Trigger: Any verification failure
- **Validation:** Rewind executes without data loss

#### 4.3.5 Localized Inpainting
- **Requirement ID:** VD-REQ-005
- **Description:** Generate a pixel-precise segmentation mask over the failing region; regenerate only that region
- **Technical Specification:**
  - Mask generation: SAM-based or DWPose skeletal segmentation
  - Mask precision: Pixel-level boundary
  - Inpainting scope: Failing region only (character, joint, facial region)
  - Regeneration: Constrained to violated constraint type
- **Validation:** Inpainting addresses specific failure region

#### 4.3.6 Constrained Regeneration
- **Requirement ID:** VD-REQ-006
- **Description:** Regeneration pass is conditioned on all active constraints with increased weight on the violated constraint
- **Technical Specification:**
  - Constraint weighting: Violated constraint weight multiplied by 1.5x (configurable)
  - Regeneration steps: Full denoising schedule
  - Re-verification: Automatic after regeneration
- **Validation:** Regenerated frame passes verification

#### 4.3.7 Max Retry Depth
- **Requirement ID:** VD-REQ-007
- **Description:** Maximum 5 retry attempts per frame before selecting highest-scoring candidate
- **Technical Specification:**
  - Max retries: 5 (configurable)
  - Selection criteria: Highest composite score (identity + kinematics)
  - Fallback: Pass highest-scoring candidate with flag for review
  - Logging: All attempts logged with scores
- **Validation:** System handles max retries gracefully

---

## 5. Technical Specifications

### 5.1 Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        P.I.V.O.T. Phase 1 Pipeline                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                  │
│  │   Reference │────▶│  InsightFace │────▶│   Identity  │                  │
│  │    Image    │     │  (ArcFace)   │     │   Vector    │                  │
│  └─────────────┘     └─────────────┘     └──────┬──────┘                  │
│                                                 │                          │
│                                                 ▼                          │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                  │
│  │    Text     │────▶│   Prompt    │────▶│   IP-Adapter│                  │
│  │   Prompt    │     │   Encoder   │     │  Conditioning│                  │
│  └─────────────┘     └─────────────┘     └──────┬──────┘                  │
│                                                 │                          │
│                                                 ▼                          │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │                    Diffusion Model (U-Net)                      │       │
│  │  ┌───────────────────┐     ┌───────────────────────────────┐   │       │
│  │  │  Cross-Attention  │◀───▶│  IP-Adapter Identity Vector   │   │       │
│  │  │    (Conditioned)  │     │  + Text Prompt Conditioning     │   │       │
│  │  └───────────────────┘     └───────────────────────────────┘   │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                                 │                          │
│                                                 ▼                          │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │                    Verification Daemon                          │       │
│  │  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐   │       │
│  │  │  DWPose     │  │  ArcFace     │  │   L_physics         │   │       │
│  │  │  Skeleton   │  │  Similarity  │  │   (Bone+ROM+Vel)    │   │       │
│  │  └──────┬───────┘  └──────┬───────┘  └──────────┬──────────┘   │       │
│  │         │                │                     │              │       │
│  │         └────────────────┴─────────────────────┘              │       │
│  │                          │                                     │       │
│  │                          ▼                                     │       │
│  │                 ┌────────────────┐                             │       │
│  │                 │  PASS/FAIL     │                             │       │
│  │                 │   Decision     │                             │       │
│  │                 └───────┬────────┘                             │       │
│  │                         │                                      │       │
│  │            ┌────────────┴────────────┐                        │       │
│  │            ▼                          ▼                        │       │
│  │  ┌─────────────────────┐   ┌─────────────────────┐             │       │
│  │  │    Advance          │   │   Correction Loop   │             │       │
│  │  │  (Write to Buffer)  │   │  Rewind→Inpaint→    │             │       │
│  │  │                     │   │  Regenerate→Verify  │             │       │
│  │  └─────────────────────┘   └─────────────────────┘             │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                                 │                          │
│                                                 ▼                          │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │                      Output Buffer                              │       │
│  │                 (Verified Frames Ready for Export)              │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Component Specifications

#### 5.2.1 Base Model
| Component | Specification |
|-----------|---------------|
| **Model** | Video diffusion backbone (SD-based architecture) |
| **Framework** | PyTorch ≥ 2.0 |
| **CUDA** | ≥ 11.8 |
| **Resolution** | 512×512 default (configurable) |
| **Frame Count** | 16–60 frames per sequence |

#### 5.2.2 Identity Stack
| Component | Model | Output |
|-----------|-------|--------|
| **Face Detection** | RetinaFace (buffalo_l) | Bounding boxes |
| **Embedding Extraction** | ArcFace (insightface) | 512-d vector |
| **Identity Conditioning** | IP-Adapter Plus | Cross-attention injection |
| **Segmentation** | SAM or DWPose | Pixel mask |

#### 5.2.3 Kinematic Stack
| Component | Model | Output |
|-----------|-------|--------|
| **Pose Estimation** | DWPose (ControlNet) | 17-keypoint skeleton |
| **Bone Tracking** | Custom | Bone length array |
| **ROM Enforcement** | Configurable joint limits | Pass/Fail per joint |
| **Velocity Tracking** | Frame delta computation | Velocity vectors |

#### 5.2.4 Loss Functions
| Loss | Formula | Threshold |
|------|---------|-----------|
| **L_identity** | `1 - cos(embedding_ref, embedding_gen)` | ≤ 0.10 |
| **L_bone** | `Σ_t || d(j_a, j_b)_t − d(j_a, j_b)_{t-1} ||^2` | ≤ tolerance |
| **L_ROM** | `max(0, angle - limit)` per joint | = 0 |
| **L_velocity** | `max(0, ||v|| - v_max)` | = 0 |
| **L_physics** | `L_bone + L_ROM + L_velocity` | ≤ 0.01 |

### 5.3 Compute & Infrastructure

| Resource | Specification | Notes |
|----------|---------------|-------|
| **Primary Dev** | Google Colab Pro | T4/A100 GPUs |
| **Extended Compute** | IndiaAI Mission | A100/H100 for heavy loads |
| **VRAM Budget** | 16GB (T4) / 24GB (A100) | Quantization for T4 |
| **Persistence** | Google Drive | Mount at runtime |
| **Python Version** | ≥ 3.9 | |
| **PyTorch Version** | ≥ 2.0 | |

### 5.4 Memory Optimization

| Technique | Implementation | Target |
|-----------|----------------|--------|
| **8-bit Quantization** | bitsandbytes | 16GB VRAM |
| **Attention Slicing** | PyTorch SDPA | Reduced memory |
| **Gradient Checkpointing** | torch.checkpoint | VRAM trade latency |
| **CPU Offloading** | Accelerated | Inactive tensors |

---

## 6. Development Requirements

### 6.1 Repository Structure
```
/core                    # Inference engine & verification daemon
/adapters               # IP-Adapter & ControlNet models
/notebooks              # Colab-ready experimentation notebooks
/sdk                    # Python package (pivot-sdk)
/scripts                # Evaluation & benchmarking utilities
```

### 6.2 Required Components
- [ ] Video diffusion model (SD-based backbone)
- [ ] InsightFace/ArcFace embedding extraction
- [ ] IP-Adapter implementation
- [ ] DWPose/ControlNet pose estimation
- [ ] L_physics loss function implementation
- [ ] Verification daemon with rewind/inpaint loop
- [ ] Unit test suite for all constraints
- [ ] Experiment notebooks in /notebooks

---

## 7. Evaluation Criteria

### 7.1 Unit Test Requirements

| Test ID | Test Description | Pass Condition |
|---------|------------------|----------------|
| **UT-001** | Bone length invariance | L_bone ≈ 0 across all frames |
| **UT-002** | ROM enforcement | All joints within limits |
| **UT-003** | Velocity limits | No joint exceeds v_max |
| **UT-004** | Identity similarity | Cosine ≥ 0.90 all frames |
| **UT-005** | Rewind execution | Latent state preserved |
| **UT-006** | Inpainting mask | Pixel-precise failure region |
| **UT-007** | Correction loop | Regeneration passes verification |
| **UT-008** | Max retry handling | Graceful degradation at limit |

### 7.2 Integration Test Requirements

| Test ID | Test Description | Pass Condition |
|---------|------------------|----------------|
| **IT-001** | Single character generation | Identity stable, kinematics valid |
| **IT-002** | Daemon-triggered correction | Auto-correction executes |
| **IT-003** | End-to-end pipeline | Verified output buffer filled |

---

## 8. Success Criteria (Definition of Done)

| Criteria | Verification Method |
|----------|---------------------|
| **SC-001** Pipeline generates single-character outputs with Identity Router and Kinematic Guardrail | Visual inspection of output |
| **SC-002** Automated L_physics unit tests passing across all three metrics | `pytest tests/physics/` |
| **SC-003** Cosine similarity ≥ 0.90 verified on all frames | Daemon logs |
| **SC-004** Rewind/Inpaint loop triggers correctly on violations | Simulated failure tests |
| **SC-005** Notebooks versioned and committed to /notebooks | Git commit history |
| **SC-006** Zero manual intervention required for correction | End-to-end test |

---

## 9. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| VRAM exhaustion on T4 | High | Pipeline failure | 8-bit quantization + attention slicing |
| Pose estimation accuracy | Medium | False positives | Confidence threshold tuning |
| Latent rewind instability | Medium | Quality degradation | Strict state preservation |
| T4 VRAM constraints for full pipeline | High | Feature reduction | A100 for full pipeline validation |

---

## 10. Timeline & Milestones

| Milestone | Deliverable | Target |
|-----------|-------------|--------|
| **M1** | Identity Router functional | Reference → IP-Adapter pipeline works |
| **M2** | Kinematic Guardrail functional | L_physics constraints enforced |
| **M3** | Verification Daemon operational | Detect → Rewind → Inpaint loop works |
| **M4** | Unit tests passing | All physics tests green |
| **M5** | End-to-end pipeline validated | Full single-character generation works |
| **M6** | Notebooks committed | All experiments in /notebooks |

---

## 11. Dependencies

| Dependency | Version | Source |
|------------|---------|--------|
| PyTorch | ≥ 2.0 | pip |
| CUDA | ≥ 11.8 | NVIDIA |
| InsightFace | latest | pip |
| Diffusers | ≥ 0.25 | pip |
| ControlNet | latest | Hugging Face |
| SAM | latest | Meta |

---

## 12. Out of Scope (Phase 1)

The following features are explicitly deferred to future phases:

- **Multi-character support** (Phase 3+)
- **Environmental physics** (Phase 5+)
- **Emotion/FACS enforcement** (Phase 2+)
- **REST API** (Phase 6)
- **Python package** (Phase 6)
- **ComfyUI node** (Phase 6)

---

*Phase 1PRD — P.I.V.O.T. — Confidential — Active Development*
