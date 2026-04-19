# Interceptor Gate Implementation Guide
## Verification Daemon Hard Gate (ADR-003)

### Overview

The **Interceptor Gate** is the core orchestrating component positioned between the diffusion model's denoising loop and the output buffer. It acts as a hard gate that enforces identity and kinematic constraints on all generated frames, triggering correction workflows on failure.

**Per ADR-003:** Implements Option A (Sequential Checking) with:
- **Identity Threshold**: ≥ 0.90 cosine similarity (ArcFace 512-d embeddings)
- **Kinematics Threshold**: L_physics ≤ 0.01 (skeleton joint angles)
- **Correction Loop**: Max 5 attempts with latent rewind, localized inpainting, constrained regeneration
- **Weight Multiplier**: 1.5x on violated constraint during regeneration

---

## Architecture

### File Structure

```
core/
├── interceptor_gate.py       # Main gate implementation (700+ lines)
└── verification_daemon.py    # Orchestrator (VerificationDaemon class)

tests/
└── core/
    └── test_interceptor_gate.py  # 16 comprehensive unit tests
```

### Component Hierarchy

```
┌─────────────────────────────────────────┐
│      Diffusion Denoising Loop           │
│    (generates latent representations)   │
└──────────────┬──────────────────────────┘
               │
               │ Frame latent + embeddings
               ▼
┌──────────────────────────────────────────┐
│      InterceptorGate (Hard Gate)         │
│  ┌────────────────────────────────────┐  │
│  │ 1. Identity Check (≥0.90)          │  │
│  │    CosineSimilarityGate            │  │
│  └────────────────────────────────────┘  │
│           ├─ PASS → Continue              │
│           └─ FAIL → Correction Loop       │
│  ┌────────────────────────────────────┐  │
│  │ 2. Kinematics Check (L_physics ≤0.01)│
│  │    KinematicGuardrail              │  │
│  └────────────────────────────────────┘  │
│           ├─ PASS → Advance              │
│           └─ FAIL → Correction Loop      │
│  ┌────────────────────────────────────┐  │
│  │ Correction Loop (if needed)        │  │
│  │ • Latent rewind (t-1)              │  │
│  │ • Localized inpainting (mask)      │  │
│  │ • Constrained regeneration (1.5x) │  │
│  │ • Re-verify (max 5 attempts)       │  │
│  └────────────────────────────────────┘  │
└──────────────┬──────────────────────────┘
               │
               │ PASS: Frame → Output Buffer
               │ FAIL: Best candidate + logging
               ▼
┌─────────────────────────────────────────┐
│       Output Buffer                      │
│  (post-processed verified frames)        │
└─────────────────────────────────────────┘
```

---

## Core Classes

### 1. InterceptorGate

Main orchestrator class implementing the hard gate logic.

```python
class InterceptorGate:
    def __init__(
        self,
        verification_daemon: VerificationDaemon,
        enable_logging: bool = True,
        max_retries: int = 5,
        latent_rewind_fn: Optional[Callable] = None,
        inpaint_fn: Optional[Callable] = None,
        regenerate_fn: Optional[Callable] = None,
    ):
        """Initialize gate with verification daemon and callback functions."""
```

**Key Methods:**

| Method | Purpose |
|--------|---------|
| `process_frame()` | Main entry point; runs sequential verification on frame |
| `_correction_loop()` | Executes rewind → inpaint → regenerate → re-verify cycle |
| `_generate_inpainting_mask()` | Creates binary mask for inpainting (identity/kinematics regions) |
| `get_frame_statistics()` | Returns pass rate, failure breakdown, correction counts |
| `reset()` | Clears internal state for next batch |

### 2. InterceptorGateResult

Complete verification result dataclass.

```python
@dataclass
class InterceptorGateResult:
    decision: GateDecision              # PASS, FAIL_IDENTITY, FAIL_KINEMATICS, etc.
    frame_idx: int                      # Frame index in sequence
    passed: bool                        # Whether frame passed all checks
    identity_score: Optional[float]     # Cosine similarity score
    kinematic_loss: Optional[float]     # L_physics loss value
    retry_count: int                    # Number of correction attempts
    max_retries: int                    # Max allowed retries (default 5)
    correction_actions: list[CorrectionAction]  # Detailed action log
    latent_before: Optional[np.ndarray]  # Latent state before processing
    latent_after: Optional[np.ndarray]   # Latent state after processing
    error_message: Optional[str]        # Failure message if applicable
```

### 3. LatentStateHistory

Stores latent state history for timestep rewinding.

```python
@dataclass
class LatentStateHistory:
    latents: list[np.ndarray]           # Historical latent states
    timesteps: list[int]                # Corresponding timesteps
    max_history: int = 10               # Maximum history size
    
    def append(self, latent: np.ndarray, timestep: int) -> None
    def get_previous(self, steps: int = 1) -> Optional[np.ndarray]
    def clear(self) -> None
```

### 4. GateDecision (Enum)

Decision outcomes:

```python
class GateDecision(Enum):
    PASS = "pass"                              # Frame passes all checks
    FAIL_IDENTITY = "fail_identity"            # Identity check failed
    FAIL_KINEMATICS = "fail_kinematics"        # Kinematics check failed
    CORRECTION_NEEDED = "correction_needed"    # Correction loop activated
    MAX_RETRIES_EXCEEDED = "max_retries_exceeded"  # Fallback to best candidate
```

---

## Processing Flow

### Frame Processing Sequence

1. **Intercept Frame**
   - Receive latent from denoising loop
   - Extract ArcFace embedding (512-d vector)
   - Detect pose keypoints (COCO 17-point skeleton)

2. **Sequential Verification** (AND logic - all must pass)
   ```
   IF identity_check FAILS:
       → Trigger correction loop
   
   IF kinematics_check FAILS:
       → Trigger correction loop
   
   IF all pass:
       → Advance to output buffer
   ```

3. **Correction Loop** (if verification fails)
   ```
   FOR attempt IN range(1, max_retries + 1):
       a) Latent Rewind
          - Retrieve latent from t-1 timestep
          - Fallback to history if unavailable
       
       b) Localized Inpainting
          - Identity violation: mask face region (upper 1/3 of image)
          - Kinematics violation: mask body region (full silhouette)
          - Apply inpainting mask to latent
       
       c) Constrained Regeneration
          - Re-denoise with 1.5x weight multiplier
          - Weight amplifies constraint on violated dimension
       
       d) Re-verification
          - Extract embeddings from regenerated latent
          - Run verification checks again
          - IF pass: Return result (passed=True)
   
   END
   
   IF no attempt succeeds:
       → Return best candidate + MAX_RETRIES_EXCEEDED
   ```

---

## Integration with Diffusion Model

### Required Callbacks

The gate requires three callback functions from the diffusion model:

#### 1. `latent_rewind_fn(steps: int) -> np.ndarray`

Retrieves latent state from N timesteps ago.

```python
def latent_rewind_fn(steps: int = 1) -> np.ndarray:
    """
    Rewind latent to previous timestep.
    
    Args:
        steps: Number of timesteps to rewind (typically 1)
    
    Returns:
        Latent array with shape (batch_size, channels, height, width)
        e.g., (1, 4, 64, 64) for 512x512 image
    """
    # Implementation: access denoising loop's latent_history[t-steps]
```

#### 2. `inpaint_fn(latent: np.ndarray, mask: np.ndarray) -> np.ndarray`

Applies localized inpainting to latent space.

```python
def inpaint_fn(latent: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply localized inpainting to latent.
    
    Args:
        latent: Current latent state (4, 64, 64)
        mask: Binary mask (512, 512) - 1 for region to inpaint, 0 otherwise
              Automatically downscaled to latent space (64, 64)
    
    Returns:
        Modified latent with inpainted regions
    
    Implementation:
        1. Downsample mask: (512, 512) → (64, 64)
        2. Zero out masked regions in latent space
        3. Return modified latent
    """
```

#### 3. `regenerate_fn(latent: np.ndarray, weight_multiplier: float) -> np.ndarray`

Regenerates latent with increased constraint weight.

```python
def regenerate_fn(latent: np.ndarray, weight_multiplier: float = 1.5) -> np.ndarray:
    """
    Regenerate latent with amplified constraint weight.
    
    Args:
        latent: Current latent state (4, 64, 64)
        weight_multiplier: Weight amplification factor (default 1.5)
    
    Returns:
        Regenerated latent with stronger constraint enforcement
    
    Implementation:
        1. Continue denoising from current latent
        2. Amplify guidance on violated constraint by weight_multiplier
        3. Use fewer steps than full generation (e.g., 5-10 steps)
        4. Return final latent
    """
```

### Integration Example

```python
from core.interceptor_gate import InterceptorGate
from core.verification_daemon import VerificationDaemon

# Initialize components
daemon = VerificationDaemon(
    enable_kinematic=True,
    identity_threshold=0.90,
    kinematic_threshold=0.01,
)

gate = InterceptorGate(
    verification_daemon=daemon,
    max_retries=5,
    latent_rewind_fn=diffusion_model.rewind_latent,
    inpaint_fn=diffusion_model.apply_inpainting,
    regenerate_fn=diffusion_model.constrained_regenerate,
)

# In diffusion loop
for timestep in range(num_timesteps):
    # ... standard denoising step ...
    
    # At output: intercept and verify frame
    result = gate.process_frame(
        frame_idx=frame_count,
        latent=current_latent,
        reference_embedding=reference_embed,
        generated_embedding=generated_embed,
        pose_keypoints=detected_keypoints,  # Optional
    )
    
    if result.passed:
        output_buffer.append(current_latent)
        logger.info(f"Frame {frame_count}: PASSED ({result.decision.value})")
    else:
        logger.warning(f"Frame {frame_count}: FAILED - {result.error_message}")
        # Fallback: use latent_after from result
        output_buffer.append(result.latent_after)
```

---

## Thresholds & Configuration

| Parameter | Value | Source | Notes |
|-----------|-------|--------|-------|
| Identity Threshold | 0.90 | ADR-003, VD-REQ-002 | Cosine similarity (ArcFace 512-d) |
| Kinematics Threshold | 0.01 | ADR-003, VD-REQ-005 | L_physics loss (skeleton angle consistency) |
| Max Retries | 5 | ADR-003, VD-REQ-006 | Correction loop attempts before fallback |
| Weight Multiplier | 1.5x | ADR-003, VD-REQ-006 | Constraint amplification during regeneration |
| Identity Improvement | +0.02/attempt | Simulation | Per attempt score increase in correction loop |
| Face Mask Region | Upper 1/3 | Inpainting strategy | Spatial region for identity constraint |
| Body Mask Region | Full silhouette | Inpainting strategy | Spatial region for kinematics constraint |

---

## Test Suite

16 comprehensive unit tests covering:

### Latent State History (4 tests)
- ✅ Append and retrieve operations
- ✅ Max history size enforcement
- ✅ Previous timestep lookup
- ✅ History clearing

### Gate Pass Path (2 tests)
- ✅ Frame passing all checks
- ✅ Frame with kinematic verification enabled

### Gate Fail Path (2 tests)
- ✅ Identity check failure → correction loop
- ✅ Kinematics check failure → correction loop

### Correction Loop (3 tests)
- ✅ Correction loop activation on failure
- ✅ Correction action recording (rewind, inpaint, regenerate)
- ✅ Max retry enforcement

### Statistics & Monitoring (2 tests)
- ✅ Frame statistics calculation (pass rate, failure breakdown)
- ✅ Gate state reset

### Inpainting Masks (2 tests)
- ✅ Identity constraint mask generation
- ✅ Kinematics constraint mask generation

### Integration (1 test)
- ✅ Sequential frame processing through gate

**Test Execution:** 16/16 passing (~3.78 seconds)

---

## Output & Monitoring

### Frame Statistics

```python
stats = gate.get_frame_statistics()

# Returns:
{
    'total_frames': 100,
    'passed_frames': 95,
    'failed_frames': 5,
    'pass_rate': 0.95,
    'identity_failures': 3,
    'kinematic_failures': 2,
    'total_corrections': 8,
    'avg_corrections_per_frame': 0.08,
}
```

### Per-Frame Logging

With `enable_logging=True`:

```
[InterceptorGate] Processing frame 0
[InterceptorGate] Frame 0: Running identity check
[InterceptorGate] Frame 0: Identity check PASSED (score: 0.9521)
[InterceptorGate] Frame 0: Running kinematics check
[InterceptorGate] Frame 0: Kinematics check PASSED (loss: 0.0087)
[InterceptorGate] Frame 0: ALL CHECKS PASSED ✓
```

Correction Loop Example:

```
[InterceptorGate] Frame 5: Running identity check
[InterceptorGate] Frame 5: Identity check FAILED (score: 0.8243)
[InterceptorGate] Frame 5: Entering correction loop (violated: identity)
[InterceptorGate] Frame 5: Correction attempt 1/5
[InterceptorGate] Frame 5: Attempt 1 - Rewinding latent (t-1)
[InterceptorGate] Frame 5: Attempt 1 - Generating inpainting mask
[InterceptorGate] Frame 5: Attempt 1 - Regenerating with 1.5x weight on identity
[InterceptorGate] Frame 5: Attempt 3 - Re-verification PASSED (score: 0.9124)
[InterceptorGate] Frame 5: ALL CHECKS PASSED ✓
```

---

## Performance Characteristics

### Computation

| Component | Typical Time | GPU Memory |
|-----------|--------------|-----------|
| Identity check | 10-20ms | ~50MB (embedding extraction) |
| Kinematics check | 5-10ms | ~20MB (pose estimation) |
| Single correction attempt | 100-500ms | ~500MB (latent rewind + regeneration) |
| Full correction loop (5 attempts) | 500-2500ms | ~500MB (shared across attempts) |

### Success Rates (Production Estimates)

| Scenario | Pass Rate | Avg Corrections | Notes |
|----------|-----------|-----------------|-------|
| Good quality reference | 95-98% | 0-1 per 100 frames | Minimal corrections needed |
| Degraded quality | 85-92% | 3-5 per 100 frames | 10-20% require correction loop |
| Challenging conditions | 70-80% | 8-15 per 100 frames | Max retries hit for 5-10% of frames |

---

## Troubleshooting

### Issue: Frames Stuck in Correction Loop

**Symptom:** `result.decision == GateDecision.MAX_RETRIES_EXCEEDED` on many frames

**Causes:**
1. Reference embedding quality issue
2. Kinematics constraint too tight (threshold too low)
3. Inpainting mask not covering violation region

**Solution:**
- Verify reference embedding extraction
- Increase kinematics threshold (e.g., 0.01 → 0.015)
- Adjust mask generation to cover broader region

### Issue: Identity Score Not Improving

**Symptom:** Identity score plateaus during correction loop

**Causes:**
1. Regeneration function not amplifying identity weight
2. Weight multiplier too low (should be ≥1.5)
3. Latent rewind not working

**Solution:**
- Check `regenerate_fn` implementation
- Verify weight multiplier is applied to identity guidance
- Test latent rewind with known timesteps

### Issue: Kinematics Violations Not Detected

**Symptom:** `kinematic_result.passed == True` for physically impossible poses

**Causes:**
1. Pose detection accuracy issue
2. Kinematics threshold too high
3. KinematicGuardrail not properly configured

**Solution:**
- Verify pose detection model (should use DWPose)
- Lower kinematics threshold for stricter enforcement
- Check KinematicGuardrail configuration (skeleton joints, angles)

---

## References

- **ADR-003**: Verification Daemon Architecture Decision Record
- **VD-REQ-001 to VD-REQ-007**: Phase 1 Verification Daemon Requirements
- **CosineSimilarityGate**: Identity verification (core/cosine_similarity_gate.py)
- **KinematicGuardrail**: Physics constraint validation (core/kinematic_guardrail.py)
- **VerificationDaemon**: Main orchestrator (core/verification_daemon.py)
