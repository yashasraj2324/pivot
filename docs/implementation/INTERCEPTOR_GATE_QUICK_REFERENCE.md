# Interceptor Gate Quick Reference
## 5-Minute Integration Guide

### Installation & Setup

```python
from core.interceptor_gate import InterceptorGate
from core.verification_daemon import VerificationDaemon

# Step 1: Initialize VerificationDaemon
daemon = VerificationDaemon(enable_kinematic=True)

# Step 2: Create InterceptorGate with diffusion callbacks
gate = InterceptorGate(
    verification_daemon=daemon,
    max_retries=5,
    latent_rewind_fn=your_diffusion_model.rewind_latent,
    inpaint_fn=your_diffusion_model.apply_inpainting,
    regenerate_fn=your_diffusion_model.constrained_regenerate,
)
```

### Basic Usage

```python
# In your diffusion loop:
for frame in frames:
    # Generate embeddings/keypoints
    reference_embedding = extract_arcface_embedding(reference_image)
    generated_embedding = extract_arcface_embedding(generated_image)
    pose_keypoints = detect_pose(generated_image)  # COCO 17-point
    
    # Process frame through gate
    result = gate.process_frame(
        frame_idx=frame_count,
        latent=current_latent,
        reference_embedding=reference_embedding,
        generated_embedding=generated_embedding,
        pose_keypoints=pose_keypoints,
    )
    
    # Handle result
    if result.passed:
        output_buffer.append(result.latent_after)
        print(f"Frame {frame_count}: ✓ PASSED")
    else:
        print(f"Frame {frame_count}: ✗ FAILED - {result.error_message}")
        output_buffer.append(result.latent_after)  # Use best candidate
```

### Key Concepts

| Concept | Definition | Threshold |
|---------|-----------|-----------|
| **Identity Check** | Cosine similarity of ArcFace embeddings | ≥0.90 |
| **Kinematics Check** | Physics loss of skeleton joint angles | ≤0.01 |
| **Sequential Verification** | All checks must pass (AND logic) | N/A |
| **Correction Loop** | Rewind → Inpaint → Regenerate → Verify | Up to 5x |
| **Weight Multiplier** | Constraint amplification during regeneration | 1.5x |

### API Reference

```python
# Main entry point
result = gate.process_frame(
    frame_idx: int,
    latent: np.ndarray,                    # (batch, 4, 64, 64)
    reference_embedding: np.ndarray,       # (512,)
    generated_embedding: np.ndarray,       # (512,)
    pose_keypoints: Optional[np.ndarray],  # (17, 3) optional
) -> InterceptorGateResult

# Result attributes
result.decision          # GateDecision enum: PASS, FAIL_*, CORRECTION_NEEDED, MAX_RETRIES_EXCEEDED
result.passed          # bool: True if frame passed all checks
result.identity_score  # float: Cosine similarity (0.0-1.0)
result.kinematic_loss  # float: Physics loss value
result.retry_count     # int: Number of correction attempts (0 if no retries)
result.correction_actions  # list: Detailed action log (rewind, inpaint, regenerate)
result.latent_after    # np.ndarray: Final latent state
result.error_message   # str: Failure reason (if applicable)

# Utility methods
stats = gate.get_frame_statistics()  # Returns dict with pass rate, failure breakdown
gate.reset()                         # Clear internal state
```

### Result Decision Enum

```python
class GateDecision(Enum):
    PASS = "pass"                              # ✓ All checks passed
    FAIL_IDENTITY = "fail_identity"            # ✗ Identity < 0.90
    FAIL_KINEMATICS = "fail_kinematics"        # ✗ L_physics > 0.01
    CORRECTION_NEEDED = "correction_needed"    # ⚙ Correction loop activated
    MAX_RETRIES_EXCEEDED = "max_retries_exceeded"  # ⚠ Fallback to best candidate
```

### Statistics & Monitoring

```python
# Get comprehensive statistics
stats = gate.get_frame_statistics()

# Returns dict:
{
    'total_frames': int,              # Total frames processed
    'passed_frames': int,             # Frames passing all checks
    'failed_frames': int,             # Frames failing verification
    'pass_rate': float,               # Percentage passing (0.0-1.0)
    'identity_failures': int,         # Frames failing identity check
    'kinematic_failures': int,        # Frames failing kinematics check
    'total_corrections': int,         # Total correction actions executed
    'avg_corrections_per_frame': float,  # Average corrections needed
}

# Example output
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

### Callback Function Templates

```python
# 1. Latent Rewind
def latent_rewind_fn(steps: int = 1) -> np.ndarray:
    """Retrieve latent from t-steps timesteps ago."""
    return denoising_loop.latent_history[-steps-1]  # Returns (batch, 4, 64, 64)

# 2. Inpainting
def inpaint_fn(latent: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply localized inpainting mask to latent."""
    # mask: (512, 512) binary mask - downsample to (64, 64)
    mask_downsampled = torch.nn.functional.interpolate(
        torch.from_numpy(mask[None, None]).float(),
        size=(64, 64),
        mode='nearest'
    ).numpy()[0, 0]
    
    latent[mask_downsampled > 0.5] = 0  # Zero out masked regions
    return latent

# 3. Constrained Regeneration
def regenerate_fn(latent: np.ndarray, weight_multiplier: float = 1.5) -> np.ndarray:
    """Regenerate latent with amplified constraint."""
    # Continue denoising from current latent
    for _ in range(5):  # 5 additional denoising steps
        latent = denoise_step(
            latent,
            guidance_scale=guidance_scale * weight_multiplier,  # Amplify!
        )
    return latent
```

### Common Patterns

#### Pattern 1: Verify Single Frame
```python
result = gate.process_frame(
    frame_idx=0,
    latent=latent,
    reference_embedding=ref_embed,
    generated_embedding=gen_embed,
    pose_keypoints=keypoints,
)

if result.decision == GateDecision.PASS:
    print("✓ Frame verified successfully")
```

#### Pattern 2: Process Sequence with Statistics
```python
for i, frame in enumerate(frames):
    result = gate.process_frame(
        frame_idx=i,
        latent=frame.latent,
        reference_embedding=frame.ref_embed,
        generated_embedding=frame.gen_embed,
        pose_keypoints=frame.keypoints,
    )
    frames[i].verification_result = result

# Get aggregate statistics
stats = gate.get_frame_statistics()
print(f"Pass rate: {stats['pass_rate']:.1%}")
print(f"Avg corrections: {stats['avg_corrections_per_frame']:.2f}")
```

#### Pattern 3: Monitor Correction Loop
```python
result = gate.process_frame(...)

if result.retry_count > 0:
    print(f"Frame required {result.retry_count} correction attempts")
    for action in result.correction_actions:
        print(f"  - {action.action_type}: {action.notes}")
```

#### Pattern 4: Batch Processing with Reset
```python
for batch_num in range(num_batches):
    # Process batch
    for frame in batch:
        result = gate.process_frame(...)
    
    # Log batch statistics
    stats = gate.get_frame_statistics()
    log_batch_stats(batch_num, stats)
    
    # Reset for next batch
    gate.reset()
```

### Threshold Tuning

```python
# Stricter verification (fewer false passes, more corrections)
gate.daemon.identity_gate.threshold = 0.95  # Default: 0.90
gate.daemon.kinematic_threshold = 0.005     # Default: 0.01

# More lenient (faster processing, more corrections accepted)
gate.daemon.identity_gate.threshold = 0.85  # Default: 0.90
gate.daemon.kinematic_threshold = 0.02      # Default: 0.01

# More correction attempts
gate.max_retries = 10  # Default: 5
```

### Logging Configuration

```python
import logging

# Enable detailed logging
gate = InterceptorGate(
    verification_daemon=daemon,
    enable_logging=True,  # Detailed step logging
)

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('core.interceptor_gate')

# Example output:
# [InterceptorGate] Processing frame 0
# [InterceptorGate] Frame 0: Running identity check
# [InterceptorGate] Frame 0: Identity check PASSED (score: 0.9521)
# [InterceptorGate] Frame 0: ALL CHECKS PASSED ✓
```

### Error Handling

```python
try:
    result = gate.process_frame(...)
except AttributeError as e:
    print(f"Invalid input shape: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")

# Check result for errors
if not result.passed:
    print(f"Error: {result.error_message}")
    if result.decision == GateDecision.MAX_RETRIES_EXCEEDED:
        # Use best candidate from correction loop
        best_latent = result.latent_after
```

### Performance Tips

1. **Batch Processing**: Process multiple frames together for efficiency
2. **Selective Kinematics**: Only enable if pose detection is fast (DWPose recommended)
3. **Callback Optimization**: Cache embeddings/keypoints to avoid recomputation
4. **Early Exit**: Skip kinematics check if already computationally expensive
5. **Adaptive Thresholds**: Use lower thresholds for high-quality references

### Test Suite

Run tests to verify integration:

```bash
pytest tests/core/test_interceptor_gate.py -v

# Output:
# test_append_stores_latent_and_timestep PASSED
# test_frame_passes_all_checks PASSED
# test_correction_loop_triggered_on_failure PASSED
# ...
# 16 passed in 3.78s
```

---

## Quick Links

- **Full Documentation**: [INTERCEPTOR_GATE_IMPLEMENTATION.md](INTERCEPTOR_GATE_IMPLEMENTATION.md)
- **ADR-003**: Verification Daemon Architecture Decision Record
- **Test Suite**: [tests/core/test_interceptor_gate.py](../../tests/core/test_interceptor_gate.py)
- **Implementation**: [core/interceptor_gate.py](../../core/interceptor_gate.py)
