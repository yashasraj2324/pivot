# Verification Daemon Test Suite — Quick Start Guide

## Quick Overview

The Verification Daemon is the enforcement layer that validates each generated frame against:
1. **Identity Check** — Face must match reference image (cosine similarity ≥ 0.90)
2. **Kinematics Check** — Motion must be biomechanically valid (L_physics ≤ 0.01)

Both checks must pass; if either fails, the correction loop automatically rewinds and regenerates.

---

## Running Tests

```bash
# Run all tests
pytest tests/core/test_verification_daemon.py -v

# Run with step messages (shows [Verification Daemon] trace)
pytest tests/core/test_verification_daemon.py -v -s

# Run specific test class
pytest tests/core/test_verification_daemon.py::TestIdentityCheck -v

# Run specific test
pytest tests/core/test_verification_daemon.py::TestIdentityCheck::test_identity_check_perfect_match_returns_1_0 -v

# Run with coverage
pytest tests/core/test_verification_daemon.py --cov=core --cov-report=term-missing
```

---

## Test Checklist

### ✅ Identity Check (5 tests)
- [x] Perfect match returns 1.0
- [x] Orthogonal embeddings return ~0.0
- [x] High similarity (0.95) passes threshold
- [x] Low similarity (0.0) fails threshold
- [x] Works with unnormalized embeddings

### ✅ Kinematics Check (3 tests)
- [x] Returns valid L_physics score
- [x] Includes bone length invariance
- [x] Valid pose passes threshold

### ✅ Frame Verification (3 tests)
- [x] Both checks pass → frame passes
- [x] Identity fails → frame fails
- [x] Verification tracked in history

### ✅ Latent Rewind (3 tests)
- [x] Single step modifies latent
- [x] Multi-step shows cumulative effect
- [x] Rewind recorded in correction history

### ✅ Mask Generation (3 tests)
- [x] Returns binary mask array
- [x] Respects image bounds
- [x] Handles low-confidence keypoints

### ✅ Constrained Regeneration (2 tests)
- [x] Applies weight multiplier (1.5x)
- [x] Records constraint type

### ✅ Correction Loop (3 tests)
- [x] May succeed within retries
- [x] Never exceeds max depth (5)
- [x] Generates all action types

### ✅ Sequential Verification (2 tests)
- [x] All frames in sequence pass
- [x] Mix of passing and failing frames

### ✅ Daemon State (3 tests)
- [x] Custom thresholds configurable
- [x] Verification history tracked
- [x] Correction history tracked

---

## Test Class Reference

### TestIdentityCheck
**Purpose**: Verify cosine similarity gate works correctly

**Key Methods**:
- `daemon.check_identity(generated_embedding, reference_embedding)` → float [0, 1]
- Validates: `score >= 0.90` (PASS) or `score < 0.90` (FAIL)

**Example**:
```python
reference = np.random.randn(512).astype(np.float32)
reference = reference / np.linalg.norm(reference)
generated = reference.copy()  # Perfect match

score = daemon.check_identity(generated, reference)
assert score == 1.0
```

---

### TestKinematicsCheck
**Purpose**: Verify L_physics composite metric

**Key Methods**:
- `daemon.check_kinematics(keypoints, bone_lengths_ref=None)` → float [0, ∞)
- Validates: `l_physics <= 0.01` (PASS) or `l_physics > 0.01` (FAIL)

**Example**:
```python
keypoints = np.zeros((17, 3), dtype=np.float32)
keypoints[:, 2] = 0.9  # High confidence
keypoints[5] = [200, 200, 0.95]  # Left shoulder
keypoints[6] = [312, 200, 0.95]  # Right shoulder

l_physics = daemon.check_kinematics(keypoints)
assert l_physics <= 0.01
```

---

### TestFrameVerification
**Purpose**: Test identity + kinematics checks together (AND logic)

**Key Methods**:
- `daemon.verify_frame(generated_embedding, reference_embedding, keypoints, latent, frame_idx)` → (bool, VerificationState)

**Example**:
```python
passed, state = daemon.verify_frame(
    generated_embedding=my_embedding,
    reference_embedding=ref_embedding,
    keypoints=my_keypoints,
    latent=my_latent,
    frame_idx=0
)
assert passed  # Only true if both checks pass
```

---

### TestLatentRewind
**Purpose**: Test rollback mechanism for diffusion state

**Key Methods**:
- `daemon.rewind_latent(latent, steps=1)` → np.ndarray

**Example**:
```python
latent = np.random.randn(4, 64, 64).astype(np.float32)
rewound = daemon.rewind_latent(latent, steps=1)
assert rewound.shape == latent.shape
assert not np.allclose(rewound, latent)  # Modified
```

---

### TestLocalizedMask
**Purpose**: Test inpainting region generation from skeleton

**Key Methods**:
- `daemon.generate_localized_mask(keypoints, image_shape=(H, W), margin=50)` → np.ndarray

**Example**:
```python
keypoints = np.random.rand(17, 3).astype(np.float32)
mask = daemon.generate_localized_mask(keypoints, image_shape=(512, 512))
assert mask.shape == (512, 512)
assert 0.0 <= mask.min() and mask.max() <= 1.0
```

---

### TestConstrainedRegeneration
**Purpose**: Test increased-weight regeneration

**Key Methods**:
- `daemon.apply_constrained_regeneration(latent, violated_constraint, weight_multiplier=1.5)` → np.ndarray

**Example**:
```python
latent = np.ones((4, 64, 64), dtype=np.float32)
regenerated = daemon.apply_constrained_regeneration(
    latent,
    violated_constraint="identity",
    weight_multiplier=1.5
)
assert regenerated.shape == latent.shape
```

---

### TestCorrectionLoop
**Purpose**: Test full rewind-inpaint-regenerate-verify workflow

**Key Methods**:
- `daemon.correction_loop(latent, generated_embedding, reference_embedding, keypoints, frame_idx)` → (bool, int, VerificationState)

**Example**:
```python
passed, retry_count, state = daemon.correction_loop(
    latent=my_latent,
    generated_embedding=my_embedding,
    reference_embedding=ref_embedding,
    keypoints=my_keypoints,
    frame_idx=5
)
assert retry_count <= 5  # Max retries enforced
print(f"Passed after {retry_count} attempts")
```

---

## Common Test Patterns

### Pattern 1: Create Valid Embedding
```python
embedding = np.random.randn(512).astype(np.float32)
embedding = embedding / np.linalg.norm(embedding)  # Normalize
# Now: norm(embedding) ≈ 1.0
```

### Pattern 2: Create Valid Keypoints
```python
keypoints = np.zeros((17, 3), dtype=np.float32)
keypoints[:, 2] = 0.9  # Set confidence to 0.9 (high)
# Set specific joints:
keypoints[5] = [200, 200, 0.95]   # Left shoulder
keypoints[6] = [312, 200, 0.95]   # Right shoulder
```

### Pattern 3: Create Latent State
```python
latent = np.random.randn(4, 64, 64).astype(np.float32)
# Shape: (channels=4, height=64, width=64)
```

### Pattern 4: Test with Fixtures
```python
@pytest.fixture
def daemon(self):
    return MockVerificationDaemon(
        identity_threshold=0.90,
        physics_threshold=0.01,
        max_retry_depth=5,
    )

def test_something(self, daemon):
    # daemon is auto-provided by pytest
    score = daemon.check_identity(gen, ref)
    assert score >= daemon.identity_threshold
```

---

## Adding New Tests

### Step 1: Choose Test Class
- **Identity-related?** → Add to `TestIdentityCheck`
- **Kinematics-related?** → Add to `TestKinematicsCheck`
- **Full verification?** → Add to `TestFrameVerification`
- **New feature?** → Create new test class

### Step 2: Create Test Method
```python
def test_descriptive_name(self, daemon):
    _step("What this test validates")
    
    # Arrange: Create test data
    reference = np.random.randn(512).astype(np.float32)
    reference = reference / np.linalg.norm(reference)
    
    # Act: Call the method being tested
    result = daemon.check_identity(generated, reference)
    
    # Assert: Verify expected behavior
    assert result >= daemon.identity_threshold
    print(f"  Result: {result:.6f}")
```

### Step 3: Run the New Test
```bash
pytest tests/core/test_verification_daemon.py::YourTestClass::test_descriptive_name -v -s
```

---

## Understanding Test Output

### Verbose Output with Step Messages
```
[Verification Daemon] Creating synthetic reference image for tests
[Verification Daemon] Testing identity check with perfect match
  Score: 1.000000
PASSED
```

### Failure Message
```
AssertionError: assert 0.85 >= 0.90
  Score: 0.85
FAILED
```

### Coverage Report
```
Name                           Stmts   Miss  Cover
--------------------------------------------------
core/verification_daemon.py    150     12    92%
tests/core/test_verification  400      0   100%
```

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'pytest'`
```bash
pip install pytest
```

### Issue: Test runs but shows no output
```bash
# Add -s flag to show print statements
pytest tests/core/test_verification_daemon.py -v -s
```

### Issue: One test fails randomly
→ Check for hardcoded random values; use fixed seeds:
```python
np.random.seed(42)
# Now random values are reproducible
```

### Issue: Test imports fail
```bash
# Ensure pivot/ is in Python path
cd /path/to/pivot
pytest tests/core/test_verification_daemon.py -v
```

---

## Key Concepts

| Concept | Definition | Threshold |
|---------|-----------|-----------|
| **Identity Score** | Cosine similarity of embeddings | ≥ 0.90 |
| **L_physics** | Composite biomechanical metric | ≤ 0.01 |
| **Hard Gate** | Both checks must pass (AND logic) | Pass/Fail |
| **Retry Depth** | Maximum correction attempts | 5 |
| **Weight Multiplier** | Constraint emphasis during regeneration | 1.5x |
| **Confidence** | Keypoint detection reliability | [0.0, 1.0] |

---

## Resources

- **Test File**: `tests/core/test_verification_daemon.py`
- **Documentation**: `tests/core/VERIFICATION_DAEMON_TESTS.md`
- **ADR-003**: `docs/architecture/ADR-003-verification-daemon.md`
- **Phase 1 PRD**: `phase1_prd.md`
- **Architecture**: `architecture.md`

---

## Summary

✅ **27 comprehensive unit tests**
✅ **~2 second execution time**
✅ **Full coverage of VD-REQ-001 through VD-REQ-007**
✅ **Integration-ready for diffusion model**
✅ **Clear documentation and step messages**

Ready to extend with integration tests and diffusion model validation.
