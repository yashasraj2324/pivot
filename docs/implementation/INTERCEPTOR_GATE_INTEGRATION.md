# Interceptor Gate Integration Guide
## Diffusion Model Pipeline Integration

### Overview

This guide explains how to integrate the Interceptor Gate into your diffusion model pipeline. The gate acts as a post-processing constraint layer that enforces identity and kinematic verification on generated frames.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Diffusion Model                      │
│                   Denoising Loop (T→0)                  │
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │ For t in range(num_timesteps, 0, -1):            │ │
│  │   1. Extract noise prediction                     │ │
│  │   2. Update latent: z_t-1 = denoise(z_t)         │ │
│  │   3. Store latent history: history[t-1] = z_t-1  │ │
│  │   4. Advance to next timestep                     │ │
│  └────────────────────────────────────────────────────┘ │
│                      │                                   │
│                      │ z (frame latent)                 │
│                      ▼                                   │
│          ┌─────────────────────────┐                    │
│          │  Decode to Pixel Space  │                    │
│          │  (VAE Decoder)          │                    │
│          └──────────────┬──────────┘                    │
└─────────────────────────┼────────────────────────────────┘
                          │ Frame (PIL Image)
                          ▼
        ┌──────────────────────────────────────┐
        │   Embedding & Keypoint Extraction    │
        │                                      │
        │  • ArcFace Embedding (512-d)        │
        │  • DWPose Keypoints (COCO 17-point) │
        └────────────┬─────────────────────────┘
                     │ Embeddings + Keypoints
                     ▼
        ┌──────────────────────────────────────────┐
        │    ⭐ INTERCEPTOR GATE ⭐               │
        │   (Verification Daemon Hard Gate)       │
        │                                         │
        │  1. Identity Check (≥0.90)              │
        │     CosineSimilarityGate                │
        │                                         │
        │  2. Kinematics Check (L_physics≤0.01)  │
        │     KinematicGuardrail                 │
        │                                         │
        │  3. Correction Loop (if needed)         │
        │     • Latent Rewind (t-1)               │
        │     • Localized Inpainting              │
        │     • Constrained Regeneration (1.5x)  │
        │     • Re-verify (max 5 attempts)        │
        │                                         │
        │  Result: Verified latent or best candidate
        └────────┬─────────────────────────────────┘
                 │ verified_latent
                 ▼
        ┌──────────────────────────────────────┐
        │     Decode & Post-Processing         │
        │                                      │
        │  • VAE Decode to Pixel Space         │
        │  • Image Normalization               │
        │  • Format Conversion                 │
        └────────────┬─────────────────────────┘
                     │ Output Image
                     ▼
        ┌──────────────────────────────────────┐
        │         Output Buffer                │
        │  (Verified & Quality-Assured Frames) │
        └──────────────────────────────────────┘
```

---

## Step-by-Step Integration

### Step 1: Prepare Diffusion Model

Your diffusion model needs to support:
1. Latent history tracking
2. Inpainting capability
3. Guided regeneration with weight multipliers

```python
class DiffusionModel:
    def __init__(self, ...):
        self.latent_history = []  # Store all latent states
        self.guidance_scale = 7.5  # Base guidance scale
    
    def denoise_loop(self, latent, timesteps):
        """Main denoising loop with history tracking."""
        for t in timesteps:
            noise_pred = self.predict_noise(latent, t)
            latent = self.update_latent(latent, noise_pred, t)
            
            # 🔑 CRITICAL: Store latent history
            self.latent_history.append(latent.clone())
        
        return latent
    
    def rewind_latent(self, steps=1):
        """Return latent from N timesteps ago."""
        if len(self.latent_history) < steps + 1:
            return None
        return self.latent_history[-(steps + 1)].clone()
    
    def apply_inpainting(self, latent, mask):
        """Apply binary mask to latent space."""
        # Downsample mask from (512, 512) to (64, 64)
        mask_down = torch.nn.functional.interpolate(
            torch.from_numpy(mask[None, None]).float().to(latent.device),
            size=(latent.shape[-2:]),
            mode='nearest'
        )
        
        # Zero out masked regions
        latent = latent * (1 - mask_down)
        return latent
    
    def constrained_regenerate(self, latent, weight_multiplier=1.5):
        """Regenerate with amplified constraint."""
        # Continue denoising for a few steps
        for step in range(5):  # 5 additional refinement steps
            noise_pred = self.predict_noise(latent, t=999-step)  # Near end of schedule
            latent = self.update_latent(
                latent,
                noise_pred,
                t=999-step,
                guidance_scale=self.guidance_scale * weight_multiplier  # Amplify!
            )
        
        return latent
```

### Step 2: Extract Embeddings & Keypoints

```python
import torch
from torchvision.transforms import Normalize
from facenet_pytorch import InceptionResnetV1
import cv2
from dwpose import DWPose

class EmbeddingExtractor:
    def __init__(self):
        # ArcFace embedding model (512-d)
        self.face_model = InceptionResnetV1(
            pretrained='vggface2',
            device='cuda'
        ).eval()
        
        # Pose detection model
        self.pose_model = DWPose()
        
        self.normalize = Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    
    def extract_face_embedding(self, image):
        """
        Extract ArcFace embedding from image.
        
        Args:
            image: PIL Image or np.ndarray (3, H, W) or (H, W, 3)
        
        Returns:
            np.ndarray: 512-d normalized embedding
        """
        # Convert to tensor
        if isinstance(image, np.ndarray):
            if image.shape[0] != 3:
                image = np.transpose(image, (2, 0, 1))
            tensor = torch.from_numpy(image).float() / 255.0
        else:  # PIL Image
            tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        # Normalize
        tensor = self.normalize(tensor).unsqueeze(0).cuda()
        
        # Extract embedding
        with torch.no_grad():
            embedding = self.face_model(tensor).cpu().numpy()[0]
        
        # Normalize to unit norm
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding  # Shape: (512,)
    
    def extract_pose_keypoints(self, image):
        """
        Extract COCO 17-point pose keypoints.
        
        Args:
            image: PIL Image or np.ndarray (H, W, 3)
        
        Returns:
            np.ndarray: (17, 3) array [x, y, confidence] for each joint
                       or None if detection fails
        """
        # Convert to numpy if needed
        if isinstance(image, np.ndarray):
            img_np = image if image.dtype == np.uint8 else (image * 255).astype(np.uint8)
        else:
            img_np = np.array(image)
        
        try:
            # Detect pose
            keypoints = self.pose_model(img_np)  # Returns (17, 3)
            return keypoints
        except Exception as e:
            print(f"Pose detection failed: {e}")
            return None
```

### Step 3: Initialize Gate & Callbacks

```python
from core.interceptor_gate import InterceptorGate
from core.verification_daemon import VerificationDaemon

class VerifiedDiffusion:
    def __init__(self, diffusion_model):
        self.diffusion = diffusion_model
        self.extractor = EmbeddingExtractor()
        
        # Initialize verification daemon
        self.daemon = VerificationDaemon(
            enable_kinematic=True,
            identity_threshold=0.90,
            kinematic_threshold=0.01,
        )
        
        # Initialize gate with callbacks
        self.gate = InterceptorGate(
            verification_daemon=self.daemon,
            max_retries=5,
            latent_rewind_fn=self._latent_rewind,
            inpaint_fn=self._apply_inpaint,
            regenerate_fn=self._regenerate,
        )
    
    def _latent_rewind(self, steps=1):
        """Callback: Rewind latent to previous timestep."""
        return self.diffusion.rewind_latent(steps=steps)
    
    def _apply_inpaint(self, latent, mask):
        """Callback: Apply inpainting mask."""
        return self.diffusion.apply_inpainting(latent, mask)
    
    def _regenerate(self, latent, weight_multiplier=1.5):
        """Callback: Regenerate with constraint weight."""
        return self.diffusion.constrained_regenerate(latent, weight_multiplier)
```

### Step 4: Generate with Verification

```python
class VerifiedDiffusion(... ):
    def generate(
        self,
        reference_image,
        prompt,
        num_frames=30,
        num_timesteps=50,
    ):
        """
        Generate video frames with identity & kinematic verification.
        
        Args:
            reference_image: PIL Image or path - reference face/body
            prompt: Text prompt for generation
            num_frames: Number of frames to generate
            num_timesteps: Denoising steps per frame
        
        Returns:
            list: Verified frame latents
        """
        
        # Extract reference embedding once
        ref_image = Image.open(reference_image) if isinstance(reference_image, str) else reference_image
        reference_embedding = self.extractor.extract_face_embedding(ref_image)
        
        print(f"Reference embedding shape: {reference_embedding.shape}")
        print(f"Reference embedding norm: {np.linalg.norm(reference_embedding):.4f}")
        
        verified_latents = []
        
        for frame_idx in range(num_frames):
            print(f"\n{'='*60}")
            print(f"Frame {frame_idx + 1}/{num_frames}")
            print(f"{'='*60}")
            
            # 1. Generate latent via denoising
            latent = torch.randn(1, 4, 64, 64).cuda()  # Start with noise
            latent = self.diffusion.denoise_loop(
                latent,
                timesteps=np.linspace(999, 0, num_timesteps, dtype=int),
                prompt=prompt,
            )
            
            # 2. Decode to pixel space
            image = self.diffusion.decode(latent)  # PIL Image
            
            # 3. Extract embeddings & keypoints
            generated_embedding = self.extractor.extract_face_embedding(image)
            pose_keypoints = self.extractor.extract_pose_keypoints(image)
            
            # 4. Process through interceptor gate
            result = self.gate.process_frame(
                frame_idx=frame_idx,
                latent=latent.cpu().numpy(),
                reference_embedding=reference_embedding,
                generated_embedding=generated_embedding,
                pose_keypoints=pose_keypoints,
            )
            
            # 5. Log result
            self._log_result(frame_idx, result)
            
            # 6. Store verified latent
            verified_latents.append(result.latent_after)
        
        # 7. Print summary statistics
        stats = self.gate.get_frame_statistics()
        self._print_statistics(stats)
        
        return verified_latents
    
    def _log_result(self, frame_idx, result):
        """Log verification result."""
        print(f"Decision: {result.decision.value}")
        print(f"Passed: {result.passed}")
        print(f"Identity Score: {result.identity_score:.4f}")
        
        if result.kinematic_loss is not None:
            print(f"Kinematic Loss: {result.kinematic_loss:.6f}")
        
        if result.retry_count > 0:
            print(f"Correction Attempts: {result.retry_count}")
            for action in result.correction_actions:
                print(f"  - {action.action_type}: {action.notes}")
        
        if not result.passed:
            print(f"Error: {result.error_message}")
    
    def _print_statistics(self, stats):
        """Print aggregate statistics."""
        print(f"\n{'='*60}")
        print("VERIFICATION STATISTICS")
        print(f"{'='*60}")
        print(f"Total Frames: {stats['total_frames']}")
        print(f"Passed: {stats['passed_frames']} ({stats['pass_rate']:.1%})")
        print(f"Failed: {stats['failed_frames']}")
        print(f"  - Identity Failures: {stats['identity_failures']}")
        print(f"  - Kinematics Failures: {stats['kinematic_failures']}")
        print(f"Total Corrections: {stats['total_corrections']}")
        print(f"Avg Corrections/Frame: {stats['avg_corrections_per_frame']:.2f}")
```

### Step 5: Complete Example

```python
from PIL import Image

# Initialize model
diffusion = DiffusionModel(model_id="model-name", device="cuda")
verified_gen = VerifiedDiffusion(diffusion)

# Run generation
reference = Image.open("reference_face.jpg")
prompt = "a person walking in the park"

verified_latents = verified_gen.generate(
    reference_image=reference,
    prompt=prompt,
    num_frames=30,
    num_timesteps=50,
)

# Decode and save results
for i, latent in enumerate(verified_latents):
    image = diffusion.decode(torch.from_numpy(latent))
    image.save(f"output/frame_{i:04d}.png")

print("✓ Generation complete - all frames verified!")
```

---

## Advanced Configuration

### Custom Threshold Tuning

```python
# Adjust thresholds based on your needs
if quality_mode == "strict":
    gate.daemon.identity_gate.threshold = 0.95  # Stricter
    gate.daemon.kinematic_threshold = 0.005     # Stricter
    gate.max_retries = 10                       # More attempts
elif quality_mode == "balanced":
    gate.daemon.identity_gate.threshold = 0.90  # Default
    gate.daemon.kinematic_threshold = 0.01      # Default
    gate.max_retries = 5                        # Default
elif quality_mode == "fast":
    gate.daemon.identity_gate.threshold = 0.85  # Lenient
    gate.daemon.enable_kinematic = False        # Skip kinematics
    gate.max_retries = 2                        # Fewer attempts
```

### Conditional Kinematics Checking

```python
# Only verify kinematics for frames where pose detection is confident
def generate_with_selective_kinematics(self, ...):
    for frame_idx in range(num_frames):
        ...
        
        pose_keypoints = self.extractor.extract_pose_keypoints(image)
        
        # Skip kinematics if confidence is low
        if pose_keypoints is None or np.mean(pose_keypoints[:, 2]) < 0.5:
            pose_keypoints = None  # Skip kinematics check
        
        result = self.gate.process_frame(
            ...
            pose_keypoints=pose_keypoints,
        )
```

### Batch Processing

```python
def generate_batch(self, reference, prompts, num_frames=30):
    """Generate multiple sequences efficiently."""
    
    all_results = []
    
    for prompt_idx, prompt in enumerate(prompts):
        print(f"\nGenerating sequence {prompt_idx + 1}/{len(prompts)}")
        
        # Reset gate for new sequence
        self.gate.reset()
        
        # Generate frames
        latents = self.generate(reference, prompt, num_frames)
        
        # Store results
        all_results.append({
            'prompt': prompt,
            'latents': latents,
            'statistics': self.gate.get_frame_statistics(),
        })
    
    return all_results
```

### Error Recovery

```python
def safe_generate(self, reference, prompt, num_frames=30):
    """Generate with error handling."""
    
    verified_latents = []
    failed_frames = []
    
    try:
        for frame_idx in range(num_frames):
            try:
                latent = self.diffusion.denoise_loop(...)
                image = self.diffusion.decode(latent)
                
                result = self.gate.process_frame(...)
                verified_latents.append(result.latent_after)
                
            except Exception as e:
                print(f"Frame {frame_idx} failed: {e}")
                failed_frames.append(frame_idx)
                
                # Use fallback: previous latent or random
                if verified_latents:
                    verified_latents.append(verified_latents[-1])
                else:
                    verified_latents.append(torch.randn(1, 4, 64, 64))
    
    finally:
        stats = self.gate.get_frame_statistics()
        print(f"\nGeneration Summary:")
        print(f"  Successful: {len(verified_latents) - len(failed_frames)}")
        print(f"  Failed: {len(failed_frames)}")
        if failed_frames:
            print(f"  Failed frame indices: {failed_frames}")
    
    return verified_latents
```

---

## Performance Optimization

### Memory Optimization

```python
# Clear latent history periodically to save memory
if frame_idx % 10 == 0:
    self.diffusion.latent_history = self.diffusion.latent_history[-5:]  # Keep only last 5
```

### Speed Optimization

```python
# Parallel embedding extraction
from concurrent.futures import ThreadPoolExecutor

def generate_fast(self, reference, prompt, num_frames=30):
    """Generate with parallel embedding extraction."""
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        for frame_idx in range(num_frames):
            # Start denoising
            latent_future = executor.submit(
                self.diffusion.denoise_loop, ...
            )
            
            # While denoising, process previous frame
            if frame_idx > 0:
                result = self.gate.process_frame(
                    previous_latent,
                    previous_embedding,
                )
            
            # Get latest result
            latent = latent_future.result()
            image = self.diffusion.decode(latent)
```

---

## Monitoring & Logging

```python
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('generation.log'),
        logging.StreamHandler(),
    ]
)

logger = logging.getLogger(__name__)

class LoggedVerification(VerifiedDiffusion):
    def generate(self, reference, prompt, num_frames=30):
        """Generate with comprehensive logging."""
        
        logger.info(f"Starting generation: {num_frames} frames, prompt: {prompt}")
        start_time = datetime.now()
        
        try:
            results = super().generate(reference, prompt, num_frames)
            
            # Log statistics
            stats = self.gate.get_frame_statistics()
            logger.info(f"Generation complete: {stats['pass_rate']:.1%} pass rate")
            logger.info(f"Total time: {(datetime.now() - start_time).total_seconds():.1f}s")
            
            # Save detailed report
            with open(f"report_{datetime.now().isoformat()}.json", "w") as f:
                json.dump(stats, f, indent=2)
            
            return results
        
        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            raise
```

---

## Testing Integration

```bash
# Test the full pipeline
pytest tests/core/test_interceptor_gate.py -v

# Test integration with mock diffusion
pytest tests/integration/test_verified_diffusion.py -v

# Profile performance
python -m cProfile -s cumulative test_generation.py > profile.txt
```

---

## References

- [Interceptor Gate Implementation](INTERCEPTOR_GATE_IMPLEMENTATION.md)
- [Quick Reference](INTERCEPTOR_GATE_QUICK_REFERENCE.md)
- [Verification Daemon](../../core/verification_daemon.py)
- [ADR-003](../architecture/ADR-003-verification-daemon.md)
