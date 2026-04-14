|**P.I.V.O.T.  —  Product Requirements Document v2.0**|**CONFIDENTIAL**|
| :- | -: |
|<p>**P.I.V.O.T.**</p><p>*Physics-Informed Video Optimization & Tracking*</p><p>————————————————————</p><p>**PRODUCT REQUIREMENTS DOCUMENT**</p><p>Version 2.0  |  Status: Active Development  |  Classification: Confidential</p><p>**Full Scene Simulation Engine  —  Character + Environment Physics**</p>|
| :-: |

|**Version**|**Date**|**Summary**|**Author**|
| :- | :- | :- | :- |
|1\.0|Initial Release|Character physics, identity lock, emotion stack|Core Team|
|2\.0|Current|**Added Environmental Physics Module (Rigid Body + Fluid Dynamics); P.I.V.O.T. becomes Full Scene Simulation Engine**|Core Team|


# **1. Executive Summary**
P.I.V.O.T. is a synthetic human performance engine that has evolved into a Full Scene Simulation Engine. It generates physically accurate, emotionally coherent, identity-stable video of one or more real people from a single reference photo per character and a text prompt — and now extends that same physical rigor to every object, vehicle, and environmental force in the scene.

**Version 1.0 solved the Shapeshifter Effect for human characters. Version 2.0 solves it for the entire scene.**

Current video diffusion models suffer systematic failure across two domains: characters morph and violate biomechanics, and environments shapeshift — cars stretch and slide without wheel rotation, smoke solidifies into plastic, explosions teleport instead of expand. P.I.V.O.T. v2.0 eliminates both failure classes through a unified physics enforcement architecture covering biomechanics, rigid body dynamics, and fluid dynamics.

|<p>**CORE PROMISE**</p><p>Give P.I.V.O.T. a photo of a person and a prompt. Receive a video where that person does exactly what the prompt describes — physically believable, emotionally coherent, identity-stable — inside an environment that obeys Newtonian mechanics and fluid physics. No matter how complex the scene.</p>|
| :- |

# **2. Problem Statement**
## **2.1 The Shapeshifter Effect — Human Domain (V1 Problem)**
Every current video diffusion model suffers from the Shapeshifter Effect across characters:

- **Identity Drift:** Identity Drift: faces gradually change across frames until the person at frame 1 is unrecognizable by frame 60.
- **Topological Morphing:** Topological Morphing: skulls, joints, and torsos change shape mid-sequence. Arms stretch. Hands grow extra fingers.
- **Kinematic Violations:** Kinematic Violations: joints hyperextend beyond physiological limits. Characters teleport between positions.
- **Emotional Incoherence:** Emotional Incoherence: expressions appear and vanish in single frames. Body language contradicts facial expression.
- **Multi-Character Bleeding:** Multi-Character Bleeding: facial features and identity attributes bleed between characters in the same scene.
## **2.2 The Shapeshifter Effect — Environmental Domain (V2 Problem)**
The same failure class that destroys character consistency also destroys environmental plausibility. Diffusion models have no understanding of rigid body mechanics or fluid dynamics:

- **Vehicle Morphing:** Vehicle Morphing: cars stretch, change model mid-shot, or slide sideways without wheel rotation. A sedan becomes an SUV between frames.
- **Impossible Inertia:** Impossible Inertia: vehicles stop instantaneously, accelerate without physical cause, or move perpendicular to their momentum vector.
- **Smoke Solidification:** Smoke Solidification: smoke and dust turn into opaque plastic-like masses. Fire flickers randomly rather than expanding from a heat source.
- **Blast Incoherence:** Blast Incoherence: explosion radii shrink instead of expanding. Debris teleports rather than following ballistic trajectories.
- **Environmental Amnesia:** Environmental Amnesia: rain falls sideways, wind-affected cloth behaves as rigid, shadows are inconsistent with light source.
## **2.3 Why Existing Solutions Fail**
Existing mitigation approaches treat these as generation quality problems rather than enforcement architecture problems:

- ControlNet alone constrains pose but cannot enforce environmental physics or fluid dynamics.
- IP-Adapters inject identity but have no mechanism for rigid body or smoke simulation.
- Manual cleanup is not scalable — VFX artists and hours of post-production per minute of video cannot be automated away.
- Retraining on larger datasets shifts the distribution but never eliminates the failure modes.

|<p>**CORE INSIGHT**</p><p>These are not generation quality problems. They are enforcement architecture problems. The solution is a verification and correction layer that treats physical and environmental laws as hard constraints — not soft preferences baked into model weights.</p>|
| :- |

# **3. Vision & Product Goals**
## **3.1 Vision Statement**

|***A synthetic human performance engine that generates cinematically consistent, physically uncompromising, emotionally coherent video of any number of real people — from a photo and a prompt — inside a fully simulated environment where every vehicle, fluid, and force obeys physics. Fully automated, at scale, with no manual cleanup required.***|
| :-: |

## **3.2 Product Goals**
- **Physics First — Character:** Physics First — Character: Every output frame satisfies all kinematic constraints for all characters simultaneously. Non-negotiable.
- **Physics First — Environment:** Physics First — Environment: Every output frame satisfies rigid body dynamics for all vehicles/objects and fluid dynamics for all volumetric elements. Non-negotiable.
- **Identity Permanence:** Identity Permanence: Every character maintains cosine similarity >= 0.90 to their reference embedding throughout the entire sequence.
- **Emotional Realism:** Emotional Realism: Expressions are anatomically grounded via FACS Action Units. Transitions are physiologically plausible.
- **Multi-Character Support:** Multi-Character Support: N characters tracked, constrained, and verified independently and in relation to each other.
- **Full Automation:** Full Automation: Zero manual QA. The system detects, corrects, and validates all violations before output reaches the user.
- **Platform Distribution:** Platform Distribution: Accessible via REST API, Python package, and ComfyUI node from a single engine.

# **4. Target Users & Use Cases**

|**User Segment**|**Primary Use Case**|**Access Surface**|
| :- | :- | :- |
|AI/ML Developers|Embed video generation into products, apps, and platforms|REST API|
|ML Researchers|Use physics enforcement as standalone infrastructure on any pipeline|Python Package|
|VFX & Filmmakers|Generate character + environment performances inside creative workflows|ComfyUI Node|
|Game Studios|Produce consistent NPC animations, cutscenes, and in-engine cinematics|REST API / Python|
|Advertising & Brand|Generate brand ambassadors in complex scenarios from one photo|REST API|
|Action / Film VFX|Generate physically accurate vehicle and explosion sequences|ComfyUI / API|
|Avatar Platforms|Power persistent, physically believable digital human systems|REST API|
|Internal Team|Production validation, demos, and R&D|Internal Tool|

# **5. Core Realism Stack**
P.I.V.O.T.’s realism model is layered. Every layer is enforced as a hard contract. No layer can be disabled. Speed is sacrificed before any realism constraint is compromised.

|<p>**V2 EXPANSION NOTE**</p><p>Layers 1–7 cover character physics, identity, emotion, and scene coherence (unchanged from v1.0). Layer 8 (Rigid Body Dynamics) and Layer 9 (Fluid Dynamics) are new in v2.0 and extend P.I.V.O.T. from a Character Consistency Tool into a Full Scene Simulation Engine.</p>|
| :- |

## **5.1 Layer 1 — Structural Physics (Non-Negotiable)**
The foundational enforcement layer for human characters. All kinematic constraints are evaluated per frame, per character, before output advances.

- **Bone Length Invariance:** Bone Length Invariance: Euclidean distance between all connected joint pairs remains constant across time, normalized for camera depth. L\_bone = Σ\_t || d(j\_a, j\_b)\_t − d(j\_a, j\_b)\_{t-1} ||^2 must equal zero.
- **Biomechanical Range of Motion:** Biomechanical Range of Motion: All joint angles clamped to human physiological limits. Shoulder: 0–180°. Elbow: 0–145°. Knee: 0–135°. No exceptions.
- **Temporal Velocity Limits:** Temporal Velocity Limits: First derivative of all joint positions is bounded. || (p\_t − p\_{t-1}) / Δt || ≤ v\_max. Teleportation is architecturally impossible.
- **Rigid Body Topology:** Rigid Body Topology: Skull, ribcage, pelvis maintain fixed geometry. No morphing, no stretching.
- **Cross-Character Contact Physics:** Cross-Character Contact Physics: Contact points enforce consistent force direction and skin deformation for both characters simultaneously.
## **5.2 Layer 2 — Identity Lock**
- **ArcFace Embedding:** ArcFace Embedding: 512-dimensional facial embedding extracted from the reference photo via InsightFace. Ground truth for all identity verification.
- **Cosine Similarity Gate:** Cosine Similarity Gate: Every frame evaluated against reference embedding. Cosine similarity < 0.90 triggers immediate halt and correction.
- **IP-Adapter Conditioning:** IP-Adapter Conditioning: Identity vector injected into U-Net cross-attention layers alongside text prompt.
- **Instance-Level Masked Attention:** Instance-Level Masked Attention: Pixel-precise segmentation masks localize each character’s identity vector. No identity bleed between characters at any proximity.
## **5.3 Layer 3 — Facial & Emotional Realism**
- **Action Unit Enforcement:** Action Unit Enforcement: Every emotion maps to specific FACS Action Units. A smile requires AU6 + AU12 simultaneously — geometry movement, not texture change.
- **Emotion Temporal Ramp:** Emotion Temporal Ramp: Expressions cannot appear or vanish in a single frame. Transitions require physiologically plausible onset, peak, and decay.
- **Scene-Prompt Coherence:** Scene-Prompt Coherence: Emotion in each character’s face and body must match the semantic content of the scene prompt at all times.
- **Micro-Expression Layer:** Micro-Expression Layer: Subtle involuntary expressions (1/25 to 1/5 second) generated preceding and following primary expressions.
## **5.4 Layer 4 — Body Language Coherence**
- **Postural Congruence:** Postural Congruence: Fear raises shoulders and contracts chest. Confidence opens chest and widens stance. All postural states indexed to emotional states.
- **Gestural Intent:** Gestural Intent: Hand and arm gestures are purposeful and directional. Characters gesture toward what they are discussing.
- **Reactive Body Language:** Reactive Body Language: In multi-character scenes, Character B’s posture and movement react physically and emotionally to Character A’s actions.
## **5.5 Layer 5 — Ocular Dynamics**
- **Gaze Direction:** Gaze Direction: Characters look at what the scene dictates. Gaze is never random.
- **Physiological Blink Rate:** Physiological Blink Rate: Baseline 15–20 blinks/min, modulated by emotional state. Follows involuntary human patterns, not a timer.
- **Pupil Response:** Pupil Response: Pupils dilate in low light and emotional arousal. Constrict in bright light and calm states.
- **Saccadic Movement:** Saccadic Movement: Rapid eye movements between fixation points follow real human oculomotor patterns — ballistic transitions with fixation holds.
## **5.6 Layer 6 — Secondary Motion & Micro-Physics**
- **Hair Dynamics:** Hair Dynamics: Hair responds to head movement, acceleration, and environmental conditions. Secondary motion continues after primary movement stops.
- **Clothing Physics:** Clothing Physics: Fabric deforms under movement, drapes according to gravity, creases at joint flexion points. No clipping.
- **Skin Deformation:** Skin Deformation: Skin compresses at contact points. A hand gripping a shoulder produces compression, not geometric intersection.
- **Breathing Visibility:** Breathing Visibility: Chest and shoulder movement reflects breathing. Rate and depth consistent with exertion level.
## **5.7 Layer 7 — Scene-Level Coherence**
- **Lighting Consistency:** Lighting Consistency: Light on all characters is consistent with the scene’s light sources.
- **Shadow Accuracy:** Shadow Accuracy: All N characters cast and receive shadows consistent with scene lighting direction and intensity.
- **Environmental Interaction:** Environmental Interaction: If the scene contains rain, wind, or other environmental conditions, all characters respond physically and consistently.



|<p>**NEW IN V2.0**</p><p>**Environmental Physics Module**</p><p>*Layers 8 & 9: Rigid Body Dynamics + Fluid Dynamics*</p>|
| :-: |

The Environmental Physics Module extends P.I.V.O.T.’s enforcement architecture beyond the human skeleton to the full scene. This requires a fundamental expansion of the mathematical constraints applied in the verification loop: from Biomechanics/Kinematics to include Rigid Body Dynamics and Fluid Dynamics.

This transition moves P.I.V.O.T. from a Character Consistency Tool into a Full Scene Simulation Engine.

## **5.8 Layer 8 — Rigid Body Dynamics (Vehicles, Machinery, Solid Debris)**

|<p>**PHYSICS DOMAIN**</p><p>Unlike humans, rigid objects have no joints — they have mass, momentum, and rigid 3D geometry. The Shapeshifter Effect in vehicles manifests as stretching, model-switching, or sideways sliding without wheel rotation. Newtonian mechanics are the enforcement basis.</p>|
| :- |

### **Extraction & Conditioning**
OpenPose cannot be used for rigid objects. The extraction stack is replaced with:

- **DepthAnything / ZoeDepth:** DepthAnything / ZoeDepth: Establishes spatial topology and z-depth for all rigid objects in the scene.
- **3D Bounding Box Tracking:** 3D Bounding Box Tracking: Establishes a rigid coordinate system for each vehicle or solid object, enabling volume preservation checks across frames.
### **Physics-Informed Loss — L\_rigid**
Newtonian constraints enforced per object, per frame:

- **Volume Preservation:** Volume Preservation: The volumetric bounding box V of any rigid object must remain constant across time t. A sedan cannot become an SUV.
- **Centroid Tracking:** Centroid Tracking: The centroid position x of each rigid object is tracked. Velocity and acceleration are derived from centroid displacement.

**v\_t = (x\_t - x\_{t-1}) / Δt     |     a\_t = (v\_t - v\_{t-1}) / Δt     |     j\_t = (a\_t - a\_{t-1}) / Δt**

- **Inertial Constraints:** Inertial Constraints: Acceleration a\_t and jerk j\_t (the derivative of acceleration) are bounded to prevent physically impossible movements such as instantaneous stopping or perpendicular direction changes.
- **Grip/Friction Limits:** Grip/Friction Limits: If a\_t exceeds the maximum possible grip or friction limits of the object class (vehicle on road, debris in air), the verification daemon triggers a rewind.
### **Fail Criteria for Rigid Body Verification**
- Bounding box volume V changes between consecutive frames beyond tolerance.
- Computed acceleration a\_t exceeds the physical maximum for the object class.
- Depth map indicates object topology has morphed (structural shape change detected).
- Wheel rotation is absent when vehicle velocity vector is non-zero.

## **5.9 Layer 9 — Fluid Dynamics & Volumetrics (Smoke, Dust, Blasts, Fire)**

|<p>**HARDEST PROBLEM IN AI VIDEO**</p><p>Smoke, fire, and volumetric blasts are amorphous — no skeleton, no fixed volume, no bounding box. Diffusion models notoriously solidify smoke into plastic or cause random flickering. Navier-Stokes approximations and optical flow consistency are the enforcement basis.</p>|
| :- |

### **Extraction & Conditioning**
Point tracking and bounding boxes are inapplicable for fluids. The extraction stack uses:

- **Optical Flow — RAFT:** Optical Flow — RAFT (Recurrent All-Pairs Field Transforms): Instead of tracking discrete points, RAFT tracks dense vector fields showing where every pixel of smoke or fire is moving between frames.
- **High-Luminance Pixel Segmentation:** High-Luminance Pixel Segmentation: Fire and blast cores are segmented by luminance threshold, establishing the expanding radius that all subsequent frames must respect.
### **Physics-Informed Loss — L\_fluid**
Navier-Stokes approximations for incompressible fluid flow (full CFD is computationally infeasible inside a diffusion loop; the system enforces a Warping / Temporal Coherence Loss instead):

- **Continuity Constraint:** Continuity Constraint: The flow field u must approximate mass conservation. For incompressible smoke, the divergence of the velocity field must be near zero:

**∇ · u = 0**

- **Optical Flow Consistency:** Optical Flow Consistency: Generated frame I\_t must mathematically match the previous frame I\_{t-1} warped by the predicted flow field F\_{t→t-1}:

**L\_flow = Σ || I\_t - warp(I\_{t-1}, F\_{t→t-1}) ||^2**

- **Expansion Rate Enforcement:** Expansion Rate Enforcement: For blast and explosion sequences, the high-luminance pixel radius must monotonically increase during the expansion phase. Radius shrinkage in the first 10 frames triggers auto-fix inpainting.
### **Fail Criteria for Fluid Verification**
- Smoke/fire pixels in frame t cannot be traced back to the expanding blast radius in frame t-1 (indicating teleportation or solidification).
- Divergence of optical flow field exceeds tolerance (mass conservation violated).
- Explosion radius shrinks rather than expanding during the established expansion window.
- High-frequency flickering detected in luminance field without a corresponding thermal source change.

# **6. Architectural Implications of Environmental Physics**
Adding environmental physics to the P.I.V.O.T. pipeline introduces three systemic engineering challenges that require explicit solutions before deployment.

## **6.1 Multi-ControlNet Routing — VRAM Overhead**
A scene containing a human running from an exploding vehicle requires the U-Net to simultaneously process three ControlNet conditioning streams:

- OpenPose — for the human character skeleton
- Depth Map — for the vehicle’s rigid body coordinate system
- Optical Flow — for the blast/smoke volumetric field

Stacking multiple ControlNets causes near-exponential VRAM consumption. Each ControlNet adds a full copy of the U-Net encoder to the inference graph.

|<p>**ENGINEERING SOLVE**</p><p>Aggressive gradient checkpointing must be implemented. Inactive ControlNet tensors are dynamically offloaded to CPU RAM between denoising steps and reloaded only when their conditioning layer is active. This trades latency for VRAM headroom on 24–40GB cards.</p>|
| :- |
## **6.2 Verification Daemon Extensions**
InsightFace only verifies human faces. To verify environmental physics, the background verification daemon must run additional lightweight heuristics in parallel with the existing character checks:

|**Domain**|**Verification Heuristic**|
| :- | :- |
|Vehicles / Rigid Objects|Run Structural Similarity Index (SSIM) on cropped bounding box of the vehicle between consecutive frames. Volume change detection via bounding box comparison.|
|Blast / Explosion|Calculate expansion rate of high-luminance pixels (fire/smoke core). If radius shrinks instead of expands in the first 10 frames, trigger Auto-Fix inpainting.|
|Smoke / Fluid|Verify optical flow divergence does not exceed threshold. Confirm frame-to-frame warp consistency via L\_flow loss.|
|Debris / Projectiles|Track ballistic trajectories via bounding box centroids. Flag deviations from expected parabolic or linear trajectories.|

## **6.3 Base Model Alignment Challenge**
If the enforcement layer forces a diffusion model to obey strict fluid dynamics for smoke, but the base model lacks sufficient training on high-framerate explosion and fluid data, it will “fight” the ControlNet conditioning. The result is blurry, over-smoothed artifacts — physically constrained but visually degraded.

|<p>**ENGINEERING SOLVE**</p><p>Inject high-frequency noise into the masked smoke/fire areas during the inpainting loop to maintain visual crispness while still obeying the optical flow path. This counteracts the base model’s smoothing prior without violating the physics constraints.</p>|
| :- |

# **7. Multi-Character Architecture**
Multi-character support is a first-class feature. The architecture is designed from the ground up to handle N simultaneous characters with independent identity locks, independent kinematic constraints, and shared cross-character physics.

## **7.1 Independent Identity Routing**
- **N Identity Vectors:** N Identity Vectors: One 512-d ArcFace embedding per character reference photo. Independently maintained and verified throughout the sequence.
- **N IP-Adapter Channels:** N IP-Adapter Channels: Each character’s identity vector routed through its own IP-Adapter conditioning channel. Characters never share a channel.
- **Instance Segmentation Masking:** Instance Segmentation Masking: Pixel-precise segmentation masks localize each character’s identity conditioning to their exact silhouette, even in direct physical contact.
## **7.2 Independent Kinematic Lattices**
- **N Pose Tracks:** N Pose Tracks: One ControlNet (DWPose) skeletal track per character. Kinematic constraints evaluated independently per character.
- **Independent L\_physics:** Independent L\_physics: The physics loss function is computed separately for each character. A violation by Character B does not mask a violation by Character A.
- **Parallel Verification:** Parallel Verification: All N kinematic evaluations run in parallel. The pipeline only advances when all N characters pass all constraints simultaneously.
## **7.3 Cross-Character Physics**
- **Contact Event Detection:** Contact Event Detection: System detects when two characters’ skeletal meshes enter contact range and activates the cross-character physics module.
- **Force Direction Consistency:** Force Direction Consistency: Contact interactions enforce Newton’s third law — force vectors are equal, opposite, and physically plausible.
- **Shared Skin Deformation:** Shared Skin Deformation: Skin compression at contact points is computed for both characters simultaneously and must be mutually consistent.

# **8. Closed-Loop Verification Daemon (v2.0)**
The Verification Daemon is P.I.V.O.T.’s central enforcement mechanism. It is not a quality filter — it is a hard gate that sits between generation and output. In v2.0, the daemon covers all character layers plus the new environmental physics layers.

## **8.1 Evaluation Phase**
Each generated frame batch is passed through the full verification stack: identity cosine similarity, L\_physics for all N characters, FACS AU consistency, body language coherence, scene-level coherence, rigid body dynamics (L\_rigid), and fluid dynamics (L\_fluid). All checks run asynchronously in parallel.
## **8.2 Trigger Conditions**

|**Trigger Condition**|**Action**|
| :- | :- |
|Cosine similarity < 0.90 (any character)|Halt. Flag identity failure. Initiate correction.|
|L\_physics > tolerance (any character)|Halt. Flag kinematic violation. Initiate correction.|
|FACS AU inconsistency detected|Halt. Flag emotional incoherence. Initiate correction.|
|Body language contradicts emotion state|Halt. Flag coherence failure. Initiate correction.|
|Cross-character contact physics invalid|Halt. Flag contact violation. Initiate correction.|
|Rigid object volume change detected (L\_rigid)|Halt. Flag rigid body violation. Initiate correction.|
|Vehicle acceleration exceeds physical limit|Halt. Flag inertial violation. Initiate correction.|
|Fluid flow divergence exceeds threshold (L\_fluid)|Halt. Flag fluid physics violation. Initiate correction.|
|Blast radius shrinks in expansion window|Halt. Flag expansion violation. Trigger auto-fix inpainting.|
|Scene lighting inconsistency|Halt. Flag scene failure. Initiate correction.|
|All checks pass (all N characters + environment)|Advance timeline. Write frame to output buffer.|

## **8.3 Correction Phase**
- **Latent Rewind:** Latent Rewind: The daemon rewinds the denoising process to t-1, preserving the last fully valid latent state.
- **Localized Inpainting:** Localized Inpainting: A pixel-precise segmentation mask is generated over the failing region (specific character, joint, facial region, vehicle bounding box, or fluid zone). Only the failing region is regenerated.
- **Constrained Regeneration:** Constrained Regeneration: The regeneration pass is conditioned on all active constraints with increased weight on the violated constraint.
- **Max Retry Depth:** Max Retry Depth: Default 5 retries. If a frame fails after max retries, the daemon logs the failure, selects the highest-scoring candidate, and flags it for review.
- **No User Intervention:** No User Intervention: The entire detect-rewind-correct-verify cycle is fully automated.

# **9. Distribution Surfaces**
P.I.V.O.T. is a single engine exposed through four distribution surfaces, each serving a distinct user segment.

## **9.1 REST API**
- **Endpoint:** Endpoint: POST /v1/generate — accepts reference image array (one per character), prompt, emotion preset, scene physics config, and output config. Returns a job ID.
- **Scene Physics Config:** Scene Physics Config: New parameter in v2.0. Specifies which environmental physics layers are active for the job: rigid\_body (vehicles), fluid\_dynamics (smoke/fire), or both.
- **Async Job Model:** Async Job Model: Video generation is asynchronous. Clients poll GET /v1/jobs/{id} or receive a webhook callback on completion.
- **Rate Limiting:** Rate Limiting: Tier-based. Free: 10 jobs/day, watermarked. Pro: unlimited, clean output. Enterprise: dedicated capacity.
- **Response Format:** Response Format: JSON envelope with job metadata, verification report (per-character + per-object compliance scores), and signed video URL.
## **9.2 Python Package**
- **Install:** Install: pip install pivot-sdk
- **Full Pipeline:** Full Pipeline: pivot.generate(characters=[img1], prompt='...', emotion='neutral', scene\_physics=True) — runs the complete stack including environmental physics.
- **Modular Access:** Modular Access: pivot.environment.rigid\_body.verify(frames), pivot.environment.fluid.verify(frames) — each subsystem callable independently.
- **Custom Constraints:** Custom Constraints: Researchers can inject custom L\_rigid or L\_fluid implementations via pivot.environment.register\_constraint(fn).
- **Standalone Daemon:** Standalone Daemon: pivot.daemon can be used as a drop-in verification layer on any external video pipeline.
## **9.3 ComfyUI Node**
- **Node Inputs:** Node Inputs: Character slots, Prompt, Emotion Preset, Physics Mode (STRICT only), Scene Mode (CHARACTER\_ONLY / FULL\_SCENE), Output Format.
- **Scene Mode:** Scene Mode: New in v2.0. FULL\_SCENE activates the environmental physics module. CHARACTER\_ONLY retains v1.0 behavior.
- **Progress Callbacks:** Progress Callbacks: Node displays per-character and per-object verification status in real time via websocket polling.
- **Install:** Install: ComfyUI Manager > Install Custom Nodes > Search ‘P.I.V.O.T.’

# **10. Infrastructure & Deployment**
The current phase prioritizes zero infrastructure cost and maximum iteration speed. All production infrastructure decisions are deferred until core pipeline validation is complete.

## **10.1 Primary Compute — Google Colab**
- **Role:** Role: Primary interactive development. All subsystem prototyping, L\_physics / L\_rigid / L\_fluid constraint validation, and notebook-based experimentation.
- **Hardware:** Hardware: T4/A100 GPUs via Colab Pro sessions.
- **Persistence:** Persistence: All artifacts written to Google Drive, mounted at runtime for continuity across sessions.
- **Limitation:** Limitation: Multi-ControlNet stacking for full-scene generation may exceed T4 VRAM (16GB). Heavy environmental physics jobs deferred to IndiaAI allocation.
## **10.2 Extended Compute — IndiaAI Mission GPUs**
- **Role:** Role: High-performance compute for full-scene generation — multi-character + vehicle + explosion sequences.
- **Hardware:** Hardware: A100/H100 allocations via IndiaAI Mission researcher portal. Full-precision inference without quantization.
- **Environmental Physics Jobs:** Environmental Physics Jobs: Overnight batch runs. Outputs written to persistent cloud storage for async review.
## **10.3 Model Hub & Demo — Hugging Face**
- **Model Hub:** Model Hub: All diffusion checkpoints, IP-Adapter weights, ControlNet models, RAFT optical flow weights, and P.I.V.O.T.-specific adapters hosted on private HF Hub repositories.
- **Version Pinning:** Version Pinning: All adapter checkpoints tagged by version and referenced by tag in the codebase.
- **HF Spaces:** HF Spaces: Lightweight internal demo deployments on ZeroGPU for stakeholder review.
## **10.4 Source Control — GitHub**
Monorepo structure updated for v2.0:

- /core — inference engine & verification daemon (updated for environmental physics)
- /adapters — IP-Adapter & ControlNet (including Depth and Optical Flow ControlNets)
- /environment — NEW: rigid body dynamics module, fluid dynamics module, RAFT integration
- /emotion — FACS enforcement & AU router
- /notebooks — Colab-ready .ipynb including new environmental physics validation notebooks
- /sdk — Python package (pivot-sdk)
- /comfyui-node — ComfyUI integration (updated for Scene Mode)
- /scripts — evaluation & benchmarking

# **11. Non-Functional Requirements**

|**Requirement**|**Specification**|
| :- | :- |
|Physics Enforcement — Characters|100% of output frames pass all kinematic constraints for all characters. Zero exceptions.|
|Physics Enforcement — Environment|100% of output frames pass all rigid body and fluid dynamics constraints. Zero exceptions.|
|Identity Stability|Cosine similarity >= 0.90 maintained for all characters throughout full sequence length.|
|Emotion Coherence|FACS AU consistency verified per frame. Transitions follow physiological ramp constraints.|
|Multi-Character Scale|Support N >= 2 characters in current phase. Architecture scales to N >= 10.|
|API Response Contract|Job accepted within 2 seconds of POST. Webhook fired within 5 seconds of completion.|
|Correction Max Depth|Verification daemon retries maximum 5 times per frame before selecting best candidate.|
|Package Compatibility|Python >= 3.9. PyTorch >= 2.0. CUDA >= 11.8.|
|ComfyUI Compatibility|ComfyUI >= 0.3.x. Tested on Windows, macOS, and Linux.|
|Security|API keys masked in all UIs. Never logged in full server-side. Scoped per endpoint group.|
|VRAM Budget (Full Scene)|Full-scene generation (character + vehicle + fluid) targets 24GB VRAM with gradient checkpointing. 40GB for unconstrained multi-character full-scene.|

# **12. Known Technical Challenges & Mitigations**
Challenges from v1.0 are retained. V2.0 introduces additional challenges specific to the Environmental Physics Module.

## **12.1 Carried Forward from V1.0**
- **Latency Overhead:** Latency Overhead: Closed-loop verification inherently increases inference time. Mitigation: Parallel async verification batches. Inpainting triggered only on confirmed failures.
- **Pose Occlusion:** Pose Occlusion: Severe self-occlusion confuses IP-Adapter and ControlNet simultaneously. Mitigation: Depth-aware masking via MiDaS/DepthAnything.
- **T4 VRAM Constraints:** T4 VRAM Constraints: 16GB limits batch size. Mitigation: 8-bit quantization and attention slicing on Colab. Full-precision on IndiaAI A100/H100.
- **Multi-Character Instance Segmentation:** Multi-Character Instance Segmentation: Pixel-precise masks required in contact-heavy scenes. Mitigation: SAM2 for robust instance segmentation.
- **FACS Ground Truth:** FACS Ground Truth: Accuracy decreases at extreme angles. Mitigation: Confidence-weighted AU enforcement.
## **12.2 New in V2.0 — Environmental Physics**
- **Multi-ControlNet VRAM:** Multi-ControlNet VRAM: Stacking OpenPose + Depth + Optical Flow causes near-exponential VRAM consumption. Mitigation: Gradient checkpointing + dynamic CPU offloading of inactive ControlNet tensors.
- **Base Model / Physics Conflict:** Base Model / Physics Conflict: Models under-trained on high-framerate explosion data fight optical flow constraints, producing over-smoothed artifacts. Mitigation: High-frequency noise injection into masked fluid regions during inpainting loop.
- **RAFT Optical Flow Latency:** RAFT Optical Flow Latency: Dense optical flow computation across full frames is expensive at inference time. Mitigation: Downsample optical flow to half resolution for the constraint check; upsample only for the inpainting mask.
- **3D Bounding Box Drift:** 3D Bounding Box Drift: Bounding box trackers drift over long occlusion windows (vehicle behind building). Mitigation: Re-acquire bounding box on re-entry using appearance-based re-identification.
- **Navier-Stokes Approximation Fidelity:** Navier-Stokes Approximation Fidelity: The L\_fluid loss is an approximation, not full CFD. Complex fluid interactions (two smoke sources merging) may produce physically implausible but mathematically passing frames. Mitigation: Luminance-based spot checks on smoke density gradients as a secondary heuristic.

# **13. Development Roadmap**

## **Phase 1 — Core Pipeline Validation (Complete)**
- Single-character generation with Identity Router and Kinematic Guardrail on Colab + IndiaAI.
- L\_physics constraint unit tests passing for all three kinematic metrics.
- Cosine similarity verification daemon operational with rewind/inpaint correction loop.
- All notebooks versioned and committed to GitHub /notebooks.
## **Phase 2 — Emotion & Body Language Stack**
- FACS Action Unit enforcement layer integrated and tested.
- Body language coherence module cross-validated against facial emotion state.
- Ocular dynamics module (gaze, blink, pupil, saccade) operational.
- Micro-expression generation and temporal ramp enforcement validated.
## **Phase 3 — Multi-Character**
- N-character identity routing with instance segmentation masking.
- Independent kinematic lattices per character with parallel verification.
- Cross-character contact physics module operational.
- Multi-character scene benchmarking suite on IndiaAI H100.
## **Phase 4 — Secondary Motion & Scene Coherence**
- Hair, clothing, and skin deformation physics.
- Breathing and secondary motion engine.
- Scene-level lighting and shadow consistency enforcement.
## **Phase 5 — Environmental Physics Module (NEW — V2.0)**

|<p>**V2.0 MILESTONE**</p><p>This phase transitions P.I.V.O.T. from a Character Consistency Tool into a Full Scene Simulation Engine. It is the highest technical complexity phase in the roadmap.</p>|
| :- |

- **Rigid Body Dynamics:** Rigid Body Dynamics: DepthAnything + 3D bounding box tracker integration. L\_rigid loss function implementation and unit testing.
- **RAFT Optical Flow Integration:** RAFT Optical Flow Integration: Dense optical flow extraction pipeline. L\_fluid loss implementation and Navier-Stokes continuity constraint validation.
- **Multi-ControlNet Routing:** Multi-ControlNet Routing: Gradient checkpointing implementation. Dynamic CPU offloading of inactive ControlNet tensors. VRAM benchmarking on T4 and A100.
- **Verification Daemon Extensions:** Verification Daemon Extensions: SSIM-based rigid body heuristics. Luminance expansion rate monitoring for blast sequences.
- **Base Model Alignment:** Base Model Alignment: High-frequency noise injection for fluid regions. Benchmark on explosion and vehicle footage datasets.
- **Full Scene Integration Tests:** Full Scene Integration Tests: Combined human + vehicle + explosion scene end-to-end on IndiaAI H100.
## **Phase 6 — Distribution**
- REST API v1 launched with authentication, rate limiting, scene physics config, and webhook callbacks.
- Python package (pivot-sdk) published to PyPI with environment module.
- ComfyUI node published to ComfyUI Manager with Scene Mode and credentials store integration.
- Hugging Face Inference Endpoints activated as dedicated inference layer.

|<p>P.I.V.O.T.  —  Product Requirements Document  —  Version 2.0  —  Confidential  —  Active Development</p><p>**Physics-Informed Video Optimization & Tracking  |  Full Scene Simulation Engine**</p>|
| :-: |

|Physics-Informed Video Optimization & Tracking  —  Active Development|Page |
| :- | -: |
