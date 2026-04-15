"""
PIVOT Project — Phase 1: Core Pipeline Validation
Requirement: IR-REQ-001
 
Implements ArcFace embedding extraction using the InsightFace buffalo_l model.
Produces stable 512-dimensional facial embeddings from reference photographs
for use by the Identity Router, cosine similarity gate, localized masking,
and Verification Daemon.
"""
from __future__ import annotations
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

@lru_cache(maxsize=1)
def _get_app(model_name: str = "buffalo_l", ctx_id: int = 0):
    """
    Load and cache the InsightFace FaceAnalysis app.
 
    Args:
        model_name: InsightFace model pack name. Default is 'buffalo_l',
                    which bundles RetinaFace (detection) + ArcFace (recognition).
        ctx_id:     GPU device id. Use 0 for first GPU, -1 for CPU.
 
    Returns:
        Prepared insightface.app.FaceAnalysis instance.
    """
    try:
        import insightface
        from insightface.app import FaceAnalysis
    except ImportError as exc:
        raise ImportError(
            "insightface is required. Install with: pip install insightface onnxruntime"
        ) from exc
 
    app = FaceAnalysis(name=model_name, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    return app


def extract_arcface_embedding(
    image_path: str,
    *,
    model_name: str = "buffalo_l",
    ctx_id: int = 0,
    multi_face_policy: str = "largest",
) -> np.ndarray:
    """
    Extract a 512-dimensional ArcFace embedding from a reference photograph.
 
    Parameters
    ----------
    image_path : str
        Path to the reference image (JPEG, PNG, BMP, TIFF, WebP, …).
    model_name : str
        InsightFace model pack. Default ``"buffalo_l"`` (ArcFace R100).
    ctx_id : int
        ONNX execution device.  ``0`` = GPU 0, ``-1`` = CPU.
    multi_face_policy : str
        How to handle images with more than one detected face:
        - ``"largest"``  — silently use the largest bounding-box face (default).
        - ``"strict"``   — raise ``ValueError`` if more than one face is found.
        - ``"first"``    — use the first face returned by the detector.
 
    Returns
    -------
    np.ndarray
        Unit-normalised 512-d embedding vector (dtype float32).
 
    Raises
    ------
    FileNotFoundError
        If *image_path* does not exist or cannot be read.
    ValueError
        If no face is detected, or *multi_face_policy* is ``"strict"`` and
        multiple faces are detected.
    """
    # ------------------------------------------------------------------ #
    # 1. Load image
    # ------------------------------------------------------------------ #
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
 
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise FileNotFoundError(
            f"OpenCV could not read the image (unsupported format or corrupt file): {image_path}"
        )
 
    # InsightFace expects BGR (same as OpenCV default)
    img = img_bgr
 
    # ------------------------------------------------------------------ #
    # 2. Detect + align faces
    # ------------------------------------------------------------------ #
    app = _get_app(model_name=model_name, ctx_id=ctx_id)
    faces = app.get(img)
 
    if not faces:
        raise ValueError(
            f"No face detected in image: {image_path}. "
            "Ensure the image contains a clear, frontal face."
        )
 
    # ------------------------------------------------------------------ #
    # 3. Face selection policy
    # ------------------------------------------------------------------ #
    if len(faces) > 1:
        if multi_face_policy == "strict":
            raise ValueError(
                f"Multiple faces ({len(faces)}) detected in image: {image_path}. "
                "Provide an image with exactly one face, or use multi_face_policy='largest'."
            )
        elif multi_face_policy == "largest":
            # det_score is unreliable for "largest"; use bbox area instead
            face = max(faces, key=lambda f: _bbox_area(f.bbox))
        else:  # "first"
            face = faces[0]
    else:
        face = faces[0]
 
    # ------------------------------------------------------------------ #
    # 4. Extract embedding
    # ------------------------------------------------------------------ #
    embedding: np.ndarray = face.normed_embedding  # already L2-normalised by InsightFace
 
    if embedding is None:
        raise ValueError(
            "InsightFace returned a face object without an embedding. "
            "Verify that the recognition module is included in the model pack."
        )
 
    embedding = np.asarray(embedding, dtype=np.float32)
 
    # Paranoia: re-normalise in case the model pack skipped it
    norm = np.linalg.norm(embedding)
    if norm > 1e-6:
        embedding = embedding / norm
 
    # ------------------------------------------------------------------ #
    # 5. Validate dimensionality
    # ------------------------------------------------------------------ #
    if embedding.shape != (512,):
        raise ValueError(
            f"Expected 512-d embedding, got shape {embedding.shape}. "
            "Confirm model_name='buffalo_l' (ArcFace R100)."
        )
 
    return embedding

def extract_arcface_embeddings_batch(
    image_paths: list[str],
    **kwargs,
) -> list[np.ndarray]:
    """
    Convenience wrapper: extract embeddings for a list of image paths.
 
    Skips images that raise exceptions and re-raises them at the end as a
    grouped ``RuntimeError`` so that one bad image does not abort the batch.
 
    Parameters
    ----------
    image_paths : list[str]
        Ordered list of image file paths.
    **kwargs
        Forwarded to :func:`extract_arcface_embedding`.
 
    Returns
    -------
    list[np.ndarray]
        Embeddings in the same order as *image_paths*.  Entries for failed
        images are ``None``; check the raised exception for details.
    """
    results: list[Optional[np.ndarray]] = []
    errors: list[str] = []
 
    for path in image_paths:
        try:
            results.append(extract_arcface_embedding(path, **kwargs))
        except Exception as exc:  # noqa: BLE001
            results.append(None)
            errors.append(f"{path}: {exc}")
 
    if errors:
        raise RuntimeError(
            f"{len(errors)} image(s) failed during batch extraction:\n" + "\n".join(errors)
        )
 
    return results  # type: ignore[return-value]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two embedding vectors.
 
    Both vectors are expected to be L2-normalised (as returned by
    :func:`extract_arcface_embedding`).  For un-normalised vectors the
    function normalises internally.
 
    Returns
    -------
    float
        Similarity score in [-1, 1].  Identical embeddings → 1.0.
    """
    a = a.astype(np.float32)
    b = b.astype(np.float32)
 
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
 
    if norm_a < 1e-6 or norm_b < 1e-6:
        return 0.0
 
    return float(np.dot(a / norm_a, b / norm_b))

def _bbox_area(bbox) -> float:
    """Return pixel area of a bounding box [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox
    return max(0.0, float((x2 - x1) * (y2 - y1)))    


def build_localized_identity_mask(
    image_path: str,
    *,
    sam_predictor=None,
    dwpose_model=None,
    pose_keypoints=None,
    instance_index: int = 0,
    min_keypoint_confidence: float = 0.35,
    mask_dilation: int = 21,
    bbox_expansion: float = 0.15,
) -> np.ndarray:
    """
    Build a localized instance mask for identity conditioning.

    The preferred path is DWPose keypoints to produce a person prior, followed
    by optional SAM refinement when a predictor is supplied. If neither model is
    available, the function falls back to a face-driven approximation so the
    caller still receives a usable mask instead of failing the pipeline.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise FileNotFoundError(
            f"OpenCV could not read the image (unsupported format or corrupt file): {image_path}"
        )

    height, width = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    instances = _coerce_pose_instances(pose_keypoints)
    if not instances and dwpose_model is not None:
        instances = _extract_dwpose_instances(dwpose_model, img_rgb)

    selected = None
    if instances:
        selected = _select_pose_instance(instances, instance_index=instance_index)

    pose_mask = None
    pose_box = None
    if selected is not None:
        pose_mask, pose_box = _pose_instance_mask(
            selected,
            height=height,
            width=width,
            min_keypoint_confidence=min_keypoint_confidence,
            dilation=mask_dilation,
            bbox_expansion=bbox_expansion,
        )

    if pose_mask is None:
        pose_mask, pose_box = _face_fallback_mask(
            img_bgr,
            height=height,
            width=width,
            bbox_expansion=bbox_expansion,
        )

    mask = pose_mask

    if sam_predictor is not None:
        sam_mask = _refine_mask_with_sam(
            sam_predictor,
            img_rgb,
            base_mask=pose_mask,
            bbox=pose_box,
        )
        if sam_mask is not None:
            mask = np.maximum(mask, sam_mask)

    mask = np.asarray(mask, dtype=np.float32)
    if mask.max() > 1.0:
        mask = mask / 255.0

    return np.clip(mask, 0.0, 1.0)


def _coerce_pose_instances(pose_keypoints) -> list[np.ndarray]:
    """Normalize a pose payload into a list of instance arrays."""
    if pose_keypoints is None:
        return []

    if isinstance(pose_keypoints, np.ndarray):
        if pose_keypoints.ndim == 2:
            return [pose_keypoints]
        if pose_keypoints.ndim == 3:
            return [pose_keypoints[index] for index in range(pose_keypoints.shape[0])]

    instances: list[np.ndarray] = []
    for item in pose_keypoints:
        if isinstance(item, dict) and "keypoints" in item:
            instances.append(np.asarray(item["keypoints"]))
        else:
            instances.append(np.asarray(item))

    return instances


def _select_pose_instance(instances: list[np.ndarray], *, instance_index: int = 0) -> np.ndarray:
    """Select a single person instance from a DWPose payload."""
    if not instances:
        raise ValueError("Cannot select a pose instance from an empty payload.")

    if 0 <= instance_index < len(instances):
        return np.asarray(instances[instance_index])

    scored = [(_pose_instance_area(instance), index) for index, instance in enumerate(instances)]
    _, best_index = max(scored, key=lambda item: item[0])
    return np.asarray(instances[best_index])


def _pose_instance_area(keypoints: np.ndarray, *, confidence_threshold: float = 0.35) -> float:
    """Estimate the visible area of a pose instance from its confident keypoints."""
    points = _valid_pose_points(keypoints, confidence_threshold=confidence_threshold)
    if not points:
        return 0.0

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return max(0.0, float((max(xs) - min(xs)) * (max(ys) - min(ys))))


def _valid_pose_points(
    keypoints: np.ndarray,
    *,
    confidence_threshold: float = 0.35,
) -> list[tuple[float, float]]:
    """Extract 2D points with sufficient confidence from a pose array."""
    coords = np.asarray(keypoints, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[0] == 0:
        return []

    points: list[tuple[float, float]] = []
    for row in coords:
        if row.shape[0] < 2:
            continue
        x = float(row[0])
        y = float(row[1])
        confidence = float(row[2]) if row.shape[0] >= 3 else 1.0
        if confidence >= confidence_threshold:
            points.append((x, y))

    return points


def _pose_instance_mask(
    keypoints: np.ndarray,
    *,
    height: int,
    width: int,
    min_keypoint_confidence: float,
    dilation: int,
    bbox_expansion: float,
) -> tuple[np.ndarray, tuple[float, float, float, float] | None]:
    """Convert pose keypoints into a person-shaped binary mask."""
    coords = np.asarray(keypoints, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[0] == 0:
        return np.zeros((height, width), dtype=np.float32), None

    canvas = np.zeros((height, width), dtype=np.uint8)
    points = _valid_pose_points(coords, confidence_threshold=min_keypoint_confidence)
    if not points:
        return np.zeros((height, width), dtype=np.float32), None

    bbox = _expand_bbox(_pose_bbox(coords, min_keypoint_confidence), width, height, bbox_expansion)

    radius = max(4, int(round(min(height, width) * 0.0125)))
    thickness = max(8, radius * 2)

    for row in coords:
        if row.shape[0] < 2:
            continue
        confidence = float(row[2]) if row.shape[0] >= 3 else 1.0
        if confidence < min_keypoint_confidence:
            continue
        center = (int(round(float(row[0]))), int(round(float(row[1]))))
        cv2.circle(canvas, center, radius, 255, -1)

    for start, end in _COCO_17_CONNECTIONS:
        if start >= len(coords) or end >= len(coords):
            continue
        row_a = coords[start]
        row_b = coords[end]
        if row_a.shape[0] < 2 or row_b.shape[0] < 2:
            continue
        confidence_a = float(row_a[2]) if row_a.shape[0] >= 3 else 1.0
        confidence_b = float(row_b[2]) if row_b.shape[0] >= 3 else 1.0
        if confidence_a < min_keypoint_confidence or confidence_b < min_keypoint_confidence:
            continue
        pt_a = (int(round(float(row_a[0]))), int(round(float(row_a[1]))))
        pt_b = (int(round(float(row_b[0]))), int(round(float(row_b[1]))))
        cv2.line(canvas, pt_a, pt_b, 255, thickness)

    if dilation > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation, dilation))
        canvas = cv2.dilate(canvas, kernel, iterations=1)

    return canvas.astype(np.float32) / 255.0, bbox


def _pose_bbox(
    keypoints: np.ndarray,
    confidence_threshold: float,
) -> tuple[int, int, int, int]:
    """Compute a tight bounding box for confident pose points."""
    coords = np.asarray(keypoints, dtype=np.float32)
    valid_x: list[float] = []
    valid_y: list[float] = []

    for row in coords:
        if row.shape[0] < 2:
            continue
        confidence = float(row[2]) if row.shape[0] >= 3 else 1.0
        if confidence < confidence_threshold:
            continue
        valid_x.append(float(row[0]))
        valid_y.append(float(row[1]))

    if not valid_x or not valid_y:
        return 0, 0, 0, 0

    return int(min(valid_x)), int(min(valid_y)), int(max(valid_x)), int(max(valid_y))


def _expand_bbox(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    expansion: float,
) -> tuple[int, int, int, int]:
    """Expand a bounding box while keeping it inside the image bounds."""
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return 0, 0, width, height

    box_w = x2 - x1
    box_h = y2 - y1
    pad_x = int(round(box_w * expansion))
    pad_y = int(round(box_h * expansion))

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)
    return x1, y1, x2, y2


def _face_fallback_mask(
    img_bgr: np.ndarray,
    *,
    height: int,
    width: int,
    bbox_expansion: float,
) -> tuple[np.ndarray, tuple[float, float, float, float] | None]:
    """Fallback mask driven by the detected face bounding box."""
    try:
        app = _get_app()
        faces = app.get(img_bgr)
    except Exception:  # noqa: BLE001
        faces = []

    if not faces:
        return np.zeros((height, width), dtype=np.float32), None

    face = max(faces, key=lambda item: _bbox_area(item.bbox))
    x1, y1, x2, y2 = face.bbox
    x1, y1, x2, y2 = _expand_bbox(
        (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
        width,
        height,
        bbox_expansion * 3.0,
    )

    mask = np.zeros((height, width), dtype=np.uint8)
    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    axes = (max(1, (x2 - x1) // 2), max(1, int((y2 - y1) * 0.9)))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return mask.astype(np.float32) / 255.0, (float(x1), float(y1), float(x2), float(y2))


def _extract_dwpose_instances(dwpose_model, img_rgb: np.ndarray) -> list[np.ndarray]:
    """Call a DWPose-like model and normalize its output to keypoint arrays."""
    candidate_outputs = []

    if hasattr(dwpose_model, "__call__"):
        try:
            candidate_outputs.append(dwpose_model(img_rgb))
        except Exception:  # noqa: BLE001
            pass

    for method_name in ("detect", "estimate", "infer", "predict", "forward"):
        if not hasattr(dwpose_model, method_name):
            continue
        method = getattr(dwpose_model, method_name)
        try:
            candidate_outputs.append(method(img_rgb))
        except TypeError:
            try:
                candidate_outputs.append(method(image=img_rgb))
            except Exception:  # noqa: BLE001
                continue
        except Exception:  # noqa: BLE001
            continue

    for output in candidate_outputs:
        instances = _coerce_pose_instances(_extract_keypoints_payload(output))
        if instances:
            return instances

    return []


def _extract_keypoints_payload(output):
    """Best-effort extraction of keypoint payloads from DWPose-style outputs."""
    if output is None:
        return None

    if isinstance(output, dict):
        for key in ("keypoints", "persons", "poses", "results", "data"):
            if key in output:
                return output[key]
        return output

    if isinstance(output, (list, tuple)):
        return output

    for attribute in ("keypoints", "persons", "poses", "results", "data"):
        if hasattr(output, attribute):
            return getattr(output, attribute)

    return output


def _refine_mask_with_sam(
    sam_predictor,
    img_rgb: np.ndarray,
    *,
    base_mask: np.ndarray,
    bbox: tuple[float, float, float, float] | None,
) -> np.ndarray | None:
    """Refine a coarse person mask with a SAM predictor when available."""
    if sam_predictor is None:
        return None

    try:
        if hasattr(sam_predictor, "set_image"):
            sam_predictor.set_image(img_rgb)

        if bbox is None:
            bbox = _mask_to_bbox(base_mask)
        if bbox is None:
            return None

        box = np.asarray(bbox, dtype=np.float32)

        if hasattr(sam_predictor, "predict"):
            try:
                predicted = sam_predictor.predict(box=box, multimask_output=False)
            except TypeError:
                predicted = sam_predictor.predict(box=box)
        elif hasattr(sam_predictor, "predict_torch"):
            predicted = sam_predictor.predict_torch(boxes=box[None, :])
        else:
            return None

        masks = _extract_mask_payload(predicted)
        if masks is None:
            return None

        mask = np.asarray(masks, dtype=np.float32)
        if mask.ndim == 3:
            mask = mask[0]
        if mask.max() > 1.0:
            mask = mask / 255.0
        return np.clip(mask, 0.0, 1.0)
    except Exception:  # noqa: BLE001
        return None


def _mask_to_bbox(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    """Compute a bounding box from a binary mask."""
    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def _extract_mask_payload(output):
    """Normalize the mask payload returned by a SAM-style predictor."""
    if output is None:
        return None

    if isinstance(output, dict):
        for key in ("masks", "mask", "pred_masks", "segmentation"):
            if key in output:
                return output[key]
        return None

    if isinstance(output, (list, tuple)):
        for item in output:
            if isinstance(item, np.ndarray):
                return item
            if hasattr(item, "detach"):
                return item.detach().cpu().numpy()
        return None

    if isinstance(output, np.ndarray):
        return output

    if hasattr(output, "detach"):
        return output.detach().cpu().numpy()

    for attribute in ("masks", "mask", "pred_masks", "segmentation"):
        if hasattr(output, attribute):
            value = getattr(output, attribute)
            if hasattr(value, "detach"):
                return value.detach().cpu().numpy()
            return value

    return None


_COCO_17_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)