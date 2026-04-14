"""
PIVOT Project — Phase 1: Core Pipeline Validation
Requirement: IR-REQ-001
 
Implements ArcFace embedding extraction using the InsightFace buffalo_l model.
Produces stable 512-dimensional facial embeddings from reference photographs
for use by the Identity Router, cosine similarity gate, and Verification Daemon.
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