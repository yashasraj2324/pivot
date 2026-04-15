"""
P.I.V.O.T. Core Module Tests
Test Identity Router components
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from core import identity_router


class DummyFace:
    def __init__(self, bbox, embedding):
        self.bbox = np.asarray(bbox, dtype=np.float32)
        self.normed_embedding = np.asarray(embedding, dtype=np.float32)


class DummyApp:
    def __init__(self, faces):
        self._faces = faces

    def get(self, img):
        return self._faces


class MockSAMPredictor:
    def __init__(self, mask: np.ndarray):
        self.mask = mask
        self.image = None
        self.received_box = None

    def set_image(self, image):
        self.image = image

    def predict(self, box=None, multimask_output=False):
        self.received_box = box
        return (np.expand_dims(self.mask, 0), None, None)


def _step(message: str) -> None:
    print(f"[Identity Router] {message}")


@pytest.fixture()
def temp_image(tmp_path: Path) -> Path:
    _step("Creating synthetic reference image for tests")
    image = np.zeros((128, 96, 3), dtype=np.uint8)
    image[24:100, 28:68] = 255
    path = tmp_path / "reference.png"
    cv2.imwrite(str(path), image)
    return path


class TestArcFaceEmbeddingExtraction:
    def test_extract_arcface_embedding_returns_normalized_512_vector(self, monkeypatch, temp_image):
        _step("Preparing mocked ArcFace face for normalized embedding extraction")
        embedding = np.linspace(1.0, 512.0, 512, dtype=np.float32)
        face = DummyFace(bbox=(0, 0, 10, 10), embedding=embedding)

        monkeypatch.setattr(identity_router.Path, "exists", lambda self: True)
        monkeypatch.setattr(identity_router.cv2, "imread", lambda path: np.zeros((32, 32, 3), dtype=np.uint8))
        monkeypatch.setattr(identity_router, "_get_app", lambda *args, **kwargs: DummyApp([face]))

        _step("Extracting embedding and checking shape, dtype, and normalization")
        result = identity_router.extract_arcface_embedding(str(temp_image))

        assert result.shape == (512,)
        assert result.dtype == np.float32
        assert np.isclose(np.linalg.norm(result), 1.0, atol=1e-5)

    def test_extract_arcface_embedding_uses_largest_face_when_multiple_faces_exist(self, monkeypatch, temp_image):
        _step("Preparing multiple mocked faces to verify largest-face selection")
        small = DummyFace(bbox=(0, 0, 10, 10), embedding=np.ones(512, dtype=np.float32))
        large = DummyFace(bbox=(0, 0, 20, 20), embedding=np.full(512, 2.0, dtype=np.float32))

        monkeypatch.setattr(identity_router.Path, "exists", lambda self: True)
        monkeypatch.setattr(identity_router.cv2, "imread", lambda path: np.zeros((32, 32, 3), dtype=np.uint8))
        monkeypatch.setattr(identity_router, "_get_app", lambda *args, **kwargs: DummyApp([small, large]))

        _step("Extracting embedding from the largest detected face")
        result = identity_router.extract_arcface_embedding(str(temp_image), multi_face_policy="largest")

        expected = np.full(512, 2.0, dtype=np.float32)
        expected = expected / np.linalg.norm(expected)
        assert np.allclose(result, expected)

    def test_extract_arcface_embedding_raises_for_multiple_faces_in_strict_mode(self, monkeypatch, temp_image):
        _step("Preparing strict multi-face failure scenario")
        faces = [
            DummyFace(bbox=(0, 0, 10, 10), embedding=np.ones(512, dtype=np.float32)),
            DummyFace(bbox=(0, 0, 20, 20), embedding=np.ones(512, dtype=np.float32)),
        ]

        monkeypatch.setattr(identity_router.Path, "exists", lambda self: True)
        monkeypatch.setattr(identity_router.cv2, "imread", lambda path: np.zeros((32, 32, 3), dtype=np.uint8))
        monkeypatch.setattr(identity_router, "_get_app", lambda *args, **kwargs: DummyApp(faces))

        _step("Verifying strict mode rejects multiple faces")
        with pytest.raises(ValueError, match="Multiple faces"):
            identity_router.extract_arcface_embedding(str(temp_image), multi_face_policy="strict")


class TestBatchExtraction:
    def test_extract_arcface_embeddings_batch_raises_when_any_image_fails(self, monkeypatch):
        _step("Mocking batch extraction with one failing image")
        good = np.ones(512, dtype=np.float32)

        def fake_extract(path, **kwargs):
            if path.endswith("bad.png"):
                raise ValueError("No face detected")
            return good

        monkeypatch.setattr(identity_router, "extract_arcface_embedding", fake_extract)

        _step("Checking batch extraction aggregates failures into one RuntimeError")
        with pytest.raises(RuntimeError, match="failed during batch extraction"):
            identity_router.extract_arcface_embeddings_batch(["good.png", "bad.png"])


class TestCosineSimilarity:
    def test_cosine_similarity_identical_vectors_is_one(self):
        _step("Comparing identical vectors for cosine similarity")
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        assert identity_router.cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_cosine_similarity_zero_vector_is_zero(self):
        _step("Checking cosine similarity handling for zero vector input")
        vec = np.zeros(3, dtype=np.float32)
        other = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        assert identity_router.cosine_similarity(vec, other) == 0.0


class TestLocalizedIdentityMask:
    def test_build_localized_identity_mask_from_pose_keypoints(self, temp_image):
        _step("Building localized mask from synthetic pose keypoints")
        keypoints = np.array(
            [
                [40, 20, 0.9],
                [40, 40, 0.9],
                [40, 60, 0.9],
                [40, 80, 0.9],
                [35, 35, 0.9],
                [45, 35, 0.9],
                [30, 45, 0.9],
                [50, 45, 0.9],
                [25, 55, 0.9],
                [55, 55, 0.9],
                [35, 70, 0.9],
                [45, 70, 0.9],
                [33, 90, 0.9],
                [47, 90, 0.9],
                [31, 110, 0.9],
                [49, 110, 0.9],
                [40, 120, 0.9],
            ],
            dtype=np.float32,
        )

        _step("Running pose-guided masking and checking mask coverage")
        mask = identity_router.build_localized_identity_mask(str(temp_image), pose_keypoints=keypoints)

        assert mask.shape == (128, 96)
        assert mask.dtype == np.float32
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0
        assert np.count_nonzero(mask) > 0

    def test_build_localized_identity_mask_refines_with_sam(self, temp_image):
        _step("Preparing pose keypoints and a mock SAM predictor")
        keypoints = np.array(
            [
                [40, 20, 0.9],
                [40, 40, 0.9],
                [40, 60, 0.9],
                [40, 80, 0.9],
                [35, 35, 0.9],
                [45, 35, 0.9],
                [30, 45, 0.9],
                [50, 45, 0.9],
                [25, 55, 0.9],
                [55, 55, 0.9],
                [35, 70, 0.9],
                [45, 70, 0.9],
                [33, 90, 0.9],
                [47, 90, 0.9],
                [31, 110, 0.9],
                [49, 110, 0.9],
                [40, 120, 0.9],
            ],
            dtype=np.float32,
        )

        sam_mask = np.zeros((128, 96), dtype=np.float32)
        sam_mask[10:100, 20:70] = 1.0
        predictor = MockSAMPredictor(sam_mask)

        _step("Running pose-guided masking with SAM refinement")
        mask = identity_router.build_localized_identity_mask(
            str(temp_image),
            pose_keypoints=keypoints,
            sam_predictor=predictor,
        )

        assert mask.shape == (128, 96)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0
        assert predictor.image is not None
        assert predictor.received_box is not None
        assert np.count_nonzero(mask) > 0

    def test_build_localized_identity_mask_falls_back_to_face_region(self, monkeypatch, temp_image):
        _step("Preparing face-only fallback scenario")
        face = DummyFace(bbox=(20, 30, 60, 90), embedding=np.ones(512, dtype=np.float32))

        monkeypatch.setattr(identity_router, "_get_app", lambda *args, **kwargs: DummyApp([face]))

        _step("Running fallback mask generation from face bounding box")
        mask = identity_router.build_localized_identity_mask(str(temp_image))

        assert mask.shape == (128, 96)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0
        assert np.count_nonzero(mask) > 0