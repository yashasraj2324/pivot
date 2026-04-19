"""
PIVOT Identity Gate Demo Script
Run this to test identity verification with ArcFace embeddings.
"""
from core.cosine_similarity_gate import create_identity_gate
from core.identity_router import extract_arcface_embedding

gate = create_identity_gate(threshold=0.6, enable_logging=True)

img1 = "/content/164_compress.jpg"
img2 = "/content/images.png"

try:
    emb1 = extract_arcface_embedding(img1)
    emb2 = extract_arcface_embedding(img2)

    result = gate(emb1, emb2)

    print(f"Image 1: {img1}")
    print(f"Image 2: {img2}")
    print(f"Cosine Similarity: {result.similarity_score:.4f}")

    if result.passed:
        print("✅ Same Person")
    else:
        print("❌ Different Person")

except Exception as e:
    print("Error:", e)