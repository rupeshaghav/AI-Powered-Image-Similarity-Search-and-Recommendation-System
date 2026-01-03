"""
search_backend.py

CLIP-based image similarity search backend.
Loads precomputed CLIP embeddings and performs cosine similarity search.
"""

import json
from pathlib import Path

import numpy as np
import torch
import clip
from PIL import Image

# =========================
# CONFIG
# =========================
EMBED_DIR = Path("data/embeddings")
EMBED_FILE = EMBED_DIR / "image_embeddings.npy"
PATHS_FILE = EMBED_DIR / "image_paths.json"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# =========================
# LOAD MODEL (ONCE)
# =========================
print("✅ Using device:", DEVICE)
print("🔄 Loading CLIP model...")

model, preprocess = clip.load("ViT-B/32", device=DEVICE)
model.eval()

# =========================
# LOAD EMBEDDINGS (ONCE)
# =========================
print("📦 Loading embeddings...")

if not EMBED_FILE.exists() or not PATHS_FILE.exists():
    raise FileNotFoundError(
        "Embeddings not found. Run feature_extraction.py first."
    )

embeddings = np.load(EMBED_FILE)              # (N, 512)
with open(PATHS_FILE, "r") as f:
    image_paths = json.load(f)

assert embeddings.shape[0] == len(image_paths), "Embeddings and paths mismatch"

print(f"✅ Embeddings loaded: {embeddings.shape}")

# Normalize embeddings once (important for cosine similarity)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# =========================
# SEARCH FUNCTION
# =========================
def search_similar_images(query_image_path, top_k=5):
    """
    Args:
        query_image_path (str or Path): path to query image
        top_k (int): number of results

    Returns:
        List of tuples: (image_path, similarity_score)
    """

    # Load and preprocess query image
    image = Image.open(query_image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    # Encode query
    with torch.no_grad():
        query_embedding = model.encode_image(image_tensor)
        query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)

    query_embedding = query_embedding.cpu().numpy()  # (1, 512)

    # Cosine similarity (dot product because vectors are normalized)
    similarities = embeddings @ query_embedding.T    # (N, 1)
    similarities = similarities.squeeze(1)           # (N,)

    # Top-K
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append((image_paths[idx], float(similarities[idx])))

    return results


# =========================
# CLI TEST (optional)
# =========================
if __name__ == "__main__":
    test_image = "data/raw/animal_faces/train/cat/pixabay_cat_000455.jpg"

    print("🔍 Searching for:", test_image)
    results = search_similar_images(test_image, top_k=5)

    print("\nTop Similar Images:")
    for path, score in results:
        print(f"{path}  →  similarity {score:.4f}")
