"""
feature_extraction.py

Extracts CLIP image embeddings from multiple datasets
and saves them for fast similarity search.
"""

import torch
import clip
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import json

# CONFIG

RAW_DATA_DIR = Path("data/raw")
EMBED_DIR = Path("data/embeddings")

EMBED_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# DEVICE

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"✅ Using device: {device}")

# LOAD CLIP

print("🔄 Loading CLIP model...")
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# COLLECT IMAGES

image_paths = [
    p for p in RAW_DATA_DIR.rglob("*")
    if p.suffix.lower() in IMAGE_EXTENSIONS
]

print(f"🖼️ Total images found: {len(image_paths)}")

# FEATURE EXTRACTION

embeddings = []
valid_paths = []

with torch.no_grad():
    for img_path in tqdm(image_paths, desc="Extracting embeddings"):
        try:
            image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
            feature = model.encode_image(image)
            feature = feature / feature.norm(dim=-1, keepdim=True)

            embeddings.append(feature.cpu().numpy()[0])
            valid_paths.append(str(img_path))

        except Exception as e:
            print(f"⚠️ Skipped {img_path}: {e}")

# SAVE OUTPUT

embeddings = np.vstack(embeddings)

np.save(EMBED_DIR / "image_embeddings.npy", embeddings)

with open(EMBED_DIR / "image_paths.json", "w") as f:
    json.dump(valid_paths, f)

print("💾 Embeddings saved successfully!")
print(f"📦 Shape: {embeddings.shape}")
