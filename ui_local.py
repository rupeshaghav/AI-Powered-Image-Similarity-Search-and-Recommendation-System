import os
import json
import numpy as np
from flask import Flask, render_template, request, send_file
from PIL import Image

import torch
import clip

# ------------------ PATHS ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "..", "data")
EMBED_DIR = os.path.join(DATA_DIR, "embeddings")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")

EMBED_FILE = os.path.join(EMBED_DIR, "image_embeddings.npy")
PATH_FILE = os.path.join(EMBED_DIR, "image_paths.json")

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ------------------ DEVICE ------------------
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"✅ Using device: {device}")

# ------------------ LOAD MODEL ------------------
print("🔄 Loading CLIP model...")
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# ------------------ LOAD EMBEDDINGS ------------------
print("📦 Loading embeddings...")
image_embeddings = np.load(EMBED_FILE)
with open(PATH_FILE, "r") as f:
    image_paths = json.load(f)

print(f"✅ Embeddings shape: {image_embeddings.shape}")

# ------------------ FLASK APP ------------------
app = Flask(__name__)

# ------------------ SEARCH FUNCTION ------------------
def search_similar_images(query_image_path, top_k=5):
    image = preprocess(Image.open(query_image_path).convert("RGB")) \
        .unsqueeze(0).to(device)

    with torch.no_grad():
        query_embedding = model.encode_image(image)
        query_embedding /= query_embedding.norm(dim=-1, keepdim=True)

    query_embedding = query_embedding.cpu().numpy()

    similarities = image_embeddings @ query_embedding.T
    similarities = similarities.squeeze()

    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "path": image_paths[idx],
            "score": float(similarities[idx])
        })

    return results

# ------------------ ROUTES ------------------
@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    query_image_url = None

    if request.method == "POST":
        if "image" not in request.files:
            return "No image uploaded", 400

        file = request.files["image"]
        if file.filename == "":
            return "Empty filename", 400

        query_image = os.path.join(UPLOAD_DIR, file.filename)
        file.save(query_image)

        results = search_similar_images(query_image)

        # 🔥 THIS IS THE KEY FIX
        query_image_url = f"/image?path={query_image}"

    return render_template(
        "index.html",
        results=results,
        query_image=query_image_url
    )

@app.route("/image")
def serve_image():
    rel_path = request.args.get("path")
    if not rel_path:
        return "Missing image path", 400

    abs_path = os.path.join(BASE_DIR, "..", rel_path)

    if not os.path.exists(abs_path):
        return f"Image not found: {abs_path}", 404

    return send_file(abs_path)

# ------------------ RUN ------------------
if __name__ == "__main__":
    app.run(debug=True)
