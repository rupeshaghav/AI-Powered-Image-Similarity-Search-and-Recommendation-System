# AI-Powered-Image-Similarity-Search-and-Recommendation-System
An AI-powered image similarity search system using OpenAI’s CLIP model to find visually similar images based on content alone. It extracts deep image embeddings, applies cosine similarity, and provides a Flask-based web interface for real-time image search and results visualization.
## Features
•⁠  ⁠CLIP-based visual embeddings
•⁠  ⁠Cosine similarity search
•⁠  ⁠Flask-based web UI
•⁠  ⁠Scalable to large image datasets

## Tech Stack
Python, PyTorch, CLIP, NumPy, Flask

## Dataset
Uses a public image dataset (e.g., Animal Faces / Intel Image Classification Dataset).
Dataset is not included in the repository due to size constraints.

## How to Run
1.⁠ ⁠Install dependencies:
  pip install -r requirements.txt

2.⁠ ⁠Generate embeddings
3. Activate virtual enviornment :
  source venv/bin/activate
4.⁠ ⁠Run:
  python app/ui_local.py
