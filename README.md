# AI-Powered Image Similarity Search and Recommendation System

An AI-powered image similarity search system using OpenAI’s CLIP model to find visually similar images based purely on visual content. The system extracts deep image embeddings, computes cosine similarity, and provides a Flask-based web interface for real-time image search and result visualization.

## Features

* CLIP-based visual embeddings
* Cosine similarity–based image retrieval
* Flask-based web user interface
* Scalable to large image datasets

## Tech Stack

Python, PyTorch, OpenAI CLIP, NumPy, Flask

## Dataset

Uses a public image dataset (e.g., Animal Faces Dataset or Intel Image Classification Dataset).
The dataset is **not included** in this repository due to size constraints.

## How to Run

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Generate image embeddings

```bash
python feature_extraction.py
```

3. (Optional) Activate virtual environment

```bash
source venv/bin/activate
```

4. Run the web application

```bash
python app/ui_local.py
```

## Platform Compatibility

* Runs on **Windows, macOS, and Linux**
* On Windows, the model runs on **CPU by default**
* Hardware acceleration is automatically selected when available (MPS on macOS, CUDA if enabled)
