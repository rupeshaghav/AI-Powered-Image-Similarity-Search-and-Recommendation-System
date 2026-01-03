import os
from PIL import Image
from tqdm import tqdm

def preprocess_images(input_dir, output_dir, size=(224, 224)):
    """
    Resizes and converts images to RGB, saving them in the same folder structure.
    """
    os.makedirs(output_dir, exist_ok=True)

    for root, _, files in os.walk(input_dir):
        for file in tqdm(files, desc=f"Processing {os.path.basename(root)}"):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                input_path = os.path.join(root, file)
                rel_path = os.path.relpath(root, input_dir)
                output_path = os.path.join(output_dir, rel_path)
                os.makedirs(output_path, exist_ok=True)

                try:
                    img = Image.open(input_path).convert("RGB")
                    img = img.resize(size)
                    img.save(os.path.join(output_path, file))
                except Exception as e:
                    print(f"⚠️  Error with {input_path}: {e}")

if __name__ == "__main__":
    preprocess_images("data/raw/intel_dataset", "data/processed/intel_dataset")
    preprocess_images("data/raw/animal_faces", "data/processed/animal_faces")