import os
import pickle

import numpy as np
from deepface import DeepFace

KNOWN_DIR = "known_faces"
OUTPUT_FILE = "encodings.pkl"
MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "retinaface"


def ensure_known_dir():
    if not os.path.exists(KNOWN_DIR):
        os.makedirs(KNOWN_DIR)
        print(f"Created directory '{KNOWN_DIR}'. Add face images (jpg/png) and rerun.")


def normalize_embedding(embedding):
    vector = np.array(embedding, dtype="float32")
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Embedding norm is zero.")
    return vector / norm


def extract_embedding(path):
    embeddings = DeepFace.represent(
        img_path=path,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=False,
    )
    embedding = embeddings[0] if isinstance(embeddings, list) else embeddings
    if isinstance(embedding, dict) and "embedding" in embedding:
        embedding = embedding["embedding"]
    return normalize_embedding(embedding)


def encode_known_faces():
    ensure_known_dir()

    known_encodings = []
    known_names = []

    for filename in sorted(os.listdir(KNOWN_DIR)):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = os.path.join(KNOWN_DIR, filename)
        name = os.path.splitext(filename)[0]

        try:
            encoding = extract_embedding(path)
            known_encodings.append(encoding)
            known_names.append(name)
            print(f"[OK] Encoded {name}")
        except Exception as e:
            print(f"[ERROR] Could not process '{filename}': {e}")

    if not known_encodings:
        print("No encoded faces. Add images under known_faces and rerun.")
        return

    payload = {
        "encodings": known_encodings,
        "names": known_names,
        "model_name": MODEL_NAME,
        "detector_backend": DETECTOR_BACKEND,
        "normalized": True,
    }

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(payload, f)

    print(f"Saved encodings to {OUTPUT_FILE} ({len(known_names)} identities)")


if __name__ == "__main__":
    encode_known_faces()
