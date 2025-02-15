import faiss
import numpy as np
import os

# Directory containing embeddings
DATA_DIR = "img/db/"
MODEL_DIR = "models/"

# Load feature vectors
feature_files = [f for f in os.listdir(DATA_DIR) if f.endswith("_embedding.npy")]
cup_embeddings = [np.load(os.path.join(DATA_DIR, f)).astype('float32') for f in feature_files]

if not cup_embeddings:
    print("No feature vectors found. Run extract_features.py first!")
    exit()

cup_embeddings = np.array(cup_embeddings)

# Create FAISS index (L2 distance for similarity search)
d = cup_embeddings.shape[1]  # Feature vector size
index = faiss.IndexFlatL2(d)  # L2 distance index
index.add(cup_embeddings)  # Add feature vectors

# Save FAISS index
os.makedirs(MODEL_DIR, exist_ok=True)
faiss.write_index(index, os.path.join(MODEL_DIR, "cava_cups.index"))

print(f"✅ Database built successfully with {len(cup_embeddings)} cava cups!")
