import faiss
import numpy as np
from extract_features import extract_features
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# Load FAISS index
index = faiss.read_index("models/cava_cups.index")

# Get list of image files in database
db_path = "img/db"
image_files = sorted([f for f in os.listdir(db_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])

# Load and process query image
query_image = "img/input/image2.jpg"
query_embedding = extract_features(query_image)
query_embedding = np.expand_dims(query_embedding.astype('float32'), axis=0)

# Search in FAISS (find 5 most similar)
D, I = index.search(query_embedding, 5)

# Print results with filenames and distances
print("\nSearch Results:")
print("-" * 50)
for idx, (match_idx, distance) in enumerate(zip(I[0], D[0])):
    if match_idx != -1:
        matched_image = image_files[match_idx]
        print(f"Match {idx+1}: {matched_image}")
        print(f"Similarity Score: {1 - distance/100:.2%}")  # Convert distance to similarity percentage
        print("-" * 50)