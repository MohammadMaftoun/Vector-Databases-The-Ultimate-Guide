import faiss
import numpy as np

# Generate sample data
d = 128  # Vector dimension
nb = 5   # Number of vectors
xb = np.random.random((nb, d)).astype('float32')  # Sample vectors (imagine metadata like ["doc0", "doc1", ...] mapped externally)
xq = np.random.random((1, d)).astype('float32')   # Query vector

# Create and populate index
index = faiss.IndexFlatL2(d)  # L2 distance (Euclidean)
index.add(xb)                 # Add vectors

# Search (top-1 similar vector)
k = 1
distances, indices = index.search(xq, k)
print(f"Top matches (indices): {indices}")  # Output: [[0]] (or similar; map to your metadata)
print(f"Distances: {distances}")
