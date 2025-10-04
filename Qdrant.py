import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Initialize client (in-memory for local testing)
client = QdrantClient(path=":memory:")  # Or host="localhost", port=6333 for Docker; add api_key for cloud

# Create collection
collection_name = "test_collection"
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=128, distance=Distance.COSINE)  # Cosine distance
)

# Generate sample data
vectors = [np.random.random(128).tolist() for _ in range(5)]
points = [
    PointStruct(id=i, vector=vec, payload={"text": f"doc{i}"}) for i, vec in enumerate(vectors)
]

# Insert (upsert for safety)
client.upsert(collection_name=collection_name, points=points)

# Query (top-1 similar)
query_vector = np.random.random(128).tolist()
hits = client.search(
    collection_name=collection_name,
    query_vector=query_vector,
    limit=1,
    with_payload=True  # Include metadata
)
print(f"Top match: ID={hits[0].id}, Score={hits[0].score}, Payload={hits[0].payload}")
