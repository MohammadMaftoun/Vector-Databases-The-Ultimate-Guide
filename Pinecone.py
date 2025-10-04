import pinecone
import numpy as np

# Initialize (cloud required)
pinecone.init(api_key="your-api-key", environment="us-west1-gcp")  # Env varies by region
index_name = "your-index-name"  # Create via Pinecone dashboard (dim=128, metric="cosine")
index = pinecone.Index(index_name)

# Generate sample data
vectors = [(str(i), np.random.random(128).tolist(), {"text": f"doc{i}"}) for i in range(5)]

# Upsert (insert or update)
index.upsert(vectors=vectors)

# Query (top-1 similar, include metadata)
query_vec = np.random.random(128).tolist()
query_response = index.query(
    vector=query_vec,
    top_k=1,
    include_metadata=True
)
match = query_response['matches'][0] if query_response['matches'] else None
print(f"Top match: ID={match['id'] if match else 'None'}, Score={match['score'] if match else 'None'}, Metadata={match['metadata'] if match else 'None'}")
