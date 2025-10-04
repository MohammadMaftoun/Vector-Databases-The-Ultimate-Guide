import weaviate
import numpy as np

# Initialize client (assumes local Docker instance)
client = weaviate.connect_to_local()  # For cloud, use connect_to_wcs as noted above

collection_name = "Test"

# Define schema (class with text property; manual vectors)
schema = {
    "classes": [
        {
            "class": collection_name,
            "vectorizer": "none",  # Provide vectors manually
            "properties": [
                {"name": "text", "dataType": ["text"], "moduleConfig": {"text2vec-contextionary": {}}}
            ]
        }
    ]
}
client.collections.delete(collection_name)  # Clean slate
client.collections.create_from_dict(schema)  # Or use client.schema.create() in older versions

# Generate sample data
vectors = [np.random.random(128).tolist() for _ in range(5)]

# Batch insert with vectors and metadata
with client.batch.dynamic() as batch:
    for i, vec in enumerate(vectors):
        batch.add_object(
            data_object={"text": f"doc{i}"},
            vector=vec,
            class_name=collection_name
        )

# Query (top-1 similar using near_vector)
query_vec = np.random.random(128).tolist()
response = (
    client.query.get(collection_name, ["text"])
    .with_near_vector(vector=query_vec)
    .with_limit(1)
    .do()
)
result = response["data"]["Get"][collection_name][0] if response["data"]["Get"][collection_name] else None
print(f"Top match: Text={result['text'] if result else 'None'}")
