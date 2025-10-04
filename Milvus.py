from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
import numpy as np

# Connect (assumes local Docker)
connections.connect(alias="default", host="localhost", port="19530")  # For cloud: uri="https://your-zilliz-url", token="your-token"

collection_name = "test_collection"

# Define schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=100)
]
schema = CollectionSchema(fields=fields, description="Test collection")
collection = Collection(name=collection_name, schema=schema)

# Generate sample data
entities = [
    ["" for _ in range(5)],  # Empty IDs for auto-generation
    [np.random.random(128).tolist() for _ in range(5)],  # Vectors
    [f"doc{i}" for i in range(5)]  # Metadata
]

# Insert and flush
collection.insert(entities)
collection.flush()

# Create index (IVF_FLAT for approx. search)
index_params = {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
collection.create_index(field_name="embedding", index_params=index_params)
collection.load()  # Load into memory

# Search (top-1 similar, include metadata)
search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
query_vec = [np.random.random(128).tolist()]
results = collection.search(
    data=query_vec,
    anns_field="embedding",
    param=search_params,
    limit=1,
    output_fields=["text"]
)
match = results[0][0] if results[0] else None
print(f"Top match: ID={match.id if match else 'None'}, Distance={match.distance if match else 'None'}, Text={match.entity.get('text') if match else 'None'}")
