from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType

def store_in_milvus(embeddings, chunks, index_type):
    connections.connect("default", host="localhost", port="19530")
    
    collection_name = f'cuda_docs_{index_type.lower()}'
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="topic", dtype=DataType.INT64)
    ]
    schema = CollectionSchema(fields, f"CUDA documentation collection ({index_type})")
    
    if Collection.has_collection(collection_name):
        Collection(collection_name).drop()
    collection = Collection(name=collection_name, schema=schema)

    entities = [
        embeddings.tolist(),
        [chunk[0] for chunk in chunks],
        [chunk[1] for chunk in chunks],
        [chunk[2] for chunk in chunks]
    ]

    collection.insert(entities)

    if index_type == "FLAT":
        index_params = {
            "index_type": "FLAT",
            "metric_type": "L2",
            "params": {}
        }
    elif index_type == "IVF_FLAT":
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128}
        }
    
    collection.create_index(field_name="embeddings", index_params=index_params)
    collection.load()
