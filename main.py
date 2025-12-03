import torch
from light_splade import SpladeEncoder
from qdrant_client import QdrantClient, models


def main() -> None:
    encoder = SpladeEncoder(model_path="bizreach-inc/light-splade-japanese-28M")
    token2id: dict[str, int] = encoder.tokenizer.get_vocab()

    corpus = [
        "日本の首都は東京です。",
        "大阪万博は2025年に開催されます。"
    ]

    with torch.inference_mode():
        embeddings: torch.Tensor = encoder.encode(corpus)
        sparse_vectors: list[dict[int, float]] = encoder.to_sparse(embeddings)
    
    # for sparse_vector in sparse_vectors:
    #     indices = []
    #     vectors = []
    #     for token, value in sparse_vector.items()
    #         indices.append(token2id[token])
    #         vectors.append(value)
    
    # https://qdrant.tech/documentation/concepts/collections/#collection-with-sparse-vectors
    print("client:")
    client = QdrantClient(url="http://localhost:6333")
    collection_name = "sparse_splade_collection"
    client.create_collection(
        collection_name=collection_name,
        vectors_config={},
        sparse_vectors_config={
            "text": models.SparseVectorParams(),
        },
    )

    # https://qdrant.tech/documentation/concepts/collections/#collection-with-sparse-vectors
    print("upsert:")
    points = []
    for i, sparse_vector in enumerate(sparse_vectors):
        indices = []
        values = []
        for token, value in sparse_vector.items():
            indices.append(token2id[token])
            values.append(value)
        
        point = models.PointStruct(
            id=i,
            vector={"text": models.SparseVector(indices=indices, values=values)},
        )
        points.append(point)
    
    # upsert
    client.upsert(
        collection_name=collection_name,
        points=points,
    )

    # https://qdrant.tech/documentation/concepts/search/#search-api
    print("search:")
    query = "日本の首都は？"
    with torch.inference_mode():
        query_embedding: torch.Tensor = encoder.encode([query])
        query_sparse_vector: dict[int, float] = encoder.to_sparse(query_embedding)[0]

    indices = []
    values = []
    for token, value in query_sparse_vector.items():
        indices.append(token2id[token])
        values.append(value)

    # get search results
    search_result: models.QueryResponse = client.query_points(
        collection_name=collection_name,
        query=models.SparseVector(indices=indices, values=values),
        using="text",
    )
    print("# show result #")
    points: list[models.ScoredPoint] = search_result.points
    print(points[0])
    print(points[1])
    print("###############")

    # delete collection
    print("delete collection:")
    client.delete_collection(collection_name=collection_name)

    print("DONE")


if __name__ == "__main__":
    main()
