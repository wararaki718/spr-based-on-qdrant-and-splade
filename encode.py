import torch
from light_splade import SpladeEncoder
from qdrant_client import models


def encode_documents2points(encoder: SpladeEncoder, docs: list[str]) -> list[models.PointStruct]:
    token2id: dict[str, int] = encoder.tokenizer.get_vocab()

    # Encode documents to sparse vectors
    with torch.inference_mode():
        embeddings: torch.Tensor = encoder.encode(docs)
        sparse_vectors: list[dict[int, float]] = encoder.to_sparse(embeddings)

    # Prepare points for Qdrant upsert
    points = []
    for i, (doc, sparse_vector) in enumerate(zip(docs, sparse_vectors)):
        indices = []
        values = []
        for token, value in sparse_vector.items():
            indices.append(token2id[token])
            values.append(value)

        point = models.PointStruct(
            id=i,
            payload={"text": doc},
            vector={"text-sparse": models.SparseVector(indices=indices, values=values)},
        )
        points.append(point)

    return points


def encode_query2vector(encoder: SpladeEncoder, query: str) -> models.SparseVector:
    token2id: dict[str, int] = encoder.tokenizer.get_vocab()

    # Encode query to sparse vector
    with torch.inference_mode():
        query_embedding: torch.Tensor = encoder.encode([query])
        query_sparse_vector: dict[int, float] = encoder.to_sparse(query_embedding)[0]

    indices = []
    values = []
    for token, value in query_sparse_vector.items():
        indices.append(token2id[token])
        values.append(value)

    return models.SparseVector(indices=indices, values=values)
