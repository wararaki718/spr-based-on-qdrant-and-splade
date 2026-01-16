import torch
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from typing import Generator
from light_splade import SpladeEncoder
from qdrant_client import models


@lru_cache(maxsize=1)
def get_token2id(tokenizer) -> dict[str, int]:
    """
    Cached token to ID mapping retrieval.
    Avoids redundant vocab lookups when called multiple times.
    """
    return tokenizer.get_vocab()


def _encode_single_doc(
    encoder: SpladeEncoder,
    doc: str,
    token2id: dict[str, int],
    doc_id: int,
) -> models.PointStruct:
    """
    Encode a single document to a Qdrant PointStruct.
    Designed for use in parallel processing.
    """
    with torch.inference_mode():
        embedding = encoder.encode([doc])
        sparse_vector = encoder.to_sparse(embedding)[0]

    indices = [token2id[token] for token in sparse_vector.keys()]
    values = list(sparse_vector.values())

    return models.PointStruct(
        id=doc_id,
        payload={"text": doc},
        vector={"text-sparse": models.SparseVector(indices=indices, values=values)},
    )


def encode_documents2points(
    encoder: SpladeEncoder,
    docs: list[str],
    use_parallel: bool = True,
) -> list[models.PointStruct]:
    """
    Encode documents to Qdrant points with optional parallel processing.
    
    Args:
        encoder: SpladeEncoder instance
        docs: List of documents to encode
        use_parallel: Enable parallel processing for large batches (> 100 docs)
    
    Returns:
        List of PointStruct objects ready for Qdrant upsert
    """
    token2id = get_token2id(encoder.tokenizer)
    
    # Use parallel processing for large batches to amortize thread overhead
    if use_parallel and len(docs) > 100:
        num_workers = min(4, cpu_count())
        points = []
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(_encode_single_doc, encoder, doc, token2id, i)
                for i, doc in enumerate(docs)
            ]
            points = [f.result() for f in futures]
        
        return points
    
    # Sequential processing for small batches
    with torch.inference_mode():
        embeddings = encoder.encode(docs)
        sparse_vectors = encoder.to_sparse(embeddings)
    
    points = []
    for i, (doc, sparse_vector) in enumerate(zip(docs, sparse_vectors)):
        indices = [token2id[token] for token in sparse_vector.keys()]
        values = list(sparse_vector.values())
        
        point = models.PointStruct(
            id=i,
            payload={"text": doc},
            vector={"text-sparse": models.SparseVector(indices=indices, values=values)},
        )
        points.append(point)
    
    return points


def encode_documents2points_batched(
    encoder: SpladeEncoder,
    docs: list[str],
    batch_size: int = 1000,
) -> Generator[list[models.PointStruct], None, None]:
    """
    Encode documents as batches for memory-efficient processing.
    Yields batches of points for chunked Qdrant upsert operations.
    
    Args:
        encoder: SpladeEncoder instance
        docs: List of documents to encode
        batch_size: Number of documents per batch
    
    Yields:
        Lists of PointStruct objects
    """
    token2id = get_token2id(encoder.tokenizer)
    
    for batch_start in range(0, len(docs), batch_size):
        batch_end = min(batch_start + batch_size, len(docs))
        batch_docs = docs[batch_start:batch_end]
        
        with torch.inference_mode():
            embeddings = encoder.encode(batch_docs)
            sparse_vectors = encoder.to_sparse(embeddings)
        
        points = []
        for i, (doc, sparse_vector) in enumerate(zip(batch_docs, sparse_vectors)):
            indices = [token2id[token] for token in sparse_vector.keys()]
            values = list(sparse_vector.values())
            
            point = models.PointStruct(
                id=batch_start + i,
                payload={"text": doc},
                vector={"text-sparse": models.SparseVector(indices=indices, values=values)},
            )
            points.append(point)
        
        yield points


def encode_query2vector(encoder: SpladeEncoder, query: str) -> models.SparseVector:
    """
    Encode a query string to a sparse vector.
    Uses cached token2id for efficient token mapping.
    """
    token2id = get_token2id(encoder.tokenizer)
    
    with torch.inference_mode():
        query_embedding = encoder.encode([query])
        query_sparse_vector = encoder.to_sparse(query_embedding)[0]
    
    indices = [token2id[token] for token in query_sparse_vector.keys()]
    values = list(query_sparse_vector.values())
    
    return models.SparseVector(indices=indices, values=values)
