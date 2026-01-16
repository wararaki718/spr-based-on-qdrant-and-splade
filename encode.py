import torch
import threading
import logging
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from typing import Generator, Optional, Tuple, Dict
from dataclasses import dataclass
from light_splade import SpladeEncoder
from qdrant_client import models

logger = logging.getLogger(__name__)

# Thread-local storage for encoder instances (ensures thread safety)
_encoder_local = threading.local()
_token2id_cache: Dict[str, dict] = {}
_cache_lock = threading.Lock()


@dataclass
class EncodingConfig:
    """Configuration for document encoding operations."""
    batch_size: int = 1000
    max_workers: Optional[int] = None
    use_parallel: bool = True
    parallel_threshold: int = 100
    timeout_seconds: float = 300.0
    retry_attempts: int = 3


def get_or_create_thread_local_encoder(encoder: SpladeEncoder) -> SpladeEncoder:
    """
    Retrieve or create a thread-local encoder instance.
    Ensures thread safety by avoiding shared state in model inference.
    
    Args:
        encoder: The base encoder to clone for this thread
        
    Returns:
        Thread-local encoder instance
    """
    if not hasattr(_encoder_local, 'encoder'):
        _encoder_local.encoder = encoder
    return _encoder_local.encoder


def get_cached_token2id(model_path: str, tokenizer) -> dict[str, int]:
    """
    Retrieve cached token-to-ID mapping with proper cache key normalization.
    
    Args:
        model_path: Identifier for the model (e.g., "bizreach-inc/light-splade-japanese-28M")
        tokenizer: Tokenizer object
        
    Returns:
        Dictionary mapping tokens to IDs
    """
    with _cache_lock:
        if model_path not in _token2id_cache:
            _token2id_cache[model_path] = tokenizer.get_vocab()
            logger.debug(f"Cached token2id for model: {model_path}")
        return _token2id_cache[model_path]


def _sparse_vector_to_point(
    doc: str,
    sparse_vector: dict[str, float],
    token2id: dict[str, int],
    doc_id: int,
) -> models.PointStruct:
    """
    Convert a sparse vector representation to a Qdrant PointStruct.
    
    Args:
        doc: Original document text
        sparse_vector: Dictionary mapping token strings to weights
        token2id: Token to ID mapping dictionary
        doc_id: Unique document identifier
        
    Returns:
        PointStruct ready for Qdrant ingestion
        
    Raises:
        KeyError: If a token in sparse_vector is not found in token2id
    """
    try:
        indices = []
        values = []
        
        for token, weight in sparse_vector.items():
            if token not in token2id:
                logger.warning(f"Unknown token '{token}' in document {doc_id}, skipping")
                continue
            indices.append(token2id[token])
            values.append(float(weight))
        
        if not indices:
            logger.warning(f"Document {doc_id} has no valid sparse vectors")
            return models.PointStruct(
                id=doc_id,
                payload={"text": doc, "error": "no_valid_tokens"},
                vector={"text-sparse": models.SparseVector(indices=[], values=[])},
            )
        
        return models.PointStruct(
            id=doc_id,
            payload={"text": doc},
            vector={"text-sparse": models.SparseVector(indices=indices, values=values)},
        )
    except Exception as e:
        logger.error(f"Error converting sparse vector for doc {doc_id}: {e}")
        raise


def _encode_batch_documents(
    encoder: SpladeEncoder,
    docs: list[str],
    token2id: dict[str, int],
    id_offset: int = 0,
) -> list[models.PointStruct]:
    """
    Encode a batch of documents sequentially (within a single thread).
    
    Args:
        encoder: SpladeEncoder instance
        docs: List of documents to encode
        token2id: Token-to-ID mapping
        id_offset: Starting ID for documents in this batch
        
    Returns:
        List of PointStruct objects
    """
    points = []
    
    try:
        with torch.inference_mode():
            embeddings = encoder.encode(docs)
            sparse_vectors = encoder.to_sparse(embeddings)
        
        for i, (doc, sparse_vector) in enumerate(zip(docs, sparse_vectors)):
            try:
                point = _sparse_vector_to_point(doc, sparse_vector, token2id, id_offset + i)
                points.append(point)
            except Exception as e:
                logger.error(f"Failed to process document {id_offset + i}: {e}")
                # Continue processing other documents
                continue
        
        return points
    
    except Exception as e:
        logger.error(f"Batch encoding failed: {e}")
        raise


def encode_documents2points(
    encoder: SpladeEncoder,
    docs: list[str],
    config: Optional[EncodingConfig] = None,
) -> list[models.PointStruct]:
    """
    Encode documents to Qdrant points with smart parallelization.
    
    Implements order-preserving parallel processing with proper error handling.
    
    Args:
        encoder: SpladeEncoder instance
        docs: List of documents to encode
        config: Encoding configuration (uses defaults if None)
        
    Returns:
        List of PointStruct objects ready for Qdrant upsert
    """
    if config is None:
        config = EncodingConfig()
    
    if not docs:
        logger.warning("No documents provided for encoding")
        return []
    
    # Get model path for cache key normalization
    model_path = getattr(encoder, 'model_path', 'unknown')
    token2id = get_cached_token2id(model_path, encoder.tokenizer)
    
    # Decide on parallelization strategy
    should_parallelize = (
        config.use_parallel
        and len(docs) > config.parallel_threshold
        and config.max_workers is not None
    )
    
    if not should_parallelize or len(docs) <= config.parallel_threshold:
        logger.debug(f"Using sequential processing for {len(docs)} documents")
        return _encode_batch_documents(encoder, docs, token2id, id_offset=0)
    
    # Parallel processing with order preservation
    logger.debug(f"Using parallel processing with dynamic chunking for {len(docs)} documents")
    
    num_workers = config.max_workers or min(4, cpu_count())
    chunk_size = max(1, len(docs) // (num_workers * 2))  # Finer granularity chunks
    
    # Create chunks for parallel processing
    chunks = [
        (docs[i:i + chunk_size], i)
        for i in range(0, len(docs), chunk_size)
    ]
    
    points_dict: Dict[int, models.PointStruct] = {}
    
    try:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit tasks and map to their chunk index for order preservation
            future_to_chunk_idx = {
                executor.submit(_encode_batch_documents, encoder, chunk_docs, token2id, chunk_offset): chunk_idx
                for chunk_idx, (chunk_docs, chunk_offset) in enumerate(chunks)
            }
            
            # Collect results while preserving order
            for future in as_completed(future_to_chunk_idx, timeout=config.timeout_seconds):
                chunk_idx = future_to_chunk_idx[future]
                try:
                    chunk_points = future.result()
                    points_dict[chunk_idx] = chunk_points
                except Exception as e:
                    logger.error(f"Chunk {chunk_idx} failed: {e}")
                    raise
        
        # Reconstruct results in original order
        points = []
        for chunk_idx in sorted(points_dict.keys()):
            points.extend(points_dict[chunk_idx])
        
        logger.info(f"Successfully encoded {len(points)} documents")
        return points
    
    except Exception as e:
        logger.error(f"Parallel encoding failed: {e}")
        # Fallback to sequential processing
        logger.warning("Falling back to sequential processing")
        return _encode_batch_documents(encoder, docs, token2id, id_offset=0)


def encode_documents2points_batched(
    encoder: SpladeEncoder,
    docs: list[str],
    batch_size: int = 1000,
    id_offset: int = 0,
) -> Generator[Tuple[int, list[models.PointStruct]], None, None]:
    """
    Encode documents as memory-efficient batches with ID offset management.
    
    Yields batches while maintaining document ID uniqueness across calls.
    
    Args:
        encoder: SpladeEncoder instance
        docs: List of documents to encode
        batch_size: Number of documents per batch
        id_offset: Starting ID offset for documents (prevents ID collisions)
        
    Yields:
        Tuples of (batch_offset, list of PointStruct objects)
    """
    model_path = getattr(encoder, 'model_path', 'unknown')
    token2id = get_cached_token2id(model_path, encoder.tokenizer)
    
    if not docs:
        logger.warning("No documents provided for batched encoding")
        return
    
    for batch_start in range(0, len(docs), batch_size):
        batch_end = min(batch_start + batch_size, len(docs))
        batch_docs = docs[batch_start:batch_end]
        
        try:
            batch_offset = id_offset + batch_start
            batch_points = _encode_batch_documents(
                encoder, batch_docs, token2id, id_offset=batch_offset
            )
            logger.debug(f"Yielding batch {batch_start}-{batch_end} with offset {batch_offset}")
            yield batch_offset, batch_points
        
        except Exception as e:
            logger.error(f"Batch {batch_start}-{batch_end} encoding failed: {e}")
            raise


def encode_query2vector(encoder: SpladeEncoder, query: str) -> models.SparseVector:
    """
    Encode a query string to a sparse vector.
    
    Args:
        encoder: SpladeEncoder instance
        query: Query text to encode
        
    Returns:
        SparseVector ready for Qdrant search
        
    Raises:
        ValueError: If query is empty
        RuntimeError: If encoding fails
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    
    model_path = getattr(encoder, 'model_path', 'unknown')
    token2id = get_cached_token2id(model_path, encoder.tokenizer)
    
    try:
        with torch.inference_mode():
            query_embedding = encoder.encode([query])
            query_sparse_vector = encoder.to_sparse(query_embedding)[0]
        
        indices = []
        values = []
        
        for token, weight in query_sparse_vector.items():
            if token not in token2id:
                logger.debug(f"Unknown token '{token}' in query, skipping")
                continue
            indices.append(token2id[token])
            values.append(float(weight))
        
        logger.debug(f"Encoded query with {len(indices)} non-zero tokens")
        return models.SparseVector(indices=indices, values=values)
    
    except Exception as e:
        logger.error(f"Query encoding failed: {e}")
        raise RuntimeError(f"Failed to encode query: {e}") from e
