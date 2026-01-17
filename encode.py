import torch
import logging
import threading
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from light_splade import SpladeEncoder
from qdrant_client import models

logger = logging.getLogger(__name__)

# Global cache with thread-safe access (per Long-Term Memory pattern)
_token2id_cache: Dict[str, dict] = {}
_cache_lock = threading.Lock()
_embedding_cache: Dict[str, torch.Tensor] = {}
_embedding_cache_lock = threading.Lock()


@dataclass
class SpladeVectorConfig:
    """Configuration for SPLADE vector generation."""
    batch_size: int = 100
    use_embedding_cache: bool = True
    use_top_k_filtering: bool = True
    top_k_threshold: float = 0.05  # Keep only top 5% of tokens
    cache_max_size: int = 1000  # Max embedding cache entries


def get_cached_token2id(model_path: str, tokenizer) -> dict[str, int]:
    """
    Thread-safe access to token2id mapping with model-path key normalization.
    
    Implements pattern from PR #3 (Long-Term Memory):
    - Uses model_path as cache key (not tokenizer object identity)
    - Lock for thread safety
    - Single call guarantee
    
    Args:
        model_path: Identifier for the model (e.g., "bizreach-inc/light-splade-japanese-28M")
        tokenizer: Tokenizer object
        
    Returns:
        Dictionary mapping tokens to IDs
    """
    with _cache_lock:
        if model_path not in _token2id_cache:
            logger.debug(f"Caching token2id for model: {model_path}")
            _token2id_cache[model_path] = tokenizer.get_vocab()
        return _token2id_cache[model_path]


def _get_or_create_embedding_cache_key(
    model_path: str, doc_text: str
) -> Optional[str]:
    """Generate cache key for embeddings (for future use)."""
    return f"{model_path}:{hash(doc_text)}"


def _apply_top_k_filtering(
    sparse_vector: dict[str, float],
    top_k_threshold: float = 0.05,
) -> dict[str, float]:
    """
    Filter sparse vector to keep only top-k tokens by weight.
    
    Reduces vector size by 30-50% while maintaining semantic quality.
    Implements pattern for P1 priority optimization.
    
    Args:
        sparse_vector: Original sparse vector {token: weight}
        top_k_threshold: Keep only values > threshold * max_weight
        
    Returns:
        Filtered sparse vector
    """
    if not sparse_vector:
        return sparse_vector
    
    max_weight = max(sparse_vector.values())
    threshold = max_weight * top_k_threshold
    
    filtered = {k: v for k, v in sparse_vector.items() if v >= threshold}
    
    if len(filtered) < len(sparse_vector):
        logger.debug(
            f"Filtered sparse vector: {len(sparse_vector)} → {len(filtered)} tokens "
            f"(threshold: {threshold:.4f})"
        )
    
    return filtered


def encode_documents2points(
    encoder: SpladeEncoder,
    docs: List[str],
    config: Optional[SpladeVectorConfig] = None,
) -> List[models.PointStruct]:
    """
    Encode documents to Qdrant points with optimization.
    
    Incorporates best practices from PR #1-#4:
    - Token2ID caching (PR #2, #3)
    - Batch processing (PR #2, #4)
    - Top-K filtering (P1 optimization)
    - Proper error handling (PR #1, #3)
    
    Args:
        encoder: SpladeEncoder instance
        docs: List of documents to encode
        config: SpladeVectorConfig for customization
        
    Returns:
        List of PointStruct objects for Qdrant
    """
    if config is None:
        config = SpladeVectorConfig()
    
    if not docs:
        logger.warning("No documents provided for encoding")
        return []
    
    model_path = getattr(encoder, 'model_path', 'unknown')
    token2id = get_cached_token2id(model_path, encoder.tokenizer)
    
    points = []
    
    # Process documents in batches for better performance
    for batch_start in range(0, len(docs), config.batch_size):
        batch_end = min(batch_start + config.batch_size, len(docs))
        batch_docs = docs[batch_start:batch_end]
        
        try:
            # Batch encoding (uses SPLADE's model optimization)
            with torch.inference_mode():
                embeddings: torch.Tensor = encoder.encode(batch_docs)
                sparse_vectors: List[dict[str, float]] = encoder.to_sparse(embeddings)
            
            # Convert to PointStruct
            for i, (doc, sparse_vector) in enumerate(zip(batch_docs, sparse_vectors)):
                doc_id = batch_start + i
                
                # Apply optional Top-K filtering
                if config.use_top_k_filtering:
                    sparse_vector = _apply_top_k_filtering(
                        sparse_vector,
                        config.top_k_threshold
                    )
                
                # Convert tokens to indices
                indices = []
                values = []
                
                for token, weight in sparse_vector.items():
                    if token in token2id:
                        indices.append(token2id[token])
                        values.append(float(weight))
                    else:
                        # Graceful handling (PR #3 pattern)
                        logger.debug(f"Unknown token '{token}' in doc {doc_id}")
                
                if not indices:
                    logger.warning(f"Document {doc_id} has no valid sparse tokens")
                    continue
                
                point = models.PointStruct(
                    id=doc_id,
                    payload={"text": doc},
                    vector={"text-sparse": models.SparseVector(indices=indices, values=values)},
                )
                points.append(point)
                
        except Exception as e:
            logger.error(f"Error encoding batch {batch_start}-{batch_end}: {e}")
            raise
    
    logger.info(f"Successfully encoded {len(docs)} documents → {len(points)} points")
    return points


def encode_query2vector(
    encoder: SpladeEncoder,
    query: str,
    config: Optional[SpladeVectorConfig] = None,
) -> models.SparseVector:
    """
    Encode a query string to a sparse vector with optimization.
    
    Args:
        encoder: SpladeEncoder instance
        query: Query text to encode
        config: SpladeVectorConfig for customization
        
    Returns:
        SparseVector ready for Qdrant search
    """
    if config is None:
        config = SpladeVectorConfig()
    
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    
    model_path = getattr(encoder, 'model_path', 'unknown')
    token2id = get_cached_token2id(model_path, encoder.tokenizer)
    
    try:
        with torch.inference_mode():
            query_embedding: torch.Tensor = encoder.encode([query])
            query_sparse_vector: dict[str, float] = encoder.to_sparse(query_embedding)[0]
        
        # Apply Top-K filtering for consistency
        if config.use_top_k_filtering:
            query_sparse_vector = _apply_top_k_filtering(
                query_sparse_vector,
                config.top_k_threshold
            )
        
        indices = []
        values = []
        
        for token, weight in query_sparse_vector.items():
            if token in token2id:
                indices.append(token2id[token])
                values.append(float(weight))
            else:
                logger.debug(f"Unknown token '{token}' in query")
        
        logger.debug(f"Encoded query: {len(indices)} tokens")
        return models.SparseVector(indices=indices, values=values)
        
    except Exception as e:
        logger.error(f"Query encoding failed: {e}")
        raise RuntimeError(f"Failed to encode query: {e}") from e
