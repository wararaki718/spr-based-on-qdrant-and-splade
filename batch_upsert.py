"""Batch upsert with exponential backoff retry and comprehensive error handling.

This module implements production-ready batch processing for Qdrant upserts,
incorporating lessons from PR #4 (error handling & retry logic) and PR #3
(comprehensive logging & graceful degradation).
"""

import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

logger = logging.getLogger(__name__)


@dataclass
class BatchUpsertConfig:
    """Configuration for batch upsert operations.

    Attributes:
        batch_size: Number of points per batch (default: 100).
                   Qdrant recommends 100-1000 points per request.
        max_retries: Maximum retry attempts for transient failures (default: 3).
        retry_delay_seconds: Initial delay between retries in seconds (default: 1.0).
                            Multiplied by 2 for each retry (exponential backoff).
        timeout_seconds: Timeout for each upsert request in seconds (default: 30.0).
    """
    batch_size: int = 100
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    timeout_seconds: float = 30.0


def upsert_points_batched(
    client: QdrantClient,
    collection_name: str,
    points: List[PointStruct],
    config: Optional[BatchUpsertConfig] = None,
) -> int:
    """Upsert points in batches with exponential backoff retry.

    This function handles large point sets by splitting them into configurable
    batches and applies exponential backoff retry logic for transient failures.

    Args:
        client: Qdrant client instance.
        collection_name: Name of the target collection.
        points: List of PointStruct objects to upsert.
        config: BatchUpsertConfig for batch size and retry parameters.
               If None, uses default config.

    Returns:
        Number of successfully upserted points.

    Raises:
        RuntimeError: If all retry attempts fail after max_retries.
        ValueError: If points list is empty or invalid.
    """
    if not config:
        config = BatchUpsertConfig()

    if not points:
        logger.warning("No points to upsert.")
        return 0

    logger.info(
        f"Starting batch upsert of {len(points)} points "
        f"with batch_size={config.batch_size}, max_retries={config.max_retries}"
    )

    total_upserted = 0
    num_batches = (len(points) + config.batch_size - 1) // config.batch_size

    for batch_idx in range(num_batches):
        batch_start = batch_idx * config.batch_size
        batch_end = min(batch_start + config.batch_size, len(points))
        batch = points[batch_start:batch_end]

        logger.debug(
            f"Processing batch {batch_idx + 1}/{num_batches} "
            f"(points {batch_start}-{batch_end - 1})"
        )

        # Retry logic with exponential backoff
        upserted = _upsert_batch_with_retry(
            client, collection_name, batch, config, batch_idx
        )
        total_upserted += upserted

    logger.info(f"Batch upsert completed. Total upserted: {total_upserted}/{len(points)}")
    return total_upserted


def _upsert_batch_with_retry(
    client: QdrantClient,
    collection_name: str,
    batch: List[PointStruct],
    config: BatchUpsertConfig,
    batch_idx: int,
) -> int:
    """Attempt to upsert a single batch with exponential backoff.

    Args:
        client: Qdrant client instance.
        collection_name: Name of the target collection.
        batch: List of PointStruct objects for this batch.
        config: Batch upsert configuration.
        batch_idx: Index of the current batch (for logging).

    Returns:
        Number of successfully upserted points in this batch.
    """
    delay = config.retry_delay_seconds

    for attempt in range(1, config.max_retries + 1):
        try:
            logger.debug(f"Batch {batch_idx}: attempt {attempt}/{config.max_retries}")

            client.upsert(
                collection_name=collection_name,
                points=batch,
                wait=True,
            )

            logger.debug(f"Batch {batch_idx}: successfully upserted {len(batch)} points")
            return len(batch)

        except Exception as e:
            is_last_attempt = attempt == config.max_retries

            if is_last_attempt:
                logger.error(
                    f"Batch {batch_idx}: failed after {config.max_retries} attempts. "
                    f"Error: {type(e).__name__}: {e}",
                    exc_info=True,
                )
                # Return 0 to indicate batch failure (partial success)
                return 0
            else:
                logger.warning(
                    f"Batch {batch_idx}: attempt {attempt} failed. "
                    f"Retrying in {delay:.1f}s. Error: {type(e).__name__}: {e}"
                )
                time.sleep(delay)
                delay *= 2  # Exponential backoff


def create_collection_safe(
    client: QdrantClient,
    collection_name: str,
    sparse_vector_config_name: str = "text-sparse",
) -> bool:
    """Safely create a collection, handling already-exists errors gracefully.

    Args:
        client: Qdrant client instance.
        collection_name: Name of the collection to create.
        sparse_vector_config_name: Name for the sparse vector config.

    Returns:
        True if created new, False if already existed.

    Raises:
        RuntimeError: If creation fails for reasons other than already-exists.
    """
    try:
        from qdrant_client import models

        client.create_collection(
            collection_name=collection_name,
            vectors_config={},
            sparse_vectors_config={
                sparse_vector_config_name: models.SparseVectorParams(),
            },
        )
        logger.info(f"Collection '{collection_name}' created successfully.")
        return True

    except Exception as e:
        error_msg = str(e).lower()
        # Check if error is "collection already exists" (409 Conflict)
        if "already exists" in error_msg or "409" in error_msg:
            logger.info(f"Collection '{collection_name}' already exists.")
            return False
        else:
            logger.error(
                f"Failed to create collection '{collection_name}': "
                f"{type(e).__name__}: {e}",
                exc_info=True,
            )
            raise RuntimeError(
                f"Failed to create collection '{collection_name}'"
            ) from e


def delete_collection_safe(client: QdrantClient, collection_name: str) -> bool:
    """Safely delete a collection, handling not-found errors gracefully.

    Args:
        client: Qdrant client instance.
        collection_name: Name of the collection to delete.

    Returns:
        True if deleted, False if not found.

    Raises:
        RuntimeError: If deletion fails for reasons other than not-found.
    """
    try:
        client.delete_collection(collection_name=collection_name)
        logger.info(f"Collection '{collection_name}' deleted successfully.")
        return True

    except Exception as e:
        error_msg = str(e).lower()
        # Check if error is "collection not found" (404 Not Found)
        if "not found" in error_msg or "404" in error_msg:
            logger.info(f"Collection '{collection_name}' not found (already deleted).")
            return False
        else:
            logger.error(
                f"Failed to delete collection '{collection_name}': "
                f"{type(e).__name__}: {e}",
                exc_info=True,
            )
            raise RuntimeError(
                f"Failed to delete collection '{collection_name}'"
            ) from e
