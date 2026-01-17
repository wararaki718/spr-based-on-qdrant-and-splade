import logging
import time
from typing import List, Optional
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

logger = logging.getLogger(__name__)


class BatchUpsertConfig:
    """Configuration for batch upsert operations."""
    def __init__(
        self,
        batch_size: int = 100,
        max_retries: int = 3,
        retry_delay_seconds: float = 1.0,
    ):
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds


def upsert_points_batched(
    client: QdrantClient,
    collection_name: str,
    points: List[models.PointStruct],
    config: Optional[BatchUpsertConfig] = None,
) -> int:
    """
    Upsert points to Qdrant in batches with error handling and retry logic.
    
    Args:
        client: QdrantClient instance
        collection_name: Name of the collection
        points: List of PointStruct objects to upsert
        config: BatchUpsertConfig (uses defaults if None)
        
    Returns:
        Number of successfully upserted points
    """
    if config is None:
        config = BatchUpsertConfig()
    
    if not points:
        logger.warning("No points to upsert")
        return 0
    
    total_upserted = 0
    failed_batches = []
    total_batches = (len(points) + config.batch_size - 1) // config.batch_size
    
    # Process points in batches
    for batch_idx, batch_start in enumerate(range(0, len(points), config.batch_size)):
        batch_end = min(batch_start + config.batch_size, len(points))
        batch_points = points[batch_start:batch_end]
        batch_num = batch_idx + 1
        
        logger.info(
            f"Batch {batch_num}/{total_batches} "
            f"(points {batch_start + 1}-{batch_end}/{len(points)})"
        )
        
        # Retry logic for each batch
        success = False
        for attempt in range(1, config.max_retries + 1):
            try:
                logger.debug(f"Batch {batch_num} - Attempt {attempt}/{config.max_retries}")
                
                client.upsert(
                    collection_name=collection_name,
                    points=batch_points,
                    wait=True,
                )
                
                batch_count = len(batch_points)
                total_upserted += batch_count
                logger.info(f"✓ Batch {batch_num} upserted ({batch_count} points)")
                success = True
                break
                
            except UnexpectedResponse as e:
                logger.warning(
                    f"Batch {batch_num} - Attempt {attempt}: "
                    f"Qdrant error: {e.status_code}"
                )
                if attempt < config.max_retries:
                    wait_time = config.retry_delay_seconds * attempt
                    time.sleep(wait_time)
                    
            except ConnectionError as e:
                logger.warning(f"Batch {batch_num} - Attempt {attempt}: Connection error")
                if attempt < config.max_retries:
                    wait_time = config.retry_delay_seconds * attempt
                    time.sleep(wait_time)
                    
            except Exception as e:
                logger.error(f"Batch {batch_num} - Attempt {attempt}: {type(e).__name__}")
                if attempt < config.max_retries:
                    wait_time = config.retry_delay_seconds * attempt
                    time.sleep(wait_time)
        
        if not success:
            logger.error(f"✗ Batch {batch_num} FAILED after {config.max_retries} retries")
            failed_batches.append(batch_num)
    
    logger.info(f"Upsert complete: {total_upserted}/{len(points)} points")
    if failed_batches:
        logger.error(f"Failed batches: {failed_batches}")
    
    return total_upserted
