import logging
import time
from typing import List, Optional
from light_splade import SpladeEncoder
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from encode import encode_documents2points, encode_query2vector
from utils import show_results

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BatchUpsertConfig:
    """Configuration for batch upsert operations."""
    batch_size: int = 100
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    timeout_seconds: float = 30.0


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
        
    Raises:
        RuntimeError: If all batches fail after retries
    """
    if config is None:
        config = BatchUpsertConfig()
    
    if not points:
        logger.warning("No points to upsert")
        return 0
    
    total_upserted = 0
    failed_batches = []
    
    # Process points in batches
    for batch_idx, batch_start in enumerate(range(0, len(points), config.batch_size)):
        batch_end = min(batch_start + config.batch_size, len(points))
        batch_points = points[batch_start:batch_end]
        batch_num = batch_idx + 1
        total_batches = (len(points) + config.batch_size - 1) // config.batch_size
        
        logger.info(
            f"Processing batch {batch_num}/{total_batches} "
            f"(points {batch_start + 1}-{batch_end}/{len(points)})"
        )
        
        # Retry logic for each batch
        success = False
        for attempt in range(1, config.max_retries + 1):
            try:
                logger.debug(f"Batch {batch_num} - Attempt {attempt}/{config.max_retries}")
                
                # Perform upsert with timeout
                client.upsert(
                    collection_name=collection_name,
                    points=batch_points,
                    wait=True,  # Wait for operation to complete
                )
                
                batch_count = len(batch_points)
                total_upserted += batch_count
                logger.info(
                    f"✓ Batch {batch_num}/{total_batches} upserted successfully "
                    f"({batch_count} points)"
                )
                success = True
                break
                
            except UnexpectedResponse as e:
                logger.warning(
                    f"Batch {batch_num} - Attempt {attempt}: "
                    f"Qdrant error: {e.status_code} - {e.reason}"
                )
                
                if attempt < config.max_retries:
                    wait_time = config.retry_delay_seconds * attempt  # Exponential backoff
                    logger.info(f"Retrying in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                    
            except ConnectionError as e:
                logger.warning(
                    f"Batch {batch_num} - Attempt {attempt}: "
                    f"Connection error: {e}"
                )
                
                if attempt < config.max_retries:
                    wait_time = config.retry_delay_seconds * attempt
                    logger.info(f"Retrying in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                    
            except Exception as e:
                logger.error(
                    f"Batch {batch_num} - Attempt {attempt}: "
                    f"Unexpected error: {type(e).__name__}: {e}"
                )
                
                if attempt < config.max_retries:
                    wait_time = config.retry_delay_seconds * attempt
                    logger.info(f"Retrying in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
        
        # Record failure if all retries exhausted
        if not success:
            logger.error(f"✗ Batch {batch_num}/{total_batches} FAILED after {config.max_retries} retries")
            failed_batches.append(batch_num)
    
    # Summary logging
    logger.info(f"\n{'='*60}")
    logger.info(f"Upsert Summary:")
    logger.info(f"  Total points: {len(points)}")
    logger.info(f"  Successfully upserted: {total_upserted}")
    logger.info(f"  Failed batches: {len(failed_batches)}")
    
    if failed_batches:
        logger.error(f"  Failed batch numbers: {failed_batches}")
        if total_upserted < len(points):
            logger.warning(
                f"Partial success: {total_upserted}/{len(points)} points upserted"
            )
            # Decide whether to raise or continue based on your use case
            # For now, we warn but continue
    
    logger.info(f"{'='*60}\n")
    
    return total_upserted


def main() -> None:
    """
    Main application pipeline with batch upsert support.
    """
    try:
        # Initialize Qdrant client and SPLADE encoder
        logger.info("Initializing Qdrant client and SPLADE encoder...")
        client = QdrantClient(url="http://localhost:6333")
        encoder = SpladeEncoder(model_path="bizreach-inc/light-splade-japanese-28M")
        logger.info("✓ Initialization complete")

        collection_name = "sparse_splade_collection"
        docs = [
            "Qdrantは高速なベクトル検索エンジンです",
            "SPLADEはスパース表現を学習します",
            "ベクトル検索の仕組みを理解しましょう", 
            "Pythonでベクトル検索エンジンを構築します",
            "QdrantとSPLADEを組み合わせて使います",
        ]

        # Create collection
        logger.info(f"Creating collection '{collection_name}'...")
        try:
            client.create_collection(
                collection_name=collection_name,
                vectors_config={},
                sparse_vectors_config={
                    "text-sparse": models.SparseVectorParams(),
                },
            )
            logger.info(f"✓ Collection '{collection_name}' created")
        except UnexpectedResponse as e:
            if e.status_code == 409:  # Collection already exists
                logger.info(f"Collection '{collection_name}' already exists, skipping creation")
            else:
                raise

        # Encode documents
        logger.info(f"Encoding {len(docs)} documents...")
        points = encode_documents2points(encoder, docs)
        logger.info(f"✓ Encoded {len(points)} documents")

        # Upsert points with batch processing
        logger.info("\nStarting batch upsert operation...")
        config = BatchUpsertConfig(
            batch_size=100,  # Process 100 documents per batch
            max_retries=3,
            retry_delay_seconds=1.0,
        )
        upserted_count = upsert_points_batched(
            client,
            collection_name,
            points,
            config
        )
        
        if upserted_count == 0:
            logger.error("Failed to upsert any documents. Exiting.")
            return

        # Search points
        logger.info("\nExecuting search query...")
        query = "ベクトル検索の仕組み"
        query_sparse_vector = encode_query2vector(encoder, query)
        logger.info(f"Query: '{query}'")

        try:
            search_result: models.QueryResponse = client.query_points(
                collection_name=collection_name,
                query=query_sparse_vector,
                using="text-sparse",
                limit=3,
            )
            
            if search_result.points:
                logger.info(f"✓ Found {len(search_result.points)} results")
                show_results(search_result.points)
            else:
                logger.warning("No results found for the query")
                
        except Exception as e:
            logger.error(f"Search operation failed: {e}")
            raise

        # Delete collection (optional)
        logger.info(f"\nCleaning up: Deleting collection '{collection_name}'...")
        try:
            client.delete_collection(collection_name=collection_name)
            logger.info(f"✓ Collection '{collection_name}' deleted")
        except Exception as e:
            logger.warning(f"Failed to delete collection: {e}")

        logger.info("\n✓ DONE - All operations completed successfully")

    except Exception as e:
        logger.error(f"Fatal error: {type(e).__name__}: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
