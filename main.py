import time
import logging
from light_splade import SpladeEncoder
from qdrant_client import QdrantClient, models

from encode import encode_documents2points, encode_documents2points_batched, encode_query2vector, EncodingConfig
from utils import show_results

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(
    use_batched: bool = False,
    batch_size: int = 1000,
    max_workers: int = 4,
    qdrant_url: str = "http://localhost:6333",
) -> float:
    """
    Main pipeline with configurable encoding strategies.
    
    Args:
        use_batched: If True, use batched encoding for memory efficiency.
        batch_size: Size of each batch when use_batched=True
        max_workers: Number of parallel workers for encoding
        qdrant_url: URL of Qdrant server
        
    Returns:
        Elapsed time in seconds
    """
    start_time = time.time()
    
    try:
        # Initialize Qdrant client and SPLADE encoder
        logger.info("Initializing Qdrant client and SPLADE encoder...")
        client = QdrantClient(url=qdrant_url)
        encoder = SpladeEncoder(model_path="bizreach-inc/light-splade-japanese-28M")
        encoder.model_path = "bizreach-inc/light-splade-japanese-28M"  # For cache key

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
        client.create_collection(
            collection_name=collection_name,
            vectors_config={},
            sparse_vectors_config={
                "text-sparse": models.SparseVectorParams(),
            },
        )
        logger.info(f"Collection '{collection_name}' created successfully.")

        # Configure encoding
        config = EncodingConfig(
            batch_size=batch_size,
            max_workers=max_workers,
            use_parallel=len(docs) > 100,
        )

        # Upsert points with optional batching
        logger.info(f"Encoding {len(docs)} documents...")
        
        if use_batched and len(docs) > batch_size:
            # Stream large batches to Qdrant
            total_upserted = 0
            for batch_offset, batch_points in encode_documents2points_batched(
                encoder, docs, batch_size=batch_size
            ):
                if batch_points:
                    client.upsert(
                        collection_name=collection_name,
                        points=batch_points,
                    )
                    logger.info(f"Upserted {len(batch_points)} points (offset: {batch_offset})")
                    total_upserted += len(batch_points)
            logger.info(f"Total points upserted: {total_upserted}")
        else:
            # Standard single encoding with parallelization
            points = encode_documents2points(encoder, docs, config=config)
            if points:
                client.upsert(
                    collection_name=collection_name,
                    points=points,
                )
                logger.info(f"Upserted {len(points)} points")

        # Search points
        logger.info("Performing search...")
        query = "ベクトル検索の仕組み"
        query_sparse_vector = encode_query2vector(encoder, query)
        logger.info(f"Query: '{query}'")

        search_result: models.QueryResponse = client.query_points(
            collection_name=collection_name,
            query=query_sparse_vector,
            using="text-sparse",
            limit=3,
        )
        
        logger.info(f"Search returned {len(search_result.points)} results")
        show_results(search_result.points)

        # Cleanup
        logger.info(f"Deleting collection '{collection_name}'...")
        client.delete_collection(collection_name=collection_name)
        logger.info(f"Collection '{collection_name}' deleted successfully.")

        elapsed_time = time.time() - start_time
        logger.info(f"Pipeline completed successfully in {elapsed_time:.2f}s")
        
        return elapsed_time

    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Pipeline failed after {elapsed_time:.2f}s: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    elapsed = main()
    print(f"\n✅ DONE (Total elapsed: {elapsed:.2f}s)")
