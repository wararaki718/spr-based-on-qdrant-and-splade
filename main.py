"""Main pipeline for SPLADE + Qdrant with production-ready error handling.

This module demonstrates batch upsert with exponential backoff retry logic,
drawing from PR #4 best practices for reliability and PR #3 logging patterns.
"""

import logging
import sys
from typing import List

from light_splade import SpladeEncoder
from qdrant_client import QdrantClient

from batch_upsert import (
    BatchUpsertConfig,
    create_collection_safe,
    delete_collection_safe,
    upsert_points_batched,
)
from encode import encode_documents2points, encode_query2vector
from utils import show_results

# Configure logging for observability (from PR #3)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main(
    qdrant_url: str = "http://localhost:6333",
    collection_name: str = "sparse_splade_collection",
    batch_size: int = 100,
    max_retries: int = 3,
) -> None:
    """Main pipeline for SPLADE + Qdrant batch processing.

    Args:
        qdrant_url: URL of the Qdrant server.
        collection_name: Name of the collection to create.
        batch_size: Points per batch for upsert.
        max_retries: Maximum retry attempts for transient failures.
    """
    logger.info("=" * 80)
    logger.info("Starting SPLADE + Qdrant Pipeline (Production-Ready)")
    logger.info("=" * 80)

    # Initialize clients
    try:
        logger.info(f"Connecting to Qdrant at {qdrant_url}...")
        client = QdrantClient(url=qdrant_url)
        logger.info("✅ Qdrant connection successful")
    except Exception as e:
        logger.error(f"❌ Failed to connect to Qdrant: {e}", exc_info=True)
        sys.exit(1)

    try:
        logger.info("Loading SPLADE encoder...")
        encoder = SpladeEncoder(model_path="bizreach-inc/light-splade-japanese-28M")
        logger.info("✅ SPLADE encoder loaded")
    except Exception as e:
        logger.error(f"❌ Failed to load SPLADE encoder: {e}", exc_info=True)
        sys.exit(1)

    # Sample documents
    docs: List[str] = [
        "Qdrantは高速なベクトル検索エンジンです",
        "SPLADEはスパース表現を学習します",
        "ベクトル検索の仕組みを理解しましょう",
        "Pythonでベクトル検索エンジンを構築します",
        "QdrantとSPLADEを組み合わせて使います",
    ]

    logger.info(f"Processing {len(docs)} documents for indexing...")

    # Step 1: Create collection (with graceful already-exists handling)
    try:
        created = create_collection_safe(
            client, collection_name, sparse_vector_config_name="text-sparse"
        )
        if created:
            logger.info(f"✅ Created new collection '{collection_name}'")
        else:
            logger.info(f"⚠️  Collection '{collection_name}' already exists (will reuse)")
    except RuntimeError as e:
        logger.error(f"❌ {e}", exc_info=True)
        sys.exit(1)

    # Step 2: Encode documents
    try:
        logger.info("Encoding documents to sparse vectors...")
        points = encode_documents2points(encoder, docs)
        logger.info(f"✅ Encoded {len(points)} points")
    except Exception as e:
        logger.error(f"❌ Failed to encode documents: {e}", exc_info=True)
        sys.exit(1)

    # Step 3: Batch upsert with exponential backoff (PR #4 integration)
    try:
        logger.info(f"Starting batch upsert (batch_size={batch_size})...")
        config = BatchUpsertConfig(
            batch_size=batch_size,
            max_retries=max_retries,
            retry_delay_seconds=1.0,
        )
        upserted_count = upsert_points_batched(
            client, collection_name, points, config
        )

        if upserted_count == len(points):
            logger.info(f"✅ All {upserted_count} points upserted successfully")
        else:
            logger.warning(
                f"⚠️  Partial success: {upserted_count}/{len(points)} points upserted"
            )
    except Exception as e:
        logger.error(f"❌ Batch upsert failed: {e}", exc_info=True)
        sys.exit(1)

    # Step 4: Search (verify indexing worked)
    try:
        query = "ベクトル検索の仕組み"
        logger.info(f"Performing search query: '{query}'")

        query_sparse_vector = encode_query2vector(encoder, query)
        logger.info("✅ Query encoded")

        search_result = client.query_points(
            collection_name=collection_name,
            query=query_sparse_vector,
            using="text-sparse",
            limit=3,
        )
        logger.info(f"✅ Found {len(search_result.points)} search results")

        if search_result.points:
            logger.info("\n📊 Top Search Results:")
            show_results(search_result.points)
        else:
            logger.warning("No search results found")

    except Exception as e:
        logger.error(f"❌ Search failed: {e}", exc_info=True)
        sys.exit(1)

    # Step 5: Cleanup (with graceful not-found handling)
    try:
        logger.info(f"Cleaning up collection '{collection_name}'...")
        deleted = delete_collection_safe(client, collection_name)
        if deleted:
            logger.info(f"✅ Deleted collection '{collection_name}'")
        else:
            logger.warning(f"⚠️  Collection '{collection_name}' not found (already deleted)")
    except RuntimeError as e:
        logger.error(f"❌ {e}", exc_info=True)
        # Don't exit here - cleanup failure is not critical

    logger.info("=" * 80)
    logger.info("✅ Pipeline completed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
