import logging
from light_splade import SpladeEncoder
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from encode import encode_documents2points, encode_query2vector
from utils import show_results
from batch_upsert import upsert_points_batched, BatchUpsertConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Main application pipeline with batch upsert support."""
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
            if e.status_code == 409:
                logger.info(f"Collection '{collection_name}' already exists")
            else:
                raise

        # Encode documents
        logger.info(f"Encoding {len(docs)} documents...")
        points = encode_documents2points(encoder, docs)
        logger.info(f"✓ Encoded {len(points)} documents")

        # Upsert points with batch processing
        logger.info("\nStarting batch upsert operation...")
        config = BatchUpsertConfig(batch_size=100, max_retries=3)
        upserted_count = upsert_points_batched(client, collection_name, points, config)
        
        if upserted_count == 0:
            logger.error("Failed to upsert any documents")
            return

        # Search points
        logger.info("\nExecuting search query...")
        query = "ベクトル検索の仕組み"
        query_sparse_vector = encode_query2vector(encoder, query)
        logger.info(f"Query: '{query}'\n")

        search_result: models.QueryResponse = client.query_points(
            collection_name=collection_name,
            query=query_sparse_vector,
            using="text-sparse",
            limit=3,
        )
        
        if search_result.points:
            show_results(search_result.points)
        else:
            logger.warning("No results found for the query")

        # Delete collection
        logger.info(f"Deleting collection '{collection_name}'...")
        client.delete_collection(collection_name=collection_name)
        logger.info(f"✓ Collection deleted")
        logger.info("DONE")

    except Exception as e:
        logger.error(f"Error: {type(e).__name__}: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
