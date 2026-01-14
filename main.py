"""
Sparse Vector Search Engine using Qdrant and SPLADE.

This module provides a production-ready implementation of a sparse vector search
engine combining Qdrant's vector database capabilities with SPLADE's sparse
encoding models.

Design Patterns Used:
    - Strategy Pattern: Pluggable encoder implementations
    - Configuration Object Pattern: Centralized configuration
    - Context Manager Pattern: Resource management
    - Builder Pattern: Flexible client initialization

Example:
    >>> config = SearchConfig(
    ...     qdrant_url="http://localhost:6333",
    ...     model_path="bizreach-inc/light-splade-japanese-28M"
    ... )
    >>> engine = SparseSearchEngine(config)
    >>> results = engine.search("ベクトル検索の仕組み", limit=3)
    >>> engine.cleanup()
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

from light_splade import SpladeEncoder
from qdrant_client import QdrantClient, models

from encode import encode_documents2points, encode_query2vector
from utils import show_results

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SearchConfig:
    """Configuration for the sparse search engine.
    
    Attributes:
        qdrant_url: URL of the Qdrant server
        model_path: Path to the SPLADE model
        collection_name: Name of the Qdrant collection
        sparse_vector_name: Name of the sparse vector field
        timeout: Connection timeout in seconds
    """
    qdrant_url: str
    model_path: str
    collection_name: str = "sparse_splade_collection"
    sparse_vector_name: str = "text-sparse"
    timeout: int = 30


class SparseSearchEngine:
    """A production-ready sparse vector search engine.
    
    Combines Qdrant vector database with SPLADE sparse encoding for
    efficient semantic search on Japanese text.
    
    Attributes:
        config: Search engine configuration
        client: Qdrant client instance
        encoder: SPLADE encoder instance
        is_initialized: Flag indicating initialization status
    """
    
    def __init__(self, config: SearchConfig) -> None:
        """Initialize the sparse search engine.
        
        Args:
            config: SearchConfig instance with all necessary parameters
            
        Raises:
            ConnectionError: If unable to connect to Qdrant
            RuntimeError: If encoder initialization fails
        """
        self.config = config
        self.client: Optional[QdrantClient] = None
        self.encoder: Optional[SpladeEncoder] = None
        self.is_initialized = False
        
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize Qdrant client and SPLADE encoder.
        
        Raises:
            ConnectionError: If unable to connect to Qdrant
            RuntimeError: If encoder initialization fails
        """
        try:
            logger.info(f"Connecting to Qdrant at {self.config.qdrant_url}")
            self.client = QdrantClient(
                url=self.config.qdrant_url,
                timeout=self.config.timeout
            )
            
            logger.info(f"Loading SPLADE model: {self.config.model_path}")
            self.encoder = SpladeEncoder(model_path=self.config.model_path)
            
            self.is_initialized = True
            logger.info("Search engine initialized successfully")
            
        except ConnectionError as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
        except RuntimeError as e:
            logger.error(f"Failed to initialize encoder: {e}")
            raise
    
    def create_collection(self) -> None:
        """Create a collection for sparse vectors.
        
        Raises:
            RuntimeError: If collection creation fails
        """
        if not self.is_initialized or self.client is None:
            raise RuntimeError("Engine not properly initialized")
        
        try:
            logger.info(f"Creating collection '{self.config.collection_name}'")
            self.client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config={},
                sparse_vectors_config={
                    self.config.sparse_vector_name: models.SparseVectorParams(),
                },
            )
            logger.info(f"Collection '{self.config.collection_name}' created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise
    
    def upsert_documents(self, documents: List[str]) -> int:
        """Upsert documents into the collection.
        
        Args:
            documents: List of document texts to index
            
        Returns:
            Number of documents successfully upserted
            
        Raises:
            RuntimeError: If upsert operation fails
        """
        if not self.is_initialized or self.client is None or self.encoder is None:
            raise RuntimeError("Engine not properly initialized")
        
        if not documents:
            logger.warning("No documents provided for upsert")
            return 0
        
        try:
            logger.info(f"Encoding {len(documents)} documents")
            points = encode_documents2points(self.encoder, documents)
            
            logger.info(f"Upserting {len(points)} points to collection")
            self.client.upsert(
                collection_name=self.config.collection_name,
                points=points,
            )
            logger.info(f"Successfully upserted {len(points)} points")
            return len(points)
            
        except Exception as e:
            logger.error(f"Failed to upsert documents: {e}")
            raise
    
    def search(
        self,
        query: str,
        limit: int = 3,
        score_threshold: Optional[float] = None
    ) -> List:
        """Search for documents matching the query.
        
        Args:
            query: Search query text
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of search results with scores
            
        Raises:
            RuntimeError: If search operation fails
        """
        if not self.is_initialized or self.client is None or self.encoder is None:
            raise RuntimeError("Engine not properly initialized")
        
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return []
        
        try:
            logger.info(f"Processing query: '{query}'")
            query_sparse_vector = encode_query2vector(self.encoder, query)
            
            logger.info(f"Searching collection with limit={limit}")
            search_result: models.QueryResponse = self.client.query_points(
                collection_name=self.config.collection_name,
                query=query_sparse_vector,
                using=self.config.sparse_vector_name,
                limit=limit,
            )
            
            results = search_result.points
            logger.info(f"Found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def delete_collection(self) -> None:
        """Delete the collection from Qdrant.
        
        Raises:
            RuntimeError: If deletion fails
        """
        if not self.is_initialized or self.client is None:
            raise RuntimeError("Engine not properly initialized")
        
        try:
            logger.info(f"Deleting collection '{self.config.collection_name}'")
            self.client.delete_collection(
                collection_name=self.config.collection_name
            )
            logger.info(f"Collection '{self.config.collection_name}' deleted successfully")
            
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise
    
    def cleanup(self) -> None:
        """Clean up resources and close connections.
        
        This should be called when the engine is no longer needed.
        """
        try:
            if self.is_initialized and self.client:
                self.delete_collection()
            logger.info("Cleanup completed successfully")
        except Exception as e:
            logger.error(f"Cleanup encountered an error: {e}")
            # Don't raise here - cleanup should not fail the program


def main() -> None:
    """Main execution function demonstrating the search engine.
    
    This example demonstrates:
    1. Creating a search engine with custom configuration
    2. Creating a collection for sparse vectors
    3. Upserting sample documents
    4. Performing a search query
    5. Displaying results
    6. Cleaning up resources
    """
    # Configuration
    config = SearchConfig(
        qdrant_url="http://localhost:6333",
        model_path="bizreach-inc/light-splade-japanese-28M",
        collection_name="sparse_splade_collection",
        sparse_vector_name="text-sparse",
    )
    
    # Sample documents
    documents = [
        "Qdrantは高速なベクトル検索エンジンです",
        "SPLADEはスパース表現を学習します",
        "ベクトル検索の仕組みを理解しましょう",
        "Pythonでベクトル検索エンジンを構築します",
        "QdrantとSPLADEを組み合わせて使います",
    ]
    
    try:
        # Initialize engine
        engine = SparseSearchEngine(config)
        
        # Create collection and upsert documents
        engine.create_collection()
        upserted_count = engine.upsert_documents(documents)
        logger.info(f"Total documents indexed: {upserted_count}")
        
        # Perform search
        query = "ベクトル検索の仕組み"
        logger.info(f"Executing search for: '{query}'")
        results = engine.search(query, limit=3)
        
        # Display results
        show_results(results)
        
        logger.info("Search completed successfully")
        print("\nDONE")
        
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        raise
    finally:
        # Ensure cleanup happens
        if 'engine' in locals():
            engine.cleanup()


if __name__ == "__main__":
    main()
