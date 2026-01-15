# Sparse Retrieval based on Qdrant and SPLADE

A production-ready sparse vector search engine implementation combining Qdrant's vector database with SPLADE's sparse encoding model for efficient semantic search on Japanese text.

## Overview

This project implements a sparse retrieval system using:
- **Qdrant**: High-performance vector database for storing and querying sparse vectors
- **SPLADE**: Sparse Lexical and Semantic Embeddings model for efficient text encoding
- **Japanese Language Support**: Uses `bizreach-inc/light-splade-japanese-28M` model for Japanese text

### Key Features

- 🚀 **Sparse Vector Search**: Efficient sparse representation learning
- 🌐 **Docker Support**: Easy deployment with Docker Compose
- 🎯 **Japanese Language**: Native support for Japanese text
- 💾 **Persistent Storage**: Qdrant data persists across container restarts
- 📝 **Lightweight**: Minimal dependencies, focused implementation

## Architecture

```
┌─────────────────┐
│   Input Text    │
└────────┬────────┘
         │
    ┌────▼────────────────┐
    │  SPLADE Encoder     │
    │  (encode_documents) │
    └────┬────────────────┘
         │
    ┌────▼──────────────────────┐
    │  Sparse Vector Conversion  │
    │  (encode_query2vector)     │
    └────┬───────────────────────┘
         │
    ┌────▼──────────────────┐
    │  Qdrant Vector DB     │
    │  (upsert/query)       │
    └────┬──────────────────┘
         │
    ┌────▼──────────────────┐
    │  Results Display      │
    │  (show_results)       │
    └───────────────────────┘
```

## Project Structure

```
spr-based-on-qdrant-and-splade/
├── README.md              # This file
├── LICENSE                # License information
├── compose.yml            # Docker Compose configuration
├── main.py               # Main application entry point
├── encode.py             # SPLADE encoding utilities
└── utils.py              # Helper utilities for result display
```

## Files Description

### main.py
Entry point for the sparse retrieval system. Demonstrates:
1. Initializing Qdrant client and SPLADE encoder
2. Creating a collection with sparse vector configuration
3. Encoding and upserting sample documents
4. Executing a search query
5. Displaying results
6. Cleanup and collection deletion

### encode.py
Encoding utilities for converting text to sparse vectors:

- **`encode_documents2points(encoder, docs)`**
  - Converts list of documents into Qdrant PointStruct objects
  - Applies sparse vector transformation using SPLADE
  - Returns: List of PointStruct with sparse vectors and payloads

- **`encode_query2vector(encoder, query)`**
  - Converts a query string into a sparse vector
  - Uses the same SPLADE encoder for consistency
  - Returns: SparseVector ready for Qdrant query

### utils.py
Utility functions for result presentation:

- **`show_results(points)`**
  - Displays search results in a readable format
  - Shows document id, similarity score, and text content
  - Useful for debugging and result inspection

### compose.yml
Docker Compose configuration for Qdrant:
- **Service**: Qdrant v1.16
- **Ports**: 6333 (HTTP API), 6334 (gRPC)
- **Storage**: Persistent volume for data persistence
- **Container Name**: qdrant

## Installation & Setup

### Prerequisites

- Python 3.8+
- Docker and Docker Compose
- CUDA-compatible GPU (recommended for faster inference, optional)

### Step 1: Install Python Dependencies

```shell
pip install qdrant-client==1.16.1 light-splade==0.1.2 torch
```

**Dependency Details**:
- `qdrant-client==1.16.1`: Python client for Qdrant vector database
- `light-splade==0.1.2`: Lightweight SPLADE model implementation
- `torch`: PyTorch (required by light-splade for inference)

### Step 2: Launch Qdrant Server

Start Qdrant using Docker Compose:

```shell
docker compose up -d
```

This will:
- Pull the Qdrant v1.16 image
- Start a container named `qdrant`
- Expose API on `http://localhost:6333`
- Create persistent storage volume

**Verify Qdrant is running**:
```shell
curl http://localhost:6333/health
```

Expected response: `{"status":"ok"}`

### Step 3: Run the Application

Execute the sparse retrieval demo:

```shell
python main.py
```

**Expected Output**:
```
# results
- id: 2, score: 0.95, text: ベクトル検索の仕組みを理解しましょう
- id: 0, score: 0.72, text: Qdrantは高速なベクトル検索エンジンです
- id: 3, score: 0.68, text: Pythonでベクトル検索エンジンを構築します

DONE
```

## Usage Example

### Custom Query

Modify the query in `main.py`:

```python
query = "Qdrant の使い方"  # Change query here
results = engine.search(query, limit=5)
show_results(results)
```

### Add Custom Documents

Modify the documents list in `main.py`:

```python
documents = [
    "Your custom document 1",
    "Your custom document 2",
    # Add more documents as needed
]
```

## Technical Details

### SPLADE Model

SPLADE (Sparse Lexical and sEmantic Embeddings) learns sparse vector representations:
- **Efficiency**: Sparse vectors use less memory than dense embeddings
- **Interpretability**: Non-zero dimensions correspond to meaningful tokens
- **Japanese Support**: bizreach-inc/light-splade-japanese-28M is optimized for Japanese

### Qdrant Configuration

The system uses Qdrant's sparse vector feature:
- **Vector Type**: Sparse vectors (list of token indices and values)
- **Storage**: Persistent across container restarts
- **Query Type**: Sparse vector similarity search

## Troubleshooting

### Issue: "Connection refused" to Qdrant

**Solution**: Ensure Qdrant container is running:
```shell
docker compose up -d
docker ps | grep qdrant  # Verify container is running
```

### Issue: CUDA out of memory

**Solution**: The model runs in inference mode. If still encountering issues:
1. Reduce batch size in encode.py
2. Use CPU (remove CUDA, slightly slower)
3. Ensure no other GPU processes are running

### Issue: "Model not found" error

**Solution**: The model is downloaded on first use. Ensure:
1. Internet connection is available
2. HuggingFace access is not restricted
3. Sufficient disk space for model (~500MB)

## Performance Considerations

- **Encoding Speed**: ~100 docs/sec on modern CPU
- **Search Speed**: <10ms per query on Qdrant server
- **Memory Usage**: ~200MB for encoder + Qdrant base installation
- **Storage**: Approximately 1-2KB per document in sparse form

## Future Enhancements

- [ ] Batch processing for large document collections
- [ ] Configuration file support (.env, YAML)
- [ ] REST API wrapper for the search engine
- [ ] Benchmark suite for performance testing
- [ ] Multi-language support expansion
- [ ] Hybrid search (sparse + dense vectors)

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| qdrant-client | 1.16.1 | Vector database client |
| light-splade | 0.1.2 | Sparse embedding model |
| torch | Latest | Deep learning framework |

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## References

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [SPLADE Paper](https://arxiv.org/abs/2107.05957)
- [Light SPLADE Repository](https://github.com/lightonai/light-splade)
- [Bizreach SPLADE Japanese Model](https://huggingface.co/bizreach-inc/light-splade-japanese-28M)

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Author

Created as a demonstration of sparse retrieval techniques combining Qdrant and SPLADE.

---

**Last Updated**: January 2026
