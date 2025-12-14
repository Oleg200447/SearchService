# Search Service with Synonym Expansion and ML Reranking

Advanced search service combining BM25 text search, HNSW vector search, synonym expansion, and ML-based reranking.

## Features

- **Synonym Expansion**: Query enhancement using n-gram matching (1-4 words)
- **Dual Search**: BM25 text search + HNSW vector search
- **RRF Merging**: Reciprocal Rank Fusion for combining search results
- **ML Reranking**: Two-stage filtering and ranking using LogisticRegression and CatBoost
- **Snippet Extraction**: BM25-based relevant snippet extraction
- **Comprehensive Error Handling**: HTTP status codes with detailed error messages
- **Health Monitoring**: Service health check endpoint
- **Async Architecture**: Fully asynchronous for high performance

## Architecture

### Search Pipeline

1. **Tokenization**: Query is tokenized using NLTK word_tokenize
2. **Synonym Expansion**: Tokens are expanded using n-gram synonym matching
3. **Embedding Generation**: Original query is embedded using vLLM service (BAAI/bge-m3)
4. **Dual Search**: For each index:
   - BM25 search with expanded query
   - HNSW vector search with query embedding
5. **RRF Merging**: Results from BM25 and HNSW are merged per index
6. **Deduplication**: Documents with same text_hash are deduplicated
7. **Two-Stage Reranking**:
   - Stage 1: LogisticRegression filters out irrelevant documents
   - Stage 2: CatBoost ranks remaining documents
8. **Threshold Filtering**: Documents below CatBoost threshold are removed
9. **Snippet Extraction**: Best matching snippet extracted for each document
10. **Response**: Top K documents returned with snippets and scores

## Requirements

### External Services

- **OpenSearch**: Document storage and search (BM25 + KNN)
- **vLLM Embedding Service**: Text embedding generation (BAAI/bge-m3 model)

### Docker Network

Services must be on the same Docker network (`test_network`) for communication.

## Installation

### Using Docker (Recommended)

1. **Ensure external services are running**:
   ```bash
   # OpenSearch should be running on test_network:9200
   # Embedding service should be running on test_network:8000
   ```

2. **Build the Docker image**:
   ```bash
   docker build -t search-service .
   ```

3. **Run the service**:
   ```bash
   docker run -d \
     --name search_service \
     --network test_network \
     -p 8008:8008 \
     search-service
   ```

### Local Development

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Update .env file** with local service URLs:
   ```env
   EMBEDDING_SERVICE_URL=http://localhost:8004
   OPENSEARCH_URL=https://localhost:9200
   ```

3. **Run the service**:
   ```bash
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8008
   ```

## Configuration

Configuration is managed via environment variables in `.env` file:

```env
# Service Configuration
SERVICE_PORT=8008
SERVICE_HOST=0.0.0.0

# External Services (Docker network names)
EMBEDDING_SERVICE_URL=http://embedding_server:8000
OPENSEARCH_URL=https://opensearch:9200
OPENSEARCH_USER=admin
OPENSEARCH_PASSWORD=our_password

# Search Parameters
RRF_K=60                    # RRF constant
DOCS_PER_INDEX=100          # Documents to select per index after RRF
TOP_K_FINAL=15              # Final number of documents to return
CATBOOST_THRESHOLD=0.5      # Minimum CatBoost score threshold
SNIPPET_WINDOW_SIZE=30      # Words per snippet window

# File Paths
SYNONYMS_PATH=data/synonyms.txt
LOGREG_MODEL_PATH=models/logreg.pkl
CATBOOST_MODEL_PATH=models/catboost.cbm

# Timeouts (seconds)
OPENSEARCH_TIMEOUT=30
EMBEDDING_TIMEOUT=60
```

## API Endpoints

### POST /search

Perform advanced search with synonym expansion and ML reranking.

**Request Body**:
```json
{
  "query": "machine learning applications",
  "user_id": "12345",
  "indices": ["user123_topic1", "system_topic1", "system_topic2"],
  "filters": {
    "date_range": {
      "from": "2024-01-01",
      "to": "2024-12-31"
    }
  }
}
```

**Response**:
```json
{
  "results": [
    {
      "doc_id": "doc123",
      "snippet": "Machine learning is a subset of artificial intelligence...",
      "score": 0.85
    }
  ],
  "query_expanded": "machine learning ml applications",
  "processing_time_ms": 245.6,
  "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
}
```

### GET /health

Check service health status.

**Response**:
```json
{
  "status": "healthy",
  "services": {
    "opensearch": "up",
    "embedding": "up",
    "models": "loaded"
  }
}
```

### GET /

Service information.

## Error Handling

The service returns appropriate HTTP status codes with detailed error messages:

- **400 Bad Request**: Invalid query format, invalid date range
- **404 Not Found**: Index not found, required files missing
- **422 Unprocessable Entity**: Validation errors (missing fields, invalid formats)
- **500 Internal Server Error**: Unexpected processing errors
- **503 Service Unavailable**: External service unavailable (OpenSearch, embedding service)
- **504 Gateway Timeout**: External service timeout

**Error Response Format**:
```json
{
  "error": {
    "code": 404,
    "message": "Index not found: user123_topic1",
    "detail": "Specified index does not exist",
    "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
  }
}
```

## File Structure

```
SearchServiceBaseLine/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI application
│   ├── config.py                  # Configuration management
│   ├── models.py                  # Pydantic models
│   ├── services/
│   │   ├── __init__.py
│   │   ├── synonym_service.py     # Synonym expansion
│   │   ├── embedding_service.py   # Embedding client
│   │   ├── opensearch_service.py  # OpenSearch client
│   │   ├── reranking_service.py   # ML reranking
│   │   └── snippet_service.py     # Snippet extraction
│   └── utils/
│       ├── __init__.py
│       ├── text_processing.py     # Text utilities
│       └── rrf.py                 # RRF algorithm
├── data/
│   └── synonyms.txt               # Synonym dictionary
├── models/
│   ├── logreg.pkl                 # LogReg model
│   └── catboost.cbm               # CatBoost model
├── .env                           # Environment configuration
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Docker configuration
└── README.md                      # Documentation
```

## Synonyms File Format

Each line contains comma-separated synonyms:

```
machine learning, ml
deep learning, dl
artificial intelligence, ai
neural network, nn
```

The service creates a bidirectional mapping where each term maps to all other terms in the same line.

## Model Requirements

### LogReg Model (logreg.pkl)

Pickle file containing:
```python
{
    'model': sklearn.linear_model.LogisticRegression,
    'scaler': sklearn.preprocessing.StandardScaler,
    'threshold': float  # Classification threshold
}
```

Features: `[bm25_score, hnsw_score]`

### CatBoost Model (catboost.cbm)

Standard CatBoost model file trained on features: `[bm25_score, hnsw_score]`

## OpenSearch Index Schema

Expected index structure:

```json
{
  "mappings": {
    "properties": {
      "text": {"type": "text"},
      "embedding": {
        "type": "knn_vector",
        "dimension": 1024,
        "method": {
          "name": "hnsw",
          "space_type": "cosinesimil"
        }
      },
      "doc_id": {"type": "keyword"},
      "text_hash": {"type": "keyword"},
      "user_upload_time": {"type": "date"}
    }
  }
}
```

## Logging

The service uses Python's standard logging with different levels:

- **INFO**: Request received, processing stages completed, results count
- **DEBUG**: Token expansion, synonym matches, search results, scores, filtering decisions
- **ERROR**: Service failures, connection errors
- **WARNING**: Missing data, unexpected conditions

Each log entry includes a `request_id` for request tracing.

## Performance Considerations

- All I/O operations are asynchronous
- Concurrent dual searches (BM25 + HNSW) per index
- Efficient RRF merging with O(n log n) complexity
- Batch snippet extraction
- Model inference optimized with NumPy vectorization

## Development

### Running Tests

```bash
# Install dev dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest tests/
```

### Adding New Synonyms

1. Edit `data/synonyms.txt`
2. Add new synonym lines
3. Restart the service

### Updating Models

1. Replace model files in `models/` directory
2. Ensure feature dimensions match (2 features: BM25 and HNSW scores)
3. Restart the service

## Troubleshooting

### Service won't start

- Check that model files exist in `models/` directory
- Verify synonyms.txt exists in `data/` directory
- Check logs for initialization errors

### OpenSearch connection failed

- Verify OpenSearch is running and accessible
- Check credentials in .env file
- Ensure services are on same Docker network

### Embedding service timeout

- Check vLLM service is running
- Increase `EMBEDDING_TIMEOUT` in .env
- Verify network connectivity

### No results returned

- Check that indices exist in OpenSearch
- Verify documents match date range filters
- Review LogReg/CatBoost thresholds (may be too strict)

## License

This project is proprietary software.

## Support

For issues and questions, contact the development team.
