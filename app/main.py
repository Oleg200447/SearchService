"""FastAPI main application with search orchestration."""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import nltk
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from opensearchpy.exceptions import NotFoundError
import httpx

from app.config import settings
from app.models import (
    SearchRequest, SearchResponse, SearchResult,
    ErrorResponse, ErrorDetail, HealthStatus
)
from app.services.synonym_service import SynonymService
from app.services.embedding_service import EmbeddingService
from app.services.opensearch_service import OpenSearchService
from app.services.reranking_service import RerankingService
from app.services.snippet_service import SnippetService
from app.utils.text_processing import TextProcessor
from app.utils.rrf import RRFMerger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global service instances
synonym_service: Optional[SynonymService] = None
embedding_service: Optional[EmbeddingService] = None
opensearch_service: Optional[OpenSearchService] = None
reranking_service: Optional[RerankingService] = None
snippet_service: Optional[SnippetService] = None
text_processor: Optional[TextProcessor] = None
rrf_merger: Optional[RRFMerger] = None


class ServiceException(Exception):
    """Base exception for service errors."""
    
    def __init__(self, status_code: int, message: str, detail: Optional[str] = None):
        self.status_code = status_code
        self.message = message
        self.detail = detail
        super().__init__(self.message)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - startup and shutdown."""
    global synonym_service, embedding_service, opensearch_service
    global reranking_service, snippet_service, text_processor, rrf_merger
    
    # Startup
    logger.info("Starting up Search Service...")
    
    try:
        # Download NLTK data
        logger.info("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt', quiet=True)
        
        # Initialize services
        logger.info("Initializing services...")
        
        synonym_service = SynonymService(settings.SYNONYMS_PATH)
        
        embedding_service = EmbeddingService(
            settings.EMBEDDING_SERVICE_URL,
            settings.EMBEDDING_TIMEOUT
        )
        
        opensearch_service = OpenSearchService(
            settings.OPENSEARCH_URL,
            settings.OPENSEARCH_USER,
            settings.OPENSEARCH_PASSWORD,
            settings.OPENSEARCH_TIMEOUT
        )
        
        reranking_service = RerankingService(
            settings.LOGREG_MODEL_PATH,
            settings.CATBOOST_MODEL_PATH,
            settings.TOP_K_FINAL,
            settings.CATBOOST_THRESHOLD
        )
        
        snippet_service = SnippetService(settings.SNIPPET_WINDOW_SIZE)
        
        text_processor = TextProcessor()
        
        rrf_merger = RRFMerger(settings.RRF_K)
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise ServiceException(503, "Service initialization failed", str(e))
    
    yield
    
    # Shutdown
    logger.info("Shutting down Search Service...")
    
    if embedding_service:
        await embedding_service.close()
    
    if opensearch_service:
        await opensearch_service.close()
    
    logger.info("Shutdown complete")


app = FastAPI(
    title="Search Service with Synonym Expansion and ML Reranking",
    description="Advanced search service using BM25, HNSW, and ML reranking",
    version="1.0.0",
    lifespan=lifespan
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add unique request ID to each request."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    
    return response


@app.exception_handler(ServiceException)
async def service_exception_handler(request: Request, exc: ServiceException):
    """Handle custom service exceptions."""
    logger.error(
        f"Service error: {exc.status_code} - {exc.message}",
        extra={"detail": exc.detail, "request_id": request.state.request_id}
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=ErrorDetail(
                code=exc.status_code,
                message=exc.message,
                detail=exc.detail,
                request_id=request.state.request_id
            )
        ).model_dump()
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors."""
    logger.warning(f"Validation error: {exc.errors()}", extra={"request_id": request.state.request_id})
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error=ErrorDetail(
                code=422,
                message="Request validation failed",
                detail=str(exc.errors()),
                request_id=request.state.request_id
            )
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error: {exc}", exc_info=True, extra={"request_id": request.state.request_id})
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error=ErrorDetail(
                code=500,
                message="Internal server error",
                detail=str(exc),
                request_id=request.state.request_id
            )
        ).model_dump()
    )


@app.get("/health", response_model=HealthStatus)
async def health_check():
    """Check service health status."""
    services_status = {
        "opensearch": "unknown",
        "embedding": "unknown",
        "models": "loaded" if reranking_service else "not_loaded"
    }
    
    # Check OpenSearch
    if opensearch_service:
        services_status["opensearch"] = "up" if await opensearch_service.health_check() else "down"
    
    # Check Embedding service
    if embedding_service:
        services_status["embedding"] = "up" if await embedding_service.health_check() else "down"
    
    # Determine overall status
    if all(v in ["up", "loaded"] for v in services_status.values()):
        overall_status = "healthy"
    elif any(v == "down" for v in services_status.values()):
        overall_status = "unhealthy"
    else:
        overall_status = "degraded"
    
    return HealthStatus(status=overall_status, services=services_status)


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest, req: Request):
    """
    Perform advanced search with synonym expansion and ML reranking.
    
    Pipeline:
    1. Tokenize and expand query with synonyms
    2. Get embedding for original query
    3. Dual search (BM25 + HNSW) in each index
    4. Merge results per index using RRF
    5. Combine all indices and deduplicate by text_hash
    6. Two-stage reranking (LogReg + CatBoost)
    7. Extract snippets from top documents
    """
    start_time = time.time()
    request_id = req.state.request_id
    
    logger.info(
        f"Search request received: query='{request.query}', "
        f"indices={request.indices}, user_id={request.user_id}",
        extra={"request_id": request_id}
    )
    
    try:
        # Validate date range if provided
        if request.filters and request.filters.date_range:
            try:
                request.filters.date_range.validate_range()
            except ValueError as e:
                raise ServiceException(400, "Invalid date range", str(e))
        
        # Step 1: Tokenize query
        tokens = text_processor.tokenize_query(request.query)
        logger.debug(f"Tokenized query: {tokens}", extra={"request_id": request_id})
        
        if not tokens:
            raise ServiceException(400, "Query contains no valid tokens", "Query must contain alphanumeric characters")
        
        # Step 2: Expand query with synonyms
        expanded_query = synonym_service.expand_query(tokens)
        expanded_tokens = expanded_query.split()
        logger.info(f"Query expanded: {len(tokens)} -> {len(expanded_tokens)} terms", extra={"request_id": request_id})
        
        # Step 3: Get embedding for ORIGINAL query (not expanded)
        try:
            embedding = await embedding_service.get_embedding(request.query)
        except httpx.TimeoutException:
            raise ServiceException(504, "Embedding service timeout", "Request to embedding service timed out")
        except httpx.HTTPStatusError as e:
            raise ServiceException(503, "Embedding service unavailable", f"HTTP {e.response.status_code}")
        except Exception as e:
            raise ServiceException(503, "Embedding service error", str(e))
        
        # Step 4: Search in all indices
        all_merged_docs = []
        date_range_dict = None
        
        if request.filters and request.filters.date_range:
            date_range_dict = {
                "from": request.filters.date_range.from_date,
                "to": request.filters.date_range.to_date
            }
        
        for index in request.indices:
            logger.debug(f"Searching in index: {index}", extra={"request_id": request_id})
            
            try:
                # Check if index exists
                if not await opensearch_service.index_exists(index):
                    logger.warning(f"Index not found: {index}", extra={"request_id": request_id})
                    raise ServiceException(404, f"Index not found: {index}", "Specified index does not exist")
                
                # Dual search
                bm25_results, hnsw_results = await opensearch_service.dual_search(
                    index,
                    expanded_query,
                    embedding,
                    date_range_dict,
                    settings.DOCS_PER_INDEX
                )
                
                logger.debug(
                    f"Index '{index}': BM25={len(bm25_results)}, HNSW={len(hnsw_results)}",
                    extra={"request_id": request_id}
                )
                
                # Merge with RRF
                merged = rrf_merger.merge(bm25_results, hnsw_results, settings.DOCS_PER_INDEX)
                all_merged_docs.extend(merged)
                
            except NotFoundError:
                raise ServiceException(404, f"Index not found: {index}", "Specified index does not exist")
            except Exception as e:
                raise ServiceException(503, f"OpenSearch error for index {index}", str(e))
        
        logger.info(f"Total documents after RRF merge: {len(all_merged_docs)}", extra={"request_id": request_id})
        
        if not all_merged_docs:
            logger.info("No documents found", extra={"request_id": request_id})
            processing_time = (time.time() - start_time) * 1000
            return SearchResponse(
                results=[],
                query_expanded=expanded_query,
                processing_time_ms=processing_time,
                request_id=request_id
            )
        
        # Step 5: Deduplicate by text_hash
        seen_hashes = set()
        unique_docs = []
        
        for doc in all_merged_docs:
            text_hash = doc.get('_source', {}).get('text_hash')
            if text_hash and text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                unique_docs.append(doc)
            elif not text_hash:
                unique_docs.append(doc)  # Include docs without hash
        
        logger.info(f"After deduplication: {len(unique_docs)} documents", extra={"request_id": request_id})
        
        # Step 6: Two-stage reranking
        try:
            reranked_docs = reranking_service.rerank(unique_docs)
        except Exception as e:
            raise ServiceException(500, "Reranking failed", str(e))
        
        logger.info(f"After reranking: {len(reranked_docs)} documents", extra={"request_id": request_id})
        
        if not reranked_docs:
            processing_time = (time.time() - start_time) * 1000
            return SearchResponse(
                results=[],
                query_expanded=expanded_query,
                processing_time_ms=processing_time,
                request_id=request_id
            )
        
        # Step 7: Extract snippets
        try:
            docs_with_snippets = snippet_service.extract_snippets_batch(
                reranked_docs,
                expanded_tokens
            )
        except Exception as e:
            raise ServiceException(500, "Snippet extraction failed", str(e))
        
        # Step 8: Prepare response
        results = []
        for doc in docs_with_snippets:
            results.append(SearchResult(
                doc_id=doc.get('_source', {}).get('doc_id', doc.get('_id', 'unknown')),
                snippet=doc.get('snippet', ''),
                score=doc.get('catboost_score', 0.0)
            ))
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(
            f"Search completed: {len(results)} results in {processing_time:.2f}ms",
            extra={"request_id": request_id}
        )
        
        return SearchResponse(
            results=results,
            query_expanded=expanded_query,
            processing_time_ms=processing_time,
            request_id=request_id
        )
        
    except ServiceException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in search: {e}", exc_info=True, extra={"request_id": request_id})
        raise ServiceException(500, "Search processing failed", str(e))


@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "Search Service",
        "version": "1.0.0",
        "status": "running"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.SERVICE_HOST,
        port=settings.SERVICE_PORT,
        log_level="info"
    )
