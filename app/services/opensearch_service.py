"""OpenSearch service for BM25 and HNSW vector search."""

import logging
from typing import List, Dict, Optional
from opensearchpy import AsyncOpenSearch, OpenSearchException
from opensearchpy.exceptions import NotFoundError

logger = logging.getLogger(__name__)


class OpenSearchService:
    """Client for OpenSearch dual search (BM25 + HNSW)."""
    
    def __init__(
        self,
        url: str,
        username: str,
        password: str,
        timeout: int = 30
    ):
        """
        Initialize OpenSearch client.
        
        Args:
            url: OpenSearch URL
            username: Authentication username
            password: Authentication password
            timeout: Request timeout in seconds
        """
        self.client = AsyncOpenSearch(
            hosts=[url],
            http_auth=(username, password),
            use_ssl=True,
            verify_certs=False,
            ssl_show_warn=False,
            timeout=timeout
        )
        logger.info(f"Initialized OpenSearch client: {url}")
    
    async def bm25_search(
        self,
        index: str,
        query_text: str,
        date_range: Optional[Dict[str, str]] = None,
        size: int = 100
    ) -> List[Dict]:
        """
        Perform BM25 text search.
        
        Args:
            index: Index name
            query_text: Query text (expanded with synonyms)
            date_range: Optional date range filter with 'from' and 'to' keys
            size: Number of results to return
            
        Returns:
            List of search results with scores
        """
        logger.debug(f"BM25 search in index '{index}' with query: {query_text[:100]}...")
        
        try:
            # Build query
            query = {
                "match": {
                    "text": {
                        "query": query_text
                    }
                }
            }
            
            # Add date range filter if provided
            body = {"query": query, "size": size}
            
            if date_range:
                body["query"] = {
                    "bool": {
                        "must": [query],
                        "filter": [
                            {
                                "range": {
                                    "user_upload_time": {
                                        "gte": date_range["from"],
                                        "lte": date_range["to"]
                                    }
                                }
                            }
                        ]
                    }
                }
            
            # Execute search
            response = await self.client.search(
                index=index,
                body=body
            )
            
            hits = response['hits']['hits']
            logger.debug(f"BM25 search returned {len(hits)} results from '{index}'")
            
            return hits
            
        except NotFoundError:
            logger.warning(f"Index not found: {index}")
            raise
        except OpenSearchException as e:
            logger.error(f"OpenSearch BM25 search error: {e}")
            raise
    
    async def hnsw_search(
        self,
        index: str,
        embedding: List[float],
        date_range: Optional[Dict[str, str]] = None,
        size: int = 100
    ) -> List[Dict]:
        """
        Perform HNSW vector search.
        
        Args:
            index: Index name
            embedding: Query embedding vector
            date_range: Optional date range filter with 'from' and 'to' keys
            size: Number of results to return
            
        Returns:
            List of search results with scores
        """
        logger.debug(f"HNSW search in index '{index}' with embedding dim {len(embedding)}")
        
        try:
            # Build KNN query
            query = {
                "knn": {
                    "embedding": {
                        "vector": embedding,
                        "k": size
                    }
                }
            }
            
            # Add date range filter if provided
            if date_range:
                query["knn"]["embedding"]["filter"] = {
                    "range": {
                        "user_upload_time": {
                            "gte": date_range["from"],
                            "lte": date_range["to"]
                        }
                    }
                }
            
            body = {"query": query, "size": size}
            
            # Execute search
            response = await self.client.search(
                index=index,
                body=body
            )
            
            hits = response['hits']['hits']
            logger.debug(f"HNSW search returned {len(hits)} results from '{index}'")
            
            return hits
            
        except NotFoundError:
            logger.warning(f"Index not found: {index}")
            raise
        except OpenSearchException as e:
            logger.error(f"OpenSearch HNSW search error: {e}")
            raise
    
    async def dual_search(
        self,
        index: str,
        query_text: str,
        embedding: List[float],
        date_range: Optional[Dict[str, str]] = None,
        size: int = 100
    ) -> tuple[List[Dict], List[Dict]]:
        """
        Perform both BM25 and HNSW searches concurrently.
        
        Args:
            index: Index name
            query_text: Query text for BM25
            embedding: Query embedding for HNSW
            date_range: Optional date range filter
            size: Number of results per search
            
        Returns:
            Tuple of (bm25_results, hnsw_results)
        """
        logger.debug(f"Dual search in index '{index}'")
        
        try:
            # Execute both searches concurrently
            import asyncio
            
            bm25_task = self.bm25_search(index, query_text, date_range, size)
            hnsw_task = self.hnsw_search(index, embedding, date_range, size)
            
            bm25_results, hnsw_results = await asyncio.gather(bm25_task, hnsw_task)
            
            return bm25_results, hnsw_results
            
        except Exception as e:
            logger.error(f"Dual search error in index '{index}': {e}")
            raise
    
    async def index_exists(self, index: str) -> bool:
        """
        Check if index exists.
        
        Args:
            index: Index name
            
        Returns:
            True if index exists, False otherwise
        """
        try:
            return await self.client.indices.exists(index=index)
        except Exception as e:
            logger.warning(f"Error checking index existence: {e}")
            return False
    
    async def health_check(self) -> bool:
        """
        Check if OpenSearch is available.
        
        Returns:
            True if service is healthy, False otherwise
        """
        try:
            health = await self.client.cluster.health()
            return health['status'] in ['green', 'yellow']
        except Exception as e:
            logger.warning(f"OpenSearch health check failed: {e}")
            return False
    
    async def close(self):
        """Close the OpenSearch client."""
        await self.client.close()
        logger.debug("OpenSearch client closed")
