"""Embedding service client for vLLM OpenAI-compatible API."""

import logging
from typing import List
import httpx

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Client for vLLM embedding service."""
    
    def __init__(self, service_url: str, timeout: int = 60):
        """
        Initialize embedding service client.
        
        Args:
            service_url: Base URL of the embedding service
            timeout: Request timeout in seconds
        """
        self.service_url = service_url.rstrip('/')
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout, verify=False)
        logger.info(f"Initialized embedding service client: {service_url}")
    
    async def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding vector for text using vLLM OpenAI-compatible API.
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector as list of floats
            
        Raises:
            httpx.HTTPStatusError: If request fails
            httpx.TimeoutException: If request times out
        """
        logger.debug(f"Requesting embedding for text of length {len(text)}")
        
        try:
            # OpenAI-compatible embeddings endpoint
            url = f"{self.service_url}/v1/embeddings"
            
            payload = {
                "model": "BAAI/bge-m3",
                "input": text
            }
            
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            
            result = response.json()
            
            # Extract embedding from OpenAI-style response
            embedding = result['data'][0]['embedding']
            
            logger.debug(f"Received embedding of dimension {len(embedding)}")
            return embedding
            
        except httpx.TimeoutException as e:
            logger.error(f"Embedding service timeout: {e}")
            raise
        except httpx.HTTPStatusError as e:
            logger.error(f"Embedding service HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Embedding service error: {e}")
            raise
    
    async def health_check(self) -> bool:
        """
        Check if embedding service is available.
        
        Returns:
            True if service is healthy, False otherwise
        """
        try:
            url = f"{self.service_url}/health"
            response = await self.client.get(url, timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Embedding service health check failed: {e}")
            return False
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
        logger.debug("Embedding service client closed")
