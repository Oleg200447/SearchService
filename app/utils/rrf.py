"""Reciprocal Rank Fusion (RRF) implementation for merging search results."""

import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


class RRFMerger:
    """Implements Reciprocal Rank Fusion for combining multiple search result rankings."""
    
    def __init__(self, k: int = 60):
        """
        Initialize RRF merger.
        
        Args:
            k: RRF constant (typically 60)
        """
        self.k = k
        logger.debug(f"Initialized RRF merger with k={k}")
    
    def merge(
        self,
        bm25_results: List[Dict],
        hnsw_results: List[Dict],
        top_n: int
    ) -> List[Dict]:
        """
        Merge BM25 and HNSW results using RRF.
        
        Args:
            bm25_results: List of BM25 search results with scores
            hnsw_results: List of HNSW search results with scores
            top_n: Number of top results to return
            
        Returns:
            Merged and sorted list of documents with RRF scores
        """
        logger.debug(f"Merging {len(bm25_results)} BM25 and {len(hnsw_results)} HNSW results")
        
        # Create doc_id to document mapping and track original scores
        doc_map = {}
        
        # Process BM25 results
        for rank, doc in enumerate(bm25_results, start=1):
            doc_id = doc['_id']
            if doc_id not in doc_map:
                doc_map[doc_id] = {
                    'doc': doc,
                    'bm25_score': doc['_score'],
                    'hnsw_score': 0.0,
                    'bm25_rank': rank,
                    'hnsw_rank': None,
                    'rrf_score': 0.0
                }
            else:
                doc_map[doc_id]['bm25_score'] = doc['_score']
                doc_map[doc_id]['bm25_rank'] = rank
        
        # Process HNSW results
        for rank, doc in enumerate(hnsw_results, start=1):
            doc_id = doc['_id']
            if doc_id not in doc_map:
                doc_map[doc_id] = {
                    'doc': doc,
                    'bm25_score': 0.0,
                    'hnsw_score': doc['_score'],
                    'bm25_rank': None,
                    'hnsw_rank': rank,
                    'rrf_score': 0.0
                }
            else:
                doc_map[doc_id]['hnsw_score'] = doc['_score']
                doc_map[doc_id]['hnsw_rank'] = rank
        
        # Calculate RRF scores
        for doc_id, info in doc_map.items():
            rrf_score = 0.0
            
            if info['bm25_rank'] is not None:
                rrf_score += 1.0 / (self.k + info['bm25_rank'])
            
            if info['hnsw_rank'] is not None:
                rrf_score += 1.0 / (self.k + info['hnsw_rank'])
            
            info['rrf_score'] = rrf_score
            
            logger.debug(
                f"Doc {doc_id}: BM25 rank={info['bm25_rank']}, "
                f"HNSW rank={info['hnsw_rank']}, RRF score={rrf_score:.6f}"
            )
        
        # Sort by RRF score and select top N
        sorted_docs = sorted(
            doc_map.values(),
            key=lambda x: x['rrf_score'],
            reverse=True
        )[:top_n]
        
        # Prepare output format
        results = []
        for item in sorted_docs:
            doc = item['doc'].copy()
            doc['bm25_score'] = item['bm25_score']
            doc['hnsw_score'] = item['hnsw_score']
            doc['rrf_score'] = item['rrf_score']
            results.append(doc)
        
        logger.debug(f"RRF merge completed: {len(results)} documents selected")
        return results
