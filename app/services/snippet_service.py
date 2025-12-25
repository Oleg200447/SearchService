"""Snippet extraction service using BM25 scoring on text windows."""

import logging
import re
from typing import List, Dict
from rank_bm25 import BM25Okapi
from app.utils.text_processing import TextProcessor

logger = logging.getLogger(__name__)


class SnippetService:
    """Extracts relevant snippets from documents using BM25."""
    
    def __init__(self, window_size: int = 30):
        """
        Initialize snippet service.
        
        Args:
            window_size: Number of words per window
        """
        self.window_size = window_size
        self.text_processor = TextProcessor()
        logger.info(f"Initialized snippet service with window size {window_size}")
    
    def extract_snippet(self, document_text: str, query_tokens: List[str]) -> str:
        """
        Extract best matching snippet from document text.
        
        Args:
            document_text: Full document text
            query_tokens: Tokenized and expanded query
            
        Returns:
            Best matching snippet
        """
        logger.debug(f"Extracting snippet from document of length {len(document_text)}")
        # Clean document text
        cleaned_text = self.text_processor.clean_document_text(document_text[:10000]) #TODO
        if not cleaned_text:
            logger.warning("Document text is empty after cleaning")
            return ""
        # Extract overlapping windows
        windows = self.text_processor.extract_overlapping_windows(
            cleaned_text,
            self.window_size
        )
        if not windows:
            logger.warning("No windows extracted from document")
            return cleaned_text[:500]  # Return first 500 chars as fallback
        
        if len(windows) == 1:
            return windows[0]
        
        # Tokenize windows for BM25 (clean punctuation for accurate scoring)
        tokenized_windows = []
        for window in windows:
            # Remove punctuation and split into tokens
            clean_window = re.sub(r'[^\w\s]', '', window)
            tokens = clean_window.lower().split()
            # Filter out empty tokens and commas
            tokens = [token.strip(',') for token in tokens if token.strip(',')]
            tokenized_windows.append(tokens)
        # Initialize BM25
        bm25 = BM25Okapi(tokenized_windows)
        # Score windows against query
        scores = bm25.get_scores(query_tokens)
        
        # Find best window
        best_idx = scores.argmax()
        best_score = scores[best_idx]
        logger.debug(f"Best snippet score: {best_score:.4f} (window {best_idx + 1}/{len(windows)})")
        
        return windows[best_idx]
    
    def extract_snippets_batch(
        self,
        documents: List[Dict],
        query_tokens: List[str]
    ) -> List[Dict]:
        """
        Extract snippets for multiple documents.
        
        Args:
            documents: List of documents with '_source' containing 'text' field
            query_tokens: Tokenized and expanded query
            
        Returns:
            Documents with added 'snippet' field
        """
        logger.info(f"Extracting snippets for {len(documents)} documents")
        
        for doc in documents:
            try:
                # Get document text
                doc_text = doc.get('_source', {}).get('text', '')
                
                if not doc_text:
                    logger.warning(f"Document {doc.get('_id', 'unknown')} has no text")
                    doc['snippet'] = ""
                    continue
                
                # Extract snippet
                snippet = self.extract_snippet(doc_text, query_tokens)
                doc['snippet'] = snippet
                
            except Exception as e:
                logger.error(f"Error extracting snippet for document {doc.get('_id', 'unknown')}: {e}")
                doc['snippet'] = ""
        
        logger.info("Snippet extraction completed")
        return documents
