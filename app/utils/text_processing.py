"""Text processing utilities for tokenization and cleaning."""

import re
import logging
from typing import List
from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)


class TextProcessor:
    """Handles text tokenization and cleaning operations."""
    
    # Characters to remove from PDFs and other documents
    PDF_CHARS_TO_REMOVE = [
        '\n', '\r', '\t', '\f',
        '\u00A0',  # Non-breaking space
        '\u2002',  # En space
        '\u2003',  # Em space
        '\u2009',  # Thin space
        '\u200B',  # Zero-width space
        '\uFEFF',  # Zero-width no-break space
    ]
    
    @staticmethod
    def tokenize_query(query: str) -> List[str]:
        """
        Tokenize query text and extract alphanumeric tokens.
        
        Args:
            query: Input query string
            
        Returns:
            List of lowercase alphanumeric tokens
        """
        logger.debug(f"Tokenizing query: {query}")
        
        # Tokenize using NLTK
        tokens = word_tokenize(query)
        
        # Keep only alphanumeric tokens and convert to lowercase
        clean_tokens = [t.lower() for t in tokens if t.isalnum()]
        
        logger.debug(f"Extracted {len(clean_tokens)} clean tokens: {clean_tokens}")
        return clean_tokens
    
    @staticmethod
    def clean_document_text(text: str) -> str:
        """
        Clean document text by removing PDF artifacts and special characters.
        
        Args:
            text: Raw document text
            
        Returns:
            Cleaned text string
        """
        logger.debug(f"Cleaning document text of length {len(text)}")
        
        cleaned = text.replace('-\n','')
        for char in TextProcessor.PDF_CHARS_TO_REMOVE:
            cleaned = cleaned.replace(char, ' ')
        
        # Collapse multiple spaces into single space
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        logger.debug(f"Cleaned text length: {len(cleaned)}")
        return cleaned
    
    @staticmethod
    def extract_overlapping_windows(text: str, window_size: int = 30) -> List[str]:
        """
        Extract overlapping text windows starting at sentence boundaries.
        Each window contains up to window_size tokens (words with punctuation).
        """

        if not text.strip():
            return []

        tokens = text.split()

        # Sentence start indices in token space
        sentence_starts = [0]

        for i in range(1, len(tokens)):
            # Start of sentence if previous token ends with sentence punctuation
            if re.search(r'[.!?]$', tokens[i - 1]):
                sentence_starts.append(i)

        windows = [
            ' '.join(tokens[start:start + window_size])
            for start in sentence_starts
        ]

        return windows
