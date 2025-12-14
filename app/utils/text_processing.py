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
        
        cleaned = text
        for char in TextProcessor.PDF_CHARS_TO_REMOVE:
            cleaned = cleaned.replace(char, ' ')
        
        # Collapse multiple spaces into single space
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        logger.debug(f"Cleaned text length: {len(cleaned)}")
        return cleaned
    
    @staticmethod
    def extract_overlapping_windows(text: str, window_size: int = 30) -> List[str]:
        """
        Extract overlapping text windows based on sentence boundaries and word count.
        Each window starts at a sentence beginning and includes the next window_size words.
        
        Args:
            text: Input text to split
            window_size: Number of words per window
            
        Returns:
            List of overlapping text windows
        """
        logger.debug(f"Extracting windows with size {window_size} from text of length {len(text)}")
        
        # Split by sentence boundaries (., !, ?)
        sentences = re.split(r'[.!?]+', text)
        
        windows = []
        current_words = []
        
        for sentence in sentences:
            words = sentence.strip().split()
            if not words:
                continue
            
            # Start a new window at each sentence
            window_start_words = current_words + words
            
            # Extract window_size words from this position
            if len(window_start_words) >= window_size:
                window = ' '.join(window_start_words[:window_size])
                windows.append(window)
            elif window_start_words:
                # If we have fewer than window_size words, take what we have
                window = ' '.join(window_start_words)
                windows.append(window)
            
            # Keep accumulating words for next window
            current_words.extend(words)
        
        logger.debug(f"Extracted {len(windows)} overlapping windows")
        return windows
