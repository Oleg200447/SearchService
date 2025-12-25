"""Synonym expansion service for query enhancement."""

import logging
from typing import List, Dict, Set
from pathlib import Path

logger = logging.getLogger(__name__)


class SynonymService:
    """Manages synonym dictionary and query expansion."""
    
    def __init__(self, synonyms_path: str):
        """
        Initialize synonym service.
        
        Args:
            synonyms_path: Path to synonyms text file
        """
        self.synonyms_path = synonyms_path
        self.synonym_dict: Dict[str, Set[str]] = {}
        self._load_synonyms()
    
    def _load_synonyms(self):
        """Load synonyms from file and build bidirectional dictionary."""
        logger.info(f"Loading synonyms from {self.synonyms_path}")
        
        try:
            path = Path(self.synonyms_path)
            if not path.exists():
                raise FileNotFoundError(f"Synonyms file not found: {self.synonyms_path}")
            
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines, start=1):
                line = line.strip()
                if not line:
                    continue
                
                # Split by comma to get synonyms
                synonyms = [s.strip().lower() for s in line.split(',') if s.strip()]
                
                if len(synonyms) < 2:
                    logger.warning(f"Line {line_num}: Less than 2 synonyms found, skipping")
                    continue
                
                # Create bidirectional mapping: each word maps to all others in the line
                for i, word in enumerate(synonyms):
                    # Get all other words except the current one
                    other_words = set(synonyms[:i] + synonyms[i+1:])
                    
                    if word in self.synonym_dict:
                        # Add to existing set
                        self.synonym_dict[word].update(other_words)
                    else:
                        # Create new set
                        self.synonym_dict[word] = other_words
                
                logger.debug(f"Loaded synonym group: {', '.join(synonyms)}")
            
            logger.info(f"Loaded {len(self.synonym_dict)} unique words with synonyms")
            
        except Exception as e:
            logger.error(f"Failed to load synonyms: {e}")
            raise
    
    def expand_query(self, tokens: List[str]) -> str:
        """
        Expand query tokens with synonyms using n-gram matching (1-4 words).
        
        Args:
            tokens: List of query tokens
            
        Returns:
            Expanded query string with original tokens and synonyms
        """
        logger.debug(f"Expanding query with tokens: {tokens}")
        
        all_terms = list(tokens)  # Start with original tokens
        found_synonyms = list()
        
        # Try n-grams from 1 to 4 words
        for n in range(1, 5):
            for i in range(len(tokens) - n + 1):
                ngram = ' '.join(tokens[i:i+n])
                
                if ngram in self.synonym_dict:
                    synonyms = self.synonym_dict[ngram]
                    found_synonyms.extend(synonyms)
                    logger.debug(f"Found synonyms for '{ngram}': {synonyms}")
        
        # Add found synonyms to the set
        all_terms.extend(found_synonyms)
        
        # Create expanded query string
        expanded_query = ' '.join(sorted(all_terms))
        
        logger.debug(f"Query expanded: {len(tokens)} original terms -> {len(all_terms)} total terms")
        logger.debug(f"Added {len(found_synonyms)} synonym terms")
        
        return expanded_query
    
    def get_synonym_count(self) -> int:
        """Get the number of unique words with synonyms."""
        return len(self.synonym_dict)
