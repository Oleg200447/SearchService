"""Reranking service using LogisticRegression and CatBoost models."""

import logging
import pickle
from typing import List, Dict
from pathlib import Path
import numpy as np
from catboost import CatBoostClassifier

logger = logging.getLogger(__name__)


class LogRegReranker:
    """Logistic Regression reranker with custom predict methods."""
    
    def __init__(self, model, scaler, threshold):
        """
        Initialize LogReg reranker.
        
        Args:
            model: Trained logistic regression model
            scaler: Feature scaler
            threshold: Classification threshold
        """
        self.model = model
        self.scaler = scaler
        self.threshold = threshold

    def predict(self, X):
        """
        Predict binary class labels.
        
        Args:
            X: Feature matrix
            
        Returns:
            Binary predictions (0 or 1)
        """
        X_scaled = self.scaler.transform(X)
        proba = self.model.predict_proba(X_scaled)[:, 1]
        return (proba >= self.threshold).astype(int)
    
    def predict_proba(self, X):
        """
        Predict probability of positive class.
        
        Args:
            X: Feature matrix
            
        Returns:
            Probability scores
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]


class RerankingService:
    """Two-stage reranking using LogReg filtering and CatBoost scoring."""
    
    def __init__(
        self,
        logreg_path: str,
        catboost_path: str,
        top_k: int = 15,
        catboost_threshold: float = 0.5
    ):
        """
        Initialize reranking service.
        
        Args:
            logreg_path: Path to LogReg model pickle file
            catboost_path: Path to CatBoost model file
            top_k: Number of final results to return
            catboost_threshold: Score threshold for CatBoost filtering
        """
        self.top_k = top_k
        self.catboost_threshold = catboost_threshold
        
        self._load_models(logreg_path, catboost_path)
    
    def _load_models(self, logreg_path: str, catboost_path: str):
        """Load trained models from disk."""
        logger.info(f"Loading models: LogReg from {logreg_path}, CatBoost from {catboost_path}")
        
        try:
            # Load LogReg model
            logreg_file = Path(logreg_path)
            if not logreg_file.exists():
                raise FileNotFoundError(f"LogReg model not found: {logreg_path}")
            
            with open(logreg_file, 'rb') as f:
                logreg_data = pickle.load(f)
            
            # Extract model, scaler, and threshold
            self.logreg = LogRegReranker(
                model=logreg_data['model'],
                scaler=logreg_data['scaler'],
                threshold=logreg_data.get('threshold', 0.5)
            )
            
            logger.info("LogReg model loaded successfully")
            
            # Load CatBoost model
            catboost_file = Path(catboost_path)
            if not catboost_file.exists():
                raise FileNotFoundError(f"CatBoost model not found: {catboost_path}")
            
            self.catboost = CatBoostClassifier()
            self.catboost.load_model(str(catboost_file))
            
            logger.info("CatBoost model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def rerank(self, documents: List[Dict]) -> List[Dict]:
        """
        Two-stage reranking: LogReg filtering + CatBoost scoring.
        
        Args:
            documents: List of documents with bm25_score and hnsw_score
            
        Returns:
            Filtered and reranked documents
        """
        if not documents:
            logger.debug("No documents to rerank")
            return []
        
        logger.info(f"Reranking {len(documents)} documents")
        
        # Extract features: [bm25_score, hnsw_score]
        features = np.array([
            [doc['bm25_score'], doc['hnsw_score']]
            for doc in documents
        ])
        
        logger.debug(f"Feature matrix shape: {features.shape}")
        
        # Stage 1: LogReg filtering
        predictions = self.logreg.predict(features)
        filtered_docs = [
            doc for doc, pred in zip(documents, predictions)
            if pred == 1
        ]
        
        logger.info(f"LogReg filtering: {len(documents)} -> {len(filtered_docs)} documents")
        
        if not filtered_docs:
            logger.warning("No documents passed LogReg filtering")
            return []
        
        # Stage 2: CatBoost scoring
        filtered_features = np.array([
            [doc['bm25_score'], doc['hnsw_score']]
            for doc in filtered_docs
        ])
        
        catboost_scores = self.catboost.predict(filtered_features, prediction_type='RawFormulaVal')
        
        # Attach CatBoost scores to documents
        for doc, score in zip(filtered_docs, catboost_scores):
            doc['catboost_score'] = float(score)
        
        # Sort by CatBoost score and select top K
        sorted_docs = sorted(
            filtered_docs,
            key=lambda x: x['catboost_score'],
            reverse=True
        )[:self.top_k]
        
        logger.debug(f"Selected top {len(sorted_docs)} documents by CatBoost score")
        
        # Apply CatBoost threshold filter
        final_docs = [
            doc for doc in sorted_docs
            if doc['catboost_score'] >= self.catboost_threshold
        ]
        
        logger.info(f"After CatBoost threshold filter: {len(final_docs)} documents")
        
        return final_docs
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models."""
        return {
            "logreg_threshold": self.logreg.threshold,
            "catboost_threshold": self.catboost_threshold,
            "top_k": self.top_k
        }
