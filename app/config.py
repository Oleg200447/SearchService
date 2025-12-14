"""Configuration management using pydantic-settings."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Service Configuration
    SERVICE_PORT: int = 8008
    SERVICE_HOST: str = "0.0.0.0"
    
    # External Services
    EMBEDDING_SERVICE_URL: str = "http://embedding_server:8000"
    OPENSEARCH_URL: str = "https://opensearch:9200"
    OPENSEARCH_USER: str = "admin"
    OPENSEARCH_PASSWORD: str = "our_password"
    
    # Search Parameters
    RRF_K: int = 60
    DOCS_PER_INDEX: int = 100
    TOP_K_FINAL: int = 15
    CATBOOST_THRESHOLD: float = 0.5
    SNIPPET_WINDOW_SIZE: int = 30
    
    # File Paths
    SYNONYMS_PATH: str = "data/synonyms.txt"
    LOGREG_MODEL_PATH: str = "models/logreg.pkl"
    CATBOOST_MODEL_PATH: str = "models/catboost.cbm"
    
    # Timeouts
    OPENSEARCH_TIMEOUT: int = 30
    EMBEDDING_TIMEOUT: int = 60
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
