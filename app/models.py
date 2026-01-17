"""Pydantic models for request/response validation."""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


class DateRange(BaseModel):
    """Date range filter for search."""
    from_date: str = Field(..., alias="from", description="Start date in ISO format")
    to_date: str = Field(..., alias="to", description="End date in ISO format")
    
    @field_validator("from_date", "to_date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        """Validate ISO date format."""
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
            return v
        except ValueError:
            raise ValueError(f"Invalid date format: {v}. Expected ISO 8601 format (e.g., 2024-01-01)")
    
    def validate_range(self):
        """Validate that from_date is before to_date."""
        from_dt = datetime.fromisoformat(self.from_date.replace('Z', '+00:00'))
        to_dt = datetime.fromisoformat(self.to_date.replace('Z', '+00:00'))
        if from_dt > to_dt:
            raise ValueError("'from' date must be before 'to' date")


class Filters(BaseModel):
    """Search filters."""
    date_range: Optional[DateRange] = None


class SearchRequest(BaseModel):
    """Search request model."""
    query: str = Field(..., min_length=1, description="Search query text")
    user_id: str = Field(..., min_length=1, description="User identifier")
    indices: List[str] = Field(..., min_length=1, description="List of OpenSearch indices to search")
    filters: Optional[Filters] = None
    
    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate query is not empty after stripping."""
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()
    
    @field_validator("indices")
    @classmethod
    def validate_indices(cls, v: List[str]) -> List[str]:
        """Validate indices list."""
        if not v:
            raise ValueError("At least one index must be specified")
        for idx in v:
            if not idx.strip():
                raise ValueError("Index name cannot be empty")
        return [idx.strip() for idx in v]


class SearchResult(BaseModel):
    """Single search result."""
    doc_id: str = Field(..., description="Document identifier")
    doc_name: str = Field(..., description="Document name")
    text: str = Field(..., description="Document's text")
    snippet: str = Field(..., description="Relevant text snippet")
    score: float = Field(..., description="Relevance score")


class SearchResponse(BaseModel):
    """Search response model."""
    results: List[SearchResult] = Field(default_factory=list, description="List of search results")
    query_expanded: str = Field(..., description="Query expanded with synonyms")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")
    request_id: str = Field(..., description="Unique request identifier")


class ErrorDetail(BaseModel):
    """Error response detail."""
    code: int = Field(..., description="HTTP status code")
    message: str = Field(..., description="Human-readable error message")
    detail: Optional[str] = Field(None, description="Technical error details")
    request_id: str = Field(..., description="Request identifier for tracking")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: ErrorDetail


class HealthStatus(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Overall health status: healthy, degraded, or unhealthy")
    services: dict = Field(..., description="Status of individual services")
