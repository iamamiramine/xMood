"""
Pipeline Config API Models

Request models for pipeline configuration operations that were previously defined
in the API controller. Moving these to the domain layer ensures proper
architectural separation.
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


class PipelineConfigGenerateRequest(BaseModel):
    """Request model for generating pipeline configuration."""
    services: List[str] = Field(..., description="List of services to include in the pipeline")
    environment: str = Field(default="production", description="Environment configuration")
    job_name: Optional[str] = Field(default=None, description="Optional job name")
    overrides: Optional[Dict[str, Any]] = Field(default=None, description="Configuration overrides")


class PipelineConfigUpdateRequest(BaseModel):
    """Request model for updating pipeline job status."""
    status: str = Field(..., description="New status for the pipeline job")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional status metadata")
