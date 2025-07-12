"""
Job Monitoring API Models

Request models for job monitoring operations that were previously defined
in the API controller. Moving these to the domain layer ensures proper
architectural separation.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class JobProgressUpdateRequest(BaseModel):
    """Request model for updating job progress."""
    progress: float = Field(..., description="Progress percentage (0.0 to 100.0)")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional progress metadata")


class JobCancelRequest(BaseModel):
    """Request model for canceling a job."""
    reason: Optional[str] = Field(default=None, description="Optional reason for cancellation") 