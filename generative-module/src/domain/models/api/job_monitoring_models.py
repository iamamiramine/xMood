"""
Job Monitoring API Models

Request models for job monitoring operations that were previously defined
in the API controller. Moving these to the domain layer ensures proper
architectural separation.
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


class JobProgressUpdateRequest(BaseModel):
    """Request model for updating job progress."""
    progress: float = Field(..., description="Progress percentage (0.0 to 100.0)")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional progress metadata")


class JobCancelRequest(BaseModel):
    """Request model for canceling a job."""
    reason: Optional[str] = Field(default=None, description="Optional reason for cancellation")


class JobListResponse(BaseModel):
    """Response model for job list."""
    jobs: List[Dict[str, Any]]
    total_count: int
    filtered_count: int


class JobStatusResponse(BaseModel):
    """Response model for job status."""
    job_id: str
    status: str
    progress: float
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    duration: Optional[float]
    service_name: str
    function_name: str
    job_name: str
    pipeline_job_id: Optional[str]
    result: Optional[Dict[str, Any]]
    error: Optional[str]
    metadata: Optional[Dict[str, Any]]


class JobStatisticsResponse(BaseModel):
    """Response model for job statistics."""
    total_jobs: int
    running_jobs: int
    max_workers: int
    status_counts: Dict[str, int]
    service_counts: Dict[str, int]
    average_duration: float
    completed_jobs: int