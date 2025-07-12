"""
Pipeline Job API Models

Request models for pipeline job operations that were previously defined
in the API controller. Moving these to the domain layer ensures proper
architectural separation.
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


class RunServiceRequest(BaseModel):
    """Request model for running a single service."""
    service_name: str = Field(..., description="Name of the service to run")
    function_name: str = Field(..., description="Name of the function to execute")
    job_priority: str = Field(default="normal", description="Job priority level")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional job metadata")


class RunPipelineRequest(BaseModel):
    """Request model for running a complete pipeline."""
    pipeline_definition: Optional[List[Dict[str, Any]]] = Field(
        default=None, 
        description="Custom pipeline definition (if not provided, runs all services)"
    )
    parallel_execution: bool = Field(
        default=False, 
        description="Whether to run services in parallel"
    )
    job_priority: str = Field(default="normal", description="Job priority level")


class PipelineDefinitionValidationRequest(BaseModel):
    """Request model for validating pipeline definition."""
    pipeline_definition: List[Dict[str, Any]] = Field(
        ..., 
        description="Pipeline definition to validate"
    )


class ServiceExecutionRequest(BaseModel):
    """Request model for executing a service with custom parameters."""
    service_name: str = Field(..., description="Name of the service")
    function_name: str = Field(..., description="Name of the function")
    parameters: Dict[str, Any] = Field(..., description="Function parameters")
    job_priority: str = Field(default="normal", description="Job priority level")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional job metadata") 