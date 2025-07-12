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


class UnifiedConfigGenerateRequest(BaseModel):
    """Request model for generating unified configuration."""
    output_path: str = Field(..., description="Path where the unified config will be saved")
    environment: str = Field(default="production", description="Environment configuration")
    services: Optional[List[str]] = Field(default=None, description="Optional list of services to include")
    overrides: Optional[Dict[str, Any]] = Field(default=None, description="Configuration overrides")


class ConfigMigrationRequest(BaseModel):
    """Request model for migrating JSON to YAML configuration."""
    json_config_path: str = Field(..., description="Path to the JSON configuration file")
    yaml_output_path: str = Field(..., description="Path where the YAML configuration will be saved")
    apply_standardization: bool = Field(default=True, description="Whether to apply parameter standardization")


class ConfigValidationRequest(BaseModel):
    """Request model for validating configuration."""
    config_path: str = Field(..., description="Path to the configuration file to validate")


class ConfigStandardizationRequest(BaseModel):
    """Request model for standardizing parameter names."""
    config: Dict[str, Any] = Field(..., description="Configuration to standardize") 