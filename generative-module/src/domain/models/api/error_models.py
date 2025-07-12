"""
Error Response Models

Standardized error and success response models for consistent API responses.
These models ensure all endpoints return errors in the same format.
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


class StandardErrorResponse(BaseModel):
    """Standard error response model used across all API endpoints."""
    status: str = Field(default="error", description="Response status")
    error_type: str = Field(..., description="Type of error (e.g., 'ValidationError', 'ConfigurationError')")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[str] = Field(default=None, description="Additional error details")
    timestamp: Optional[str] = Field(default=None, description="Error timestamp")
    request_id: Optional[str] = Field(default=None, description="Request ID for tracking")


class ValidationErrorResponse(BaseModel):
    """Validation error response model for input validation failures."""
    status: str = Field(default="error", description="Response status")
    error_type: str = Field(default="ValidationError", description="Error type")
    message: str = Field(..., description="Human-readable error message")
    validation_errors: List[Dict[str, Any]] = Field(..., description="List of validation errors")
    timestamp: Optional[str] = Field(default=None, description="Error timestamp")
    request_id: Optional[str] = Field(default=None, description="Request ID for tracking")


class SuccessResponse(BaseModel):
    """Standard success response model for consistent API responses."""
    status: str = Field(default="success", description="Response status")
    message: str = Field(..., description="Success message")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Response data")
    timestamp: Optional[str] = Field(default=None, description="Response timestamp")
    request_id: Optional[str] = Field(default=None, description="Request ID for tracking")


class ConfigurationErrorResponse(BaseModel):
    """Configuration error response model for configuration-related failures."""
    status: str = Field(default="error", description="Response status")
    error_type: str = Field(default="ConfigurationError", description="Error type")
    message: str = Field(..., description="Human-readable error message")
    config_path: Optional[str] = Field(default=None, description="Configuration file path")
    missing_keys: Optional[List[str]] = Field(default=None, description="Missing configuration keys")
    invalid_values: Optional[Dict[str, str]] = Field(default=None, description="Invalid configuration values")
    timestamp: Optional[str] = Field(default=None, description="Error timestamp")
    request_id: Optional[str] = Field(default=None, description="Request ID for tracking")


class ServiceErrorResponse(BaseModel):
    """Service error response model for service execution failures."""
    status: str = Field(default="error", description="Response status")
    error_type: str = Field(default="ServiceError", description="Error type")
    message: str = Field(..., description="Human-readable error message")
    service_name: Optional[str] = Field(default=None, description="Name of the service that failed")
    function_name: Optional[str] = Field(default=None, description="Name of the function that failed")
    job_id: Optional[str] = Field(default=None, description="Background job ID if applicable")
    timestamp: Optional[str] = Field(default=None, description="Error timestamp")
    request_id: Optional[str] = Field(default=None, description="Request ID for tracking") 