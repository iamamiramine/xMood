"""
Standardized Error Handlers

This module provides utility functions for consistent error handling across
all API controllers. It ensures that errors are returned in a standardized
format and properly logged.
"""

import logging
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, Union
from fastapi import HTTPException
from pydantic import ValidationError

from domain.exceptions.global_exceptions import (
    ConfigurationError,
    NotFoundException,
    GenericException,
)

logger = logging.getLogger(__name__)


def generate_request_id() -> str:
    """Generate a unique request ID for error tracking."""
    from uuid import uuid4
    return str(uuid4())[:8]


def create_error_response(
    error_type: str,
    message: str,
    details: Optional[str] = None,
    request_id: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Create a standardized error response.
    
    Args:
        error_type: Type of error (e.g., 'ValidationError', 'ConfigurationError')
        message: Human-readable error message
        details: Additional error details
        request_id: Request ID for tracking
        **kwargs: Additional fields for the error response
        
    Returns:
        Standardized error response dictionary
    """
    if request_id is None:
        request_id = generate_request_id()
    
    error_response = {
        "status": "error",
        "error_type": error_type,
        "message": message,
        "timestamp": datetime.now().isoformat(),
        "request_id": request_id,
    }
    
    if details:
        error_response["details"] = details
    
    # Add any additional fields
    error_response.update(kwargs)
    
    return error_response


def handle_configuration_error(
    error: ConfigurationError,
    request_id: Optional[str] = None
) -> HTTPException:
    """
    Handle ConfigurationError and return standardized HTTP exception.
    
    Args:
        error: Configuration error instance
        request_id: Request ID for tracking
        
    Returns:
        HTTPException with standardized error response
    """
    error_response = create_error_response(
        error_type="ConfigurationError",
        message=str(error),
        details=getattr(error, 'info', None),
        request_id=request_id
    )
    
    logger.error(f"Configuration error [{request_id}]: {error}")
    return HTTPException(status_code=400, detail=error_response)


def handle_validation_error(
    error: ValidationError,
    request_id: Optional[str] = None
) -> HTTPException:
    """
    Handle Pydantic ValidationError and return standardized HTTP exception.
    
    Args:
        error: Pydantic validation error instance
        request_id: Request ID for tracking
        
    Returns:
        HTTPException with standardized error response
    """
    validation_errors = []
    for err in error.errors():
        validation_errors.append({
            "field": ".".join(str(x) for x in err.get("loc", [])),
            "message": err.get("msg", "Validation error"),
            "type": err.get("type", "unknown"),
        })
    
    error_response = create_error_response(
        error_type="ValidationError",
        message="Request validation failed",
        request_id=request_id,
        validation_errors=validation_errors
    )
    
    logger.error(f"Validation error [{request_id}]: {error}")
    return HTTPException(status_code=422, detail=error_response)


def handle_not_found_error(
    error: NotFoundException,
    request_id: Optional[str] = None
) -> HTTPException:
    """
    Handle NotFoundException and return standardized HTTP exception.
    
    Args:
        error: Not found error instance
        request_id: Request ID for tracking
        
    Returns:
        HTTPException with standardized error response
    """
    error_response = create_error_response(
        error_type="NotFoundException",
        message=str(error),
        details=getattr(error, 'info', None),
        request_id=request_id
    )
    
    logger.error(f"Not found error [{request_id}]: {error}")
    return HTTPException(status_code=404, detail=error_response)


def handle_service_error(
    error: Exception,
    service_name: str,
    function_name: str,
    job_id: Optional[str] = None,
    request_id: Optional[str] = None
) -> HTTPException:
    """
    Handle service execution errors and return standardized HTTP exception.
    
    Args:
        error: Service execution error instance
        service_name: Name of the service that failed
        function_name: Name of the function that failed
        job_id: Background job ID if applicable
        request_id: Request ID for tracking
        
    Returns:
        HTTPException with standardized error response
    """
    error_response = create_error_response(
        error_type="ServiceError",
        message=f"Service {service_name}.{function_name} failed: {str(error)}",
        details=traceback.format_exc(),
        request_id=request_id,
        service_name=service_name,
        function_name=function_name,
        job_id=job_id
    )
    
    logger.error(f"Service error [{request_id}]: {service_name}.{function_name} - {error}")
    return HTTPException(status_code=500, detail=error_response)


def handle_generic_error(
    error: Exception,
    message: Optional[str] = None,
    request_id: Optional[str] = None
) -> HTTPException:
    """
    Handle generic errors and return standardized HTTP exception.
    
    Args:
        error: Generic error instance
        message: Custom error message (optional)
        request_id: Request ID for tracking
        
    Returns:
        HTTPException with standardized error response
    """
    error_message = message or f"An unexpected error occurred: {str(error)}"
    
    error_response = create_error_response(
        error_type="InternalError",
        message=error_message,
        details=traceback.format_exc(),
        request_id=request_id
    )
    
    logger.error(f"Generic error [{request_id}]: {error}")
    return HTTPException(status_code=500, detail=error_response)


def handle_error_with_fallback(
    error: Exception,
    service_name: Optional[str] = None,
    function_name: Optional[str] = None,
    job_id: Optional[str] = None,
    request_id: Optional[str] = None
) -> HTTPException:
    """
    Handle any error with automatic fallback to appropriate handler.
    
    Args:
        error: Error instance
        service_name: Name of the service (if applicable)
        function_name: Name of the function (if applicable)
        job_id: Background job ID (if applicable)
        request_id: Request ID for tracking
        
    Returns:
        HTTPException with standardized error response
    """
    if request_id is None:
        request_id = generate_request_id()
    
    # Handle specific error types
    if isinstance(error, ConfigurationError):
        return handle_configuration_error(error, request_id)
    elif isinstance(error, ValidationError):
        return handle_validation_error(error, request_id)
    elif isinstance(error, NotFoundException):
        return handle_not_found_error(error, request_id)
    elif isinstance(error, ValueError):
        return handle_generic_error(error, f"Invalid input: {str(error)}", request_id)
    elif service_name and function_name:
        return handle_service_error(error, service_name, function_name, job_id, request_id)
    else:
        return handle_generic_error(error, request_id=request_id)


def create_success_response(
    message: str,
    data: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Create a standardized success response.
    
    Args:
        message: Success message
        data: Response data (optional)
        request_id: Request ID for tracking
        **kwargs: Additional fields for the response
        
    Returns:
        Standardized success response dictionary
    """
    if request_id is None:
        request_id = generate_request_id()
    
    success_response = {
        "status": "success",
        "message": message,
        "timestamp": datetime.now().isoformat(),
        "request_id": request_id,
    }
    
    if data:
        success_response["data"] = data
    
    # Add any additional fields
    success_response.update(kwargs)
    
    return success_response 