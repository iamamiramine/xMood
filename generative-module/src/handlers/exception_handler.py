from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
from pydantic import ValidationError
import logging
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, Union

from domain.exceptions.global_exceptions import *
from handlers.exception_handler import (
    create_error_response,
    generate_request_id,
    handle_configuration_error,
    handle_validation_error,
    handle_not_found_error,
    handle_generic_error,
)

from domain.exceptions.global_exceptions import (
    ConfigurationException,
    NotFoundException,
    GenericException,
)

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


def add_exception_handlers(app: FastAPI):
    @app.exception_handler(GenericException)
    async def generic_exception_handler(request: Request, exc: GenericException):
        """Handle custom generic exceptions with standardized format."""
        request_id = generate_request_id()
        error_response = create_error_response(
            error_type="GenericException",
            message=exc.name or "Generic error occurred",
            details=exc.message,
            request_id=request_id,
            additional_info=exc.info
        )
        
        logger.error(f"Generic exception [{request_id}]: {exc.name} - {exc.message}")
        return JSONResponse(
            status_code=400,
            content=error_response
        )

    @app.exception_handler(NotFoundException)
    async def not_found_exception_handler(request: Request, exc: NotFoundException):
        """Handle not found exceptions with standardized format."""
        request_id = generate_request_id()
        error_response = create_error_response(
            error_type="NotFoundException",
            message=exc.name or "Resource not found",
            details=exc.message,
            request_id=request_id,
            additional_info=exc.info
        )
        
        logger.error(f"Not found exception [{request_id}]: {exc.name} - {exc.message}")
        return JSONResponse(
            status_code=404,
            content=error_response
        )

    @app.exception_handler(ConfigurationException)
    async def configuration_error_handler(request: Request, exc: ConfigurationException):
        """Handle configuration errors with standardized format."""
        request_id = generate_request_id()
        error_response = create_error_response(
            error_type="ConfigurationException",
            message=str(exc),
            details=getattr(exc, 'info', None),
            request_id=request_id
        )
        
        logger.error(f"Configuration error [{request_id}]: {exc}")
        return JSONResponse(
            status_code=400,
            content=error_response
        )

    @app.exception_handler(ValidationError)
    async def validation_error_handler(request: Request, exc: ValidationError):
        """Handle Pydantic validation errors with standardized format."""
        request_id = generate_request_id()
        
        validation_errors = []
        for err in exc.errors():
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
        
        logger.error(f"Validation error [{request_id}]: {exc}")
        return JSONResponse(
            status_code=422,
            content=error_response
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        """Handle value errors with standardized format."""
        request_id = generate_request_id()
        error_response = create_error_response(
            error_type="ValueError",
            message=f"Invalid input: {str(exc)}",
            request_id=request_id
        )
        
        logger.error(f"Value error [{request_id}]: {exc}")
        return JSONResponse(
            status_code=400,
            content=error_response
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        """Handle all other exceptions with standardized format."""
        request_id = generate_request_id()
        error_response = create_error_response(
            error_type="InternalError",
            message="An unexpected error occurred",
            details=str(exc),
            request_id=request_id
        )
        
        logger.error(f"Unhandled exception [{request_id}]: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content=error_response
        )


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
        error_type: Type of error (e.g., 'ValidationError', 'ConfigurationException')
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
    error: ConfigurationException,
    request_id: Optional[str] = None
) -> HTTPException:
    """
    Handle ConfigurationException and return standardized HTTP exception.
    
    Args:
        error: Configuration error instance
        request_id: Request ID for tracking
        
    Returns:
        HTTPException with standardized error response
    """
    error_response = create_error_response(
        error_type="ConfigurationException",
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
    if isinstance(error, ConfigurationException):
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