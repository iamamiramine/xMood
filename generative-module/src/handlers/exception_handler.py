from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
from pydantic import ValidationError
import logging

from domain.exceptions.global_exceptions import *
from application.shared.helpers.error_handlers import (
    create_error_response,
    generate_request_id,
    handle_configuration_error,
    handle_validation_error,
    handle_not_found_error,
    handle_generic_error,
)

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

    @app.exception_handler(ConfigurationError)
    async def configuration_error_handler(request: Request, exc: ConfigurationError):
        """Handle configuration errors with standardized format."""
        request_id = generate_request_id()
        error_response = create_error_response(
            error_type="ConfigurationError",
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
