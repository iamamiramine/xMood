import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI

from api.controllers import (
    encoder_controller,
    music_base_controller,
    health_controller,
    feature_extraction_controller,
    generator_controller,
    pipeline_config_controller,
    job_monitoring_controller,
    pipeline_job_controller,
)
from handlers.exception_handler import add_exception_handlers

# Import unified configuration and dependency injection
from application.shared.services.unified_config_service import get_unified_config_service
from application.shared.initialization.service_initialization import initialize_services, reset_services, get_service_health_check
from domain.interfaces.dependency_injection import get_dependency_container

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for cleanup
_config_service = None
_dependency_container = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    
    Handles:
    - Configuration service initialization
    - Dependency injection container setup
    - Service registration and health checks
    - Cleanup on shutdown
    """
    global _config_service, _dependency_container
    
    # Startup
    logger.info("Initializing PA-AI-2 Generative Module...")
    
    try:
        # Initialize unified configuration service
        _config_service = get_unified_config_service()
        
        # Set environment from environment variable or default to production
        env_name = os.getenv('PA_AI_ENVIRONMENT', 'production')
        logger.info(f"Setting configuration environment: {env_name}")
        _config_service.set_environment(env_name)
        
        # Load configuration
        config = _config_service.load_configuration()
        logger.info("Configuration loaded successfully")
        
        # Initialize dependency injection container
        _dependency_container = get_dependency_container()
        logger.info("Dependency injection container initialized")
        
        # Initialize and register all services
        initialize_services()
        logger.info("Services initialized and registered")
        
        # Perform health checks
        health_status = get_service_health_check()
        logger.info(f"Service health check completed: {health_status}")
        
        # Log configuration info
        config_info = _config_service.get_configuration_info()
        logger.info(f"Configuration info: {config_info}")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise
    
    # Shutdown
    logger.info("Shutting down PA-AI-2 Generative Module...")
    
    try:
        # Reset services
        reset_services()
        logger.info("Services reset completed")
        
        # Clear dependency container
        if _dependency_container:
            _dependency_container.clear()
            logger.info("Dependency container cleared")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
    
    logger.info("Application shutdown complete")


tags_metadata = [
    {
        "name": "health",
        "description": "checks the health of the API services",
    },
    {
        "name": "music_base",
        "description": "Music Base",
    },
    {
        "name": "encoder",
        "description": "Encoder",
    },
    {
        "name": "feature_extraction",
        "description": "Feature Extraction Module",
    },
    {
        "name": "generator",
        "description": "Generator",
    },
    {
        "name": "Pipeline Configuration",
        "description": "Unified configuration management for all services",
    },
    {
        "name": "Job Monitoring",
        "description": "Background job management and monitoring",
    },
    {
        "name": "Pipeline Jobs",
        "description": "Pipeline orchestration and execution",
    },
]


app = FastAPI(
    version="1.0",
    title="PA-AI API",
    description="API for PA-AI with unified configuration and dependency injection",
    openapi_tags=tags_metadata,
    lifespan=lifespan,
)

app.include_router(
    health_controller.router,
    prefix="/health",
    tags=["health"],
    responses={404: {"description": "Not found"}},
)
app.include_router(
    music_base_controller.router,
    prefix="/music_base",
    tags=["music_base"],
    responses={404: {"description": "Not found"}},
)
app.include_router(
    encoder_controller.router,
    prefix="/encoder",
    tags=["encoder"],
    responses={404: {"description": "Not found"}},
)
app.include_router(
    feature_extraction_controller.router,
    prefix="/feature_extraction",
    tags=["feature_extraction"],
    responses={404: {"description": "Not found"}},
)
app.include_router(
    generator_controller.router,
    prefix="/generator",
    tags=["generator"],
    responses={404: {"description": "Not found"}},
)
app.include_router(
    pipeline_config_controller.router,
    responses={404: {"description": "Not found"}},
)
app.include_router(
    job_monitoring_controller.router,
    responses={404: {"description": "Not found"}},
)
app.include_router(
    pipeline_job_controller.router,
    responses={404: {"description": "Not found"}},
)


add_exception_handlers(app=app)


# Add configuration endpoint for debugging
@app.get("/config/info")
async def get_config_info():
    """Get current configuration information."""
    try:
        if _config_service:
            return _config_service.get_configuration_info()
        return {"error": "Configuration service not initialized"}
    except Exception as e:
        return {"error": f"Failed to get configuration info: {e}"}


@app.get("/services/health")
async def get_services_health():
    """Get health status of all registered services."""
    try:
        health_status = get_service_health_check()
        return health_status
    except Exception as e:
        return {"error": f"Failed to get service health: {e}"}


# if __name__ == "__main__":
#     import uvicorn
#
#     uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
