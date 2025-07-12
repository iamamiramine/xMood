from fastapi import APIRouter, HTTPException, Query, Body
from typing import Dict, Any, Optional, List
import os

from application.pipeline_config.services.pipeline_config_service import (
    pipeline_config_service,
    generate_unified_config,
    migrate_json_to_yaml,
    validate_unified_config,
    get_service_config_template,
    list_available_services,
    standardize_parameter_names
)
from application.shared.helpers.error_handlers import (
    handle_error_with_fallback,
    create_success_response,
)
from domain.exceptions.global_exceptions import ConfigurationError
from domain.models.api.pipeline_config_models import (
    PipelineConfigGenerateRequest,
    PipelineConfigUpdateRequest,
    UnifiedConfigGenerateRequest,
    ConfigMigrationRequest,
    ConfigValidationRequest,
    ConfigStandardizationRequest,
)




class ConfigValidationResponse(BaseModel):
    valid: bool
    message: str
    config_path: str
    services: Optional[List[str]] = None


class ConfigGenerationResponse(BaseModel):
    success: bool
    message: str
    output_path: str
    services_included: List[str]
    environment: str


class ServiceTemplateResponse(BaseModel):
    service_name: str
    template: Dict[str, Any]


# Router
router = APIRouter(prefix="/pipeline-config", tags=["Pipeline Configuration"])


@router.post("/generate", response_model=Dict[str, Any])
async def generate_pipeline_config(request: PipelineConfigGenerateRequest) -> Dict[str, Any]:
    """
    Generate a unique pipeline configuration for a specific job.
    
    Args:
        request: Pipeline configuration generation request
        
    Returns:
        Generated pipeline configuration with unique job ID
    """
    try:
        config = pipeline_config_service.generate_pipeline_config(
            services=request.services,
            environment=request.environment,
            job_name=request.job_name,
            overrides=request.overrides
        )
        return create_success_response(
            message="Pipeline configuration generated successfully",
            data={
                "job_id": config['pipeline_metadata']['job_id'],
                "config_path": config['pipeline_metadata']['config_path'],
                "config": config
            }
        )
    except Exception as e:
        raise handle_error_with_fallback(
            error=e,
            service_name="pipeline_config",
            function_name="generate_pipeline_config"
        )


@router.get("/jobs", response_model=List[Dict[str, Any]])
async def list_pipeline_jobs(status: Optional[str] = Query(None, description="Filter by status")) -> List[Dict[str, Any]]:
    """
    List all pipeline jobs, optionally filtered by status.
    
    Args:
        status: Optional status filter
        
    Returns:
        List of pipeline job information
    """
    try:
        jobs = pipeline_config_service.list_pipeline_jobs(status=status)
        return jobs
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list pipeline jobs: {str(e)}")


@router.get("/jobs/{job_id}", response_model=Dict[str, Any])
async def get_pipeline_job(job_id: str) -> Dict[str, Any]:
    """
    Get specific pipeline job configuration.
    
    Args:
        job_id: Unique job ID
        
    Returns:
        Pipeline job configuration
    """
    try:
        config = pipeline_config_service.get_pipeline_config(job_id)
        return {
            "status": "success",
            "job_id": job_id,
            "config": config
        }
    except ConfigurationError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get pipeline job: {str(e)}")


@router.put("/jobs/{job_id}/status", response_model=Dict[str, Any])
async def update_pipeline_job_status(job_id: str, request: PipelineConfigUpdateRequest) -> Dict[str, Any]:
    """
    Update pipeline job status.
    
    Args:
        job_id: Unique job ID
        request: Status update request
        
    Returns:
        Updated job information
    """
    try:
        job_info = pipeline_config_service.update_pipeline_status(
            job_id=job_id,
            status=request.status,
            metadata=request.metadata
        )
        return {
            "status": "success",
            "message": f"Pipeline job {job_id} status updated to {request.status}",
            "job_info": job_info
        }
    except ConfigurationError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update pipeline job status: {str(e)}")


@router.delete("/jobs/{job_id}", response_model=Dict[str, Any])
async def delete_pipeline_job(job_id: str) -> Dict[str, Any]:
    """
    Delete a pipeline job and its configuration.
    
    Args:
        job_id: Unique job ID
        
    Returns:
        Deletion confirmation
    """
    try:
        result = pipeline_config_service.delete_pipeline_job(job_id)
        return result
    except ConfigurationError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete pipeline job: {str(e)}")


@router.get("/jobs/{job_id}/services/{service_name}", response_model=Dict[str, Any])
async def get_service_config_from_pipeline(job_id: str, service_name: str) -> Dict[str, Any]:
    """
    Extract specific service configuration from pipeline config.
    
    Args:
        job_id: Unique job ID
        service_name: Name of the service
        
    Returns:
        Service-specific configuration
    """
    try:
        service_config = pipeline_config_service.get_service_config_from_pipeline(job_id, service_name)
        return {
            "status": "success",
            "job_id": job_id,
            "service_name": service_name,
            "config": service_config
        }
    except ConfigurationError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get service config: {str(e)}")


# Legacy unified config methods (kept for backward compatibility)
@router.post("/unified/generate", response_model=Dict[str, Any])
async def generate_unified_config_legacy(request: UnifiedConfigGenerateRequest) -> Dict[str, Any]:
    """
    Generate a unified YAML configuration file for all services (legacy method).
    
    Args:
        request: Unified configuration generation request
        
    Returns:
        Generated unified configuration
    """
    try:
        config = generate_unified_config(
            output_path=request.output_path,
            environment=request.environment,
            services=request.services,
            overrides=request.overrides
        )
        return {
            "status": "success",
            "message": "Unified configuration generated successfully",
            "output_path": request.output_path,
            "config": config
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate unified config: {str(e)}")


@router.post("/migrate", response_model=Dict[str, Any])
async def migrate_json_to_yaml_config(request: ConfigMigrationRequest) -> Dict[str, Any]:
    """
    Migrate existing JSON configuration to unified YAML format.
    
    Args:
        request: Configuration migration request
        
    Returns:
        Migrated configuration
    """
    try:
        config = migrate_json_to_yaml(
            json_config_path=request.json_config_path,
            yaml_output_path=request.yaml_output_path,
            apply_standardization=request.apply_standardization
        )
        return {
            "status": "success",
            "message": "Configuration migrated successfully",
            "input_path": request.json_config_path,
            "output_path": request.yaml_output_path,
            "config": config
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to migrate configuration: {str(e)}")


@router.post("/validate", response_model=Dict[str, Any])
async def validate_config(request: ConfigValidationRequest) -> Dict[str, Any]:
    """
    Validate a unified configuration file.
    
    Args:
        request: Configuration validation request
        
    Returns:
        Validation results
    """
    try:
        result = validate_unified_config(request.config_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to validate configuration: {str(e)}")


@router.get("/services", response_model=List[str])
async def get_available_services() -> List[str]:
    """
    List all available services in the configuration.
    
    Returns:
        List of available service names
    """
    try:
        services = list_available_services()
        return services
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list services: {str(e)}")


@router.get("/services/{service_name}/template", response_model=Dict[str, Any])
async def get_service_template(service_name: str) -> Dict[str, Any]:
    """
    Get a configuration template for a specific service.
    
    Args:
        service_name: Name of the service
        
    Returns:
        Service configuration template
    """
    try:
        template = get_service_config_template(service_name)
        return {
            "status": "success",
            "service_name": service_name,
            "template": template
        }
    except ConfigurationError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get service template: {str(e)}")


@router.post("/standardize", response_model=Dict[str, Any])
async def standardize_config_parameters(request: ConfigStandardizationRequest) -> Dict[str, Any]:
    """
    Apply parameter name standardization to configuration.
    
    Args:
        request: Configuration standardization request
        
    Returns:
        Standardized configuration
    """
    try:
        standardized_config = standardize_parameter_names(request.config)
        return {
            "status": "success",
            "message": "Configuration parameters standardized successfully",
            "config": standardized_config
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to standardize configuration: {str(e)}")


@router.get("/health")
async def health_check():
    """
    Health check endpoint for the pipeline configuration service.
    
    Returns service health status.
    """
    return {
        "status": "healthy",
        "service": "pipeline-config",
        "version": "1.0.0"
    } 