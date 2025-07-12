from fastapi import APIRouter, HTTPException, Query, Body
from typing import Dict, Any, Optional, List
import os

from application.pipeline_config.services.pipeline_config_service import (
    pipeline_config_service,
    get_service_config_template,
    list_available_services,
)
from handlers.exception_handler import (
    handle_error_with_fallback,
    create_success_response,
)
from domain.exceptions.global_exceptions import ConfigurationException
from domain.models.api.pipeline_config_models import (
    PipelineConfigGenerateRequest,
    PipelineConfigUpdateRequest,
)

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
    except ConfigurationException as e:
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
    except ConfigurationException as e:
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
    except ConfigurationException as e:
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
    except ConfigurationException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get service config: {str(e)}")


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
    except ConfigurationException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get service template: {str(e)}")


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