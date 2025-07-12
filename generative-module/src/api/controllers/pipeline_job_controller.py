from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional, List
from datetime import datetime

from application.pipeline_job.services.pipeline_job_service import pipeline_job_service
from application.shared.services.job_management_service import JobPriority
from application.shared.helpers import enum_helpers
from handlers.exception_handler import (
    handle_error_with_fallback,
    create_success_response,
)
from domain.exceptions.global_exceptions import ConfigurationException
from domain.models.api.pipeline_job_models import (
    RunServiceRequest,
    RunPipelineRequest,
    PipelineDefinitionValidationRequest,
    ServiceExecutionRequest,
)

router = APIRouter(prefix="/pipeline-jobs", tags=["Pipeline Jobs"])





@router.post("/{pipeline_job_id}/services/run", response_model=Dict[str, Any])
async def run_service_from_pipeline(
    pipeline_job_id: str,
    request: RunServiceRequest
) -> Dict[str, Any]:
    """
    Run a single service using pipeline configuration.
    
    Args:
        pipeline_job_id: Pipeline job ID
        request: Service execution request
        
    Returns:
        Service execution result with background job ID
    """
    try:
        # Parse job priority
        job_priority = enum_helpers.parse_job_priority(request.job_priority)
        
        # Run service from pipeline config
        job_id = pipeline_job_service.run_service_from_pipeline_config(
            pipeline_job_id=pipeline_job_id,
            service_name=request.service_name,
            function_name=request.function_name,
            job_priority=job_priority,
            metadata=request.metadata
        )
        
        return create_success_response(
            message=f"Service {request.service_name}.{request.function_name} started successfully",
            data={
                "pipeline_job_id": pipeline_job_id,
                "service_name": request.service_name,
                "function_name": request.function_name,
                "background_job_id": job_id,
                "job_priority": request.job_priority
            }
        )
        
    except Exception as e:
        raise handle_error_with_fallback(
            error=e,
            service_name=request.service_name,
            function_name=request.function_name,
            job_id=None
        )


@router.post("/{pipeline_job_id}/services/execute", response_model=Dict[str, Any])
async def execute_service_with_parameters(
    pipeline_job_id: str,
    request: ServiceExecutionRequest
) -> Dict[str, Any]:
    """
    Execute a service with custom parameters (not from pipeline config).
    
    Args:
        pipeline_job_id: Pipeline job ID
        request: Service execution request with custom parameters
        
    Returns:
        Service execution result with background job ID
    """
    try:
        # Parse job priority
        job_priority = enum_helpers.parse_job_priority(request.job_priority)
        
        # Parse parameters to BaseModel
        parameters = pipeline_job_service.parse_service_config_to_parameters(
            service_name=request.service_name,
            function_name=request.function_name,
            config=request.parameters
        )
        
        # Execute service function
        job_id = pipeline_job_service.execute_service_function(
            pipeline_job_id=pipeline_job_id,
            service_name=request.service_name,
            function_name=request.function_name,
            parameters=parameters,
            job_priority=job_priority,
            metadata=request.metadata
        )
        
        return {
            "status": "success",
            "message": f"Service {request.service_name}.{request.function_name} started successfully",
            "pipeline_job_id": pipeline_job_id,
            "service_name": request.service_name,
            "function_name": request.function_name,
            "background_job_id": job_id,
            "job_priority": request.job_priority
        }
        
    except ConfigurationException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to execute service: {str(e)}")


@router.post("/{pipeline_job_id}/run", response_model=Dict[str, Any])
async def run_pipeline(
    pipeline_job_id: str,
    request: RunPipelineRequest
) -> Dict[str, Any]:
    """
    Run a complete pipeline with multiple services.
    
    Args:
        pipeline_job_id: Pipeline job ID
        request: Pipeline execution request
        
    Returns:
        Pipeline execution result with all job IDs
    """
    try:
        # Parse job priority
        job_priority = enum_helpers.parse_job_priority(request.job_priority)
        
        # Run pipeline
        result = pipeline_job_service.run_pipeline(
            pipeline_job_id=pipeline_job_id,
            pipeline_definition=request.pipeline_definition,
            parallel_execution=request.parallel_execution,
            job_priority=job_priority
        )
        
        return {
            "status": "success",
            "message": f"Pipeline started successfully with {result['total_jobs']} jobs",
            "pipeline_job_id": pipeline_job_id,
            "execution_mode": result["execution_mode"],
            "total_jobs": result["total_jobs"],
            "submitted_jobs": result["submitted_jobs"],
            "parallel_execution": request.parallel_execution,
            "job_priority": request.job_priority
        }
        
    except ConfigurationException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to run pipeline: {str(e)}")


@router.get("/{pipeline_job_id}/status", response_model=Dict[str, Any])
async def get_pipeline_status(pipeline_job_id: str) -> Dict[str, Any]:
    """
    Get comprehensive pipeline status including all sub-jobs.
    
    Args:
        pipeline_job_id: Pipeline job ID
        
    Returns:
        Pipeline status with sub-job details
    """
    try:
        status = pipeline_job_service.get_pipeline_status(pipeline_job_id)
        
        return {
            "status": "success",
            "pipeline_status": status
        }
        
    except ConfigurationException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get pipeline status: {str(e)}")


@router.post("/{pipeline_job_id}/cancel", response_model=Dict[str, Any])
async def cancel_pipeline(pipeline_job_id: str) -> Dict[str, Any]:
    """
    Cancel a pipeline and all its sub-jobs.
    
    Args:
        pipeline_job_id: Pipeline job ID
        
    Returns:
        Cancellation result
    """
    try:
        result = pipeline_job_service.cancel_pipeline(pipeline_job_id)
        
        return {
            "status": "success",
            "message": result["message"],
            "cancellation_result": result
        }
        
    except ConfigurationException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel pipeline: {str(e)}")


@router.get("/services/available", response_model=Dict[str, List[str]])
async def get_available_services() -> Dict[str, List[str]]:
    """
    Get available services and their functions.
    
    Returns:
        Dictionary of services and their available functions
    """
    try:
        services = pipeline_job_service.get_available_services()
        return services
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get available services: {str(e)}")


@router.post("/validate/pipeline-definition", response_model=Dict[str, Any])
async def validate_pipeline_definition(
    request: PipelineDefinitionValidationRequest
) -> Dict[str, Any]:
    """
    Validate a pipeline definition.
    
    Args:
        request: Pipeline definition validation request
        
    Returns:
        Validation result
    """
    try:
        validation_result = pipeline_job_service.validate_pipeline_definition(
            request.pipeline_definition
        )
        
        return {
            "status": "success",
            "validation_result": validation_result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to validate pipeline definition: {str(e)}")


@router.get("/services/{service_name}/functions", response_model=List[str])
async def get_service_functions(service_name: str) -> List[str]:
    """
    Get available functions for a specific service.
    
    Args:
        service_name: Name of the service
        
    Returns:
        List of available functions for the service
    """
    try:
        services = pipeline_job_service.get_available_services()
        
        if service_name not in services:
            raise HTTPException(status_code=404, detail=f"Service '{service_name}' not found")
        
        return services[service_name]
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get service functions: {str(e)}")


@router.get("/services/{service_name}/functions/{function_name}/parameters", response_model=Dict[str, Any])
async def get_service_function_parameters(service_name: str, function_name: str) -> Dict[str, Any]:
    """
    Get parameter schema for a specific service function.
    
    Args:
        service_name: Name of the service
        function_name: Name of the function
        
    Returns:
        Parameter schema for the service function
    """
    try:
        # Get parameter model class
        parameter_registry = pipeline_job_service.parameter_registry
        
        if service_name not in parameter_registry:
            raise HTTPException(status_code=404, detail=f"Service '{service_name}' not found")
        
        service_params = parameter_registry[service_name]
        if function_name not in service_params:
            raise HTTPException(
                status_code=404, 
                detail=f"Function '{function_name}' not found for service '{service_name}'"
            )
        
        parameter_model = service_params[function_name]
        
        # Get the schema from the Pydantic model
        schema = parameter_model.model_json_schema()
        
        return {
            "status": "success",
            "service_name": service_name,
            "function_name": function_name,
            "parameter_schema": schema
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get parameter schema: {str(e)}")


@router.get("/health/check", response_model=Dict[str, Any])
async def pipeline_job_health_check() -> Dict[str, Any]:
    """
    Health check for the pipeline job system.
    
    Returns:
        Health status and system information
    """
    try:
        # Get available services
        services = pipeline_job_service.get_available_services()
        
        # Get service registry stats
        total_services = len(services)
        total_functions = sum(len(functions) for functions in services.values())
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "system_info": {
                "total_services": total_services,
                "total_functions": total_functions,
                "available_services": list(services.keys())
            },
            "services": services
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "message": "Pipeline job system health check failed"
        }


@router.get("/examples/pipeline-definitions", response_model=Dict[str, Any])
async def get_example_pipeline_definitions() -> Dict[str, Any]:
    """
    Get example pipeline definitions for different use cases.
    
    Returns:
        Example pipeline definitions
    """
    try:
        examples = {
            "full_training_pipeline": [
                {"service": "encoder", "function": "encode_dataset", "dependencies": []},
                {"service": "feature_extraction", "function": "extract_symbolic_features", "dependencies": ["encoder"]},
                {"service": "music_base", "function": "extract_chords", "dependencies": ["encoder"]},
                {"service": "feature_extraction", "function": "train_vae", "dependencies": ["feature_extraction"]},
                {"service": "feature_extraction", "function": "generate_latent_representations", "dependencies": ["feature_extraction"]},
                {"service": "multimodal_mapping", "function": "train", "dependencies": ["feature_extraction"]},
                {"service": "generator", "function": "train", "dependencies": ["multimodal_mapping"]},
            ],
            "inference_pipeline": [
                {"service": "encoder", "function": "encode_dataset", "dependencies": []},
                {"service": "feature_extraction", "function": "extract_symbolic_features", "dependencies": ["encoder"]},
                {"service": "multimodal_mapping", "function": "generate_representations", "dependencies": ["feature_extraction"]},
                {"service": "generator", "function": "generate_from_midi", "dependencies": ["multimodal_mapping"]},
            ],
            "preprocessing_pipeline": [
                {"service": "encoder", "function": "encode_dataset", "dependencies": []},
                {"service": "encoder", "function": "tokenize_remi_dataset", "dependencies": ["encoder"]},
                {"service": "feature_extraction", "function": "extract_symbolic_features", "dependencies": ["encoder"]},
                {"service": "music_base", "function": "extract_chords", "dependencies": ["encoder"]},
            ]
        }
        
        return {
            "status": "success",
            "examples": examples,
            "total_examples": len(examples)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get example pipeline definitions: {str(e)}")


@router.get("/tutorials/getting-started", response_model=Dict[str, Any])
async def get_getting_started_tutorial() -> Dict[str, Any]:
    """
    Get a getting started tutorial for using pipeline jobs.
    
    Returns:
        Tutorial information and examples
    """
    try:
        tutorial = {
            "title": "Pipeline Jobs Getting Started Tutorial",
            "description": "Learn how to use the Pipeline Jobs system to orchestrate machine learning workflows",
            "steps": [
                {
                    "step": 1,
                    "title": "Generate Pipeline Configuration",
                    "description": "Create a unique pipeline configuration for your job",
                    "endpoint": "POST /pipeline-config/generate",
                    "example": {
                        "services": ["encoder", "feature_extraction"],
                        "environment": "development",
                        "job_name": "my_first_pipeline"
                    }
                },
                {
                    "step": 2,
                    "title": "Run Individual Service",
                    "description": "Execute a single service from your pipeline",
                    "endpoint": "POST /pipeline-jobs/{pipeline_job_id}/services/run",
                    "example": {
                        "service_name": "encoder",
                        "function_name": "encode_dataset",
                        "job_priority": "normal"
                    }
                },
                {
                    "step": 3,
                    "title": "Monitor Job Progress",
                    "description": "Check the status of your background jobs",
                    "endpoint": "GET /jobs/{job_id}",
                    "example": "Returns job status, progress, and results"
                },
                {
                    "step": 4,
                    "title": "Run Complete Pipeline",
                    "description": "Execute multiple services in sequence or parallel",
                    "endpoint": "POST /pipeline-jobs/{pipeline_job_id}/run",
                    "example": {
                        "parallel_execution": true,
                        "job_priority": "high"
                    }
                },
                {
                    "step": 5,
                    "title": "Get Pipeline Status",
                    "description": "Monitor overall pipeline progress",
                    "endpoint": "GET /pipeline-jobs/{pipeline_job_id}/status",
                    "example": "Returns comprehensive pipeline status with all sub-jobs"
                }
            ],
            "best_practices": [
                "Use development environment for testing",
                "Start with individual services before running full pipelines",
                "Monitor job progress regularly",
                "Use appropriate job priorities",
                "Validate pipeline definitions before execution"
            ],
            "troubleshooting": [
                "Check job logs if execution fails",
                "Verify configuration parameters",
                "Ensure required data files are available",
                "Monitor system resources"
            ]
        }
        
        return {
            "status": "success",
            "tutorial": tutorial
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get tutorial: {str(e)}") 