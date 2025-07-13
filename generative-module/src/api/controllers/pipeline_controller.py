"""
Pipeline Controller

This module provides FastAPI endpoints for pipeline orchestration and job management.
It allows users to execute single jobs, full pipelines, and monitor job status.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Query

from domain.models.pipeline.pipeline_model import (
    PipelineJobRequest,
    PipelineExecutionRequest,
    ParameterSetupRequest,
    ParameterSetupResponse,
    JobStatus
)
from application.pipeline.services.pipeline_service import pipeline_service

router = APIRouter()


@router.post("/setup_parameters", response_model=ParameterSetupResponse)
async def setup_parameters(request: ParameterSetupRequest) -> dict:
    """
    Set up parameters for pipeline configuration and export to YAML file.
    
    This endpoint allows users to:
    1. Set common parameters that apply to all services
    2. Set service-specific parameters for individual functions
    3. Generate a FullConfig object with proper parameter merging
    4. Export the configuration as a YAML file with unique ID
    
    Args:
        request: ParameterSetupRequest containing common and service-specific parameters
        
    Returns:
        dict: Response with configuration ID, file path, and applied parameters
    """
    try:
        response = pipeline_service.setup_parameters(request)
        return {
            "status": "success",
            "message": "Parameters setup completed successfully",
            "data": {
                "config_id": response.config_id,
                "config_name": response.config_name,
                "config_file_path": response.config_file_path,
                "created_at": response.created_at.isoformat(),
                "validation_passed": response.validation_passed,
                "validation_warnings": response.validation_warnings,
                "summary": {
                    "common_parameters_count": len(response.common_parameters_applied),
                    "service_parameters_configured": list(response.service_parameters_applied.keys()),
                    "total_functions_configured": sum(
                        len(service_config) if isinstance(service_config, dict) else 0
                        for service_config in response.service_parameters_applied.values()
                        if service_config is not None
                    )
                }
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/configs")
async def list_configurations() -> dict:
    """
    List all available configuration files.
    
    Returns:
        dict: List of configuration files with metadata
    """
    try:
        from domain.constants.paths_constants import list_available_configs
        import os
        
        config_files = list_available_configs()
        
        configs_info = []
        for config_file in config_files:
            try:
                # Get file metadata
                stat = os.stat(config_file)
                file_size = stat.st_size
                modified_time = stat.st_mtime
                
                # Extract config name from filename
                filename = os.path.basename(config_file)
                config_name = filename.replace('.yaml', '')
                
                configs_info.append({
                    "file_path": config_file,
                    "config_name": config_name,
                    "file_size": file_size,
                    "modified_time": modified_time,
                    "modified_date": os.path.getmtime(config_file)
                })
            except Exception as e:
                # Skip files that can't be read
                continue
        
        return {
            "status": "success",
            "message": f"Found {len(configs_info)} configuration files",
            "data": {
                "configs": configs_info,
                "total_count": len(configs_info)
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/configs/set")
async def set_configuration(config_file_path: str) -> dict:
    """
    Set a specific configuration file to use for the pipeline.
    
    Args:
        config_file_path: Path to the configuration file to use
        
    Returns:
        dict: Response indicating success or failure
    """
    try:
        from domain.constants.paths_constants import set_config_file, get_config_file_path
        import os
        
        # Validate that the file exists
        if not os.path.exists(config_file_path):
            return {
                "status": "error", 
                "message": f"Configuration file not found: {config_file_path}"
            }
        
        # Set the configuration file
        set_config_file(config_file_path)
        
        # Verify it's being used
        current_config = get_config_file_path()
        
        return {
            "status": "success",
            "message": f"Configuration file set successfully",
            "data": {
                "config_file_path": config_file_path,
                "current_config": current_config,
                "is_active": current_config == config_file_path
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/configs/current")
async def get_current_configuration() -> dict:
    """
    Get the currently active configuration file.
    
    Returns:
        dict: Information about the current configuration
    """
    try:
        from domain.constants.paths_constants import get_current_config_file, get_config_file_path
        import os
        
        explicitly_set = get_current_config_file()
        current_config = get_config_file_path()
        
        config_info = {
            "explicitly_set": explicitly_set,
            "current_config": current_config,
            "is_using_latest": explicitly_set is None and current_config is not None
        }
        
        if current_config and os.path.exists(current_config):
            try:
                stat = os.stat(current_config)
                config_info.update({
                    "file_exists": True,
                    "file_size": stat.st_size,
                    "modified_time": stat.st_mtime,
                    "config_name": os.path.basename(current_config).replace('.yaml', '')
                })
            except Exception:
                config_info["file_exists"] = False
        else:
            config_info["file_exists"] = False
        
        return {
            "status": "success",
            "message": "Current configuration retrieved successfully",
            "data": config_info
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/configs/reset")
async def reset_configuration() -> dict:
    """
    Reset to use the latest configuration file (remove explicit setting).
    
    Returns:
        dict: Response indicating success or failure
    """
    try:
        from domain.constants.paths_constants import set_config_file, get_config_file_path
        
        # Reset to use latest config
        set_config_file(None)
        
        # Get the current config (should be latest)
        current_config = get_config_file_path()
        
        return {
            "status": "success",
            "message": "Configuration reset to use latest file",
            "data": {
                "current_config": current_config,
                "is_using_latest": True
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/execute_job")
async def execute_job(request: PipelineJobRequest) -> dict:
    """
    Execute a single pipeline job in the background.
    
    This endpoint uses the globally set configuration file. Make sure to set a configuration
    using the /configs/set endpoint before executing jobs.
    
    Args:
        request: PipelineJobRequest containing service and function to execute
        
    Returns:
        dict: Job execution response with job ID and status
    """
    try:
        # Validate that a configuration is set
        from domain.constants.paths_constants import get_config_file_path
        current_config = get_config_file_path()
        
        if not current_config:
            return {
                "status": "error",
                "message": "No configuration file is set. Please use /configs/set to set a configuration file first."
            }
        
        job_info = await pipeline_service.execute_job(request)
        return {
            "status": "success",
            "message": "Job submitted successfully",
            "data": {
                "job_id": job_info.job_id,
                "job_name": job_info.job_name,
                "service_name": job_info.service_name,
                "function_name": job_info.function_name,
                "status": job_info.status,
                "created_at": job_info.created_at.isoformat(),
                "config_used": current_config
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/execute_pipeline")
async def execute_pipeline(request: PipelineExecutionRequest) -> dict:
    """
    Execute a complete pipeline with multiple services.
    
    This endpoint uses the globally set configuration file. Make sure to set a configuration
    using the /configs/set endpoint before executing pipelines.
    
    Args:
        request: PipelineExecutionRequest containing pipeline configuration
        
    Returns:
        dict: Pipeline execution response with job IDs
    """
    try:
        # Validate that a configuration is set
        from domain.constants.paths_constants import get_config_file_path
        current_config = get_config_file_path()
        
        if not current_config:
            return {
                "status": "error",
                "message": "No configuration file is set. Please use /configs/set to set a configuration file first."
            }
        
        response = await pipeline_service.execute_pipeline(request)
        return {
            "status": "success",
            "message": "Pipeline started successfully",
            "data": {
                "pipeline_id": response.pipeline_id,
                "pipeline_name": response.pipeline_name,
                "job_ids": response.job_ids,
                "total_jobs": len(response.job_ids),
                "created_at": response.created_at.isoformat(),
                "config_used": current_config
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/job/{job_id}")
async def get_job_status(job_id: str) -> dict:
    """
    Get the status of a specific job.
    
    Args:
        job_id: Unique job identifier
        
    Returns:
        dict: Job status information
    """
    try:
        job_info = pipeline_service.get_job_status(job_id)
        
        if job_info is None:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Calculate duration if available
        duration = None
        if job_info.started_at and job_info.completed_at:
            duration = (job_info.completed_at - job_info.started_at).total_seconds()
        
        return {
            "status": "success",
            "data": {
                "job_id": job_info.job_id,
                "job_name": job_info.job_name,
                "service_name": job_info.service_name,
                "function_name": job_info.function_name,
                "status": job_info.status,
                "progress": job_info.progress,
                "created_at": job_info.created_at.isoformat(),
                "started_at": job_info.started_at.isoformat() if job_info.started_at else None,
                "completed_at": job_info.completed_at.isoformat() if job_info.completed_at else None,
                "duration_seconds": duration,
                "error_message": job_info.error_message,
                "result": job_info.result
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/jobs")
async def get_all_jobs(
    status: Optional[JobStatus] = Query(None, description="Filter jobs by status"),
    limit: Optional[int] = Query(None, description="Limit number of jobs returned"),
    offset: Optional[int] = Query(0, description="Offset for pagination")
) -> dict:
    """
    Get all pipeline jobs with optional filtering and pagination.
    
    Args:
        status: Optional status filter
        limit: Optional limit for pagination
        offset: Optional offset for pagination
        
    Returns:
        dict: List of jobs with pagination info
    """
    try:
        all_jobs = pipeline_service.get_all_jobs()
        
        # Filter by status if provided
        if status:
            all_jobs = [job for job in all_jobs if job.status == status]
        
        # Sort by creation time (newest first)
        all_jobs.sort(key=lambda x: x.created_at, reverse=True)
        
        # Apply pagination
        total_jobs = len(all_jobs)
        start_idx = offset
        end_idx = start_idx + limit if limit else len(all_jobs)
        paginated_jobs = all_jobs[start_idx:end_idx]
        
        # Format job data
        formatted_jobs = []
        for job in paginated_jobs:
            duration = None
            if job.started_at and job.completed_at:
                duration = (job.completed_at - job.started_at).total_seconds()
            
            formatted_jobs.append({
                "job_id": job.job_id,
                "job_name": job.job_name,
                "service_name": job.service_name,
                "function_name": job.function_name,
                "status": job.status,
                "progress": job.progress,
                "created_at": job.created_at.isoformat(),
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "duration_seconds": duration,
                "error_message": job.error_message
            })
        
        return {
            "status": "success",
            "data": {
                "jobs": formatted_jobs,
                "pagination": {
                    "total": total_jobs,
                    "limit": limit,
                    "offset": offset,
                    "count": len(formatted_jobs)
                }
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/job/{job_id}/cancel")
async def cancel_job(job_id: str) -> dict:
    """
    Cancel a running or pending job.
    
    Args:
        job_id: Unique job identifier
        
    Returns:
        dict: Cancellation result
    """
    try:
        success = pipeline_service.cancel_job(job_id)
        
        if success:
            return {
                "status": "success",
                "message": f"Job {job_id} cancelled successfully"
            }
        else:
            return {
                "status": "error",
                "message": f"Failed to cancel job {job_id}. Job may not exist or already completed."
            }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/jobs/summary")
async def get_jobs_summary() -> dict:
    """
    Get a summary of all jobs grouped by status.
    
    Returns:
        dict: Job summary with counts by status
    """
    try:
        all_jobs = pipeline_service.get_all_jobs()
        
        # Count jobs by status
        status_counts = {}
        for status in JobStatus:
            status_counts[status.value] = len([job for job in all_jobs if job.status == status])
        
        # Get recent jobs (last 10)
        recent_jobs = sorted(all_jobs, key=lambda x: x.created_at, reverse=True)[:10]
        
        # Format recent jobs
        formatted_recent = []
        for job in recent_jobs:
            formatted_recent.append({
                "job_id": job.job_id,
                "job_name": job.job_name,
                "service_name": job.service_name,
                "function_name": job.function_name,
                "status": job.status,
                "created_at": job.created_at.isoformat()
            })
        
        return {
            "status": "success",
            "data": {
                "total_jobs": len(all_jobs),
                "status_counts": status_counts,
                "recent_jobs": formatted_recent
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/jobs/cleanup")
async def cleanup_completed_jobs(
    max_age_hours: int = Query(24, description="Maximum age in hours for jobs to keep")
) -> dict:
    """
    Clean up completed jobs older than specified age.
    
    Args:
        max_age_hours: Maximum age in hours for jobs to keep
        
    Returns:
        dict: Cleanup result with number of jobs removed
    """
    try:
        removed_count = pipeline_service.cleanup_completed_jobs(max_age_hours)
        
        return {
            "status": "success",
            "message": f"Cleaned up {removed_count} completed jobs older than {max_age_hours} hours",
            "data": {
                "removed_count": removed_count,
                "max_age_hours": max_age_hours
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/health")
async def pipeline_health() -> dict:
    """
    Get pipeline service health status.
    
    Returns:
        dict: Health status information
    """
    try:
        all_jobs = pipeline_service.get_all_jobs()
        running_jobs = [job for job in all_jobs if job.status == JobStatus.RUNNING]
        
        return {
            "status": "success",
            "data": {
                "service": "pipeline",
                "healthy": True,
                "total_jobs": len(all_jobs),
                "running_jobs": len(running_jobs),
                "timestamp": "2024-12-20T10:00:00Z"
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e), "healthy": False} 