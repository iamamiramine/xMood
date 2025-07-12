from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional, List
from datetime import datetime

from application.shared.services.job_management_service import (
    job_manager,
    JobStatus,
    JobPriority,
    JobInfo
)
from application.shared.helpers.error_handlers import (
    handle_error_with_fallback,
    create_success_response,
)
from domain.exceptions.global_exceptions import ConfigurationError
from domain.models.api.job_monitoring_models import (
    JobProgressUpdateRequest,
    JobCancelRequest,
)

router = APIRouter(prefix="/jobs", tags=["Job Monitoring"])


class JobListResponse(BaseModel):
    """Response model for job list."""
    jobs: List[Dict[str, Any]]
    total_count: int
    filtered_count: int


class JobStatusResponse(BaseModel):
    """Response model for job status."""
    job_id: str
    status: str
    progress: float
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    duration: Optional[float]
    service_name: str
    function_name: str
    job_name: str
    pipeline_job_id: Optional[str]
    result: Optional[Dict[str, Any]]
    error: Optional[str]
    metadata: Optional[Dict[str, Any]]


class JobStatisticsResponse(BaseModel):
    """Response model for job statistics."""
    total_jobs: int
    running_jobs: int
    max_workers: int
    status_counts: Dict[str, int]
    service_counts: Dict[str, int]
    average_duration: float
    completed_jobs: int


@router.get("/", response_model=JobListResponse)
async def list_jobs(
    service_name: Optional[str] = Query(None, description="Filter by service name"),
    status: Optional[str] = Query(None, description="Filter by job status"),
    pipeline_job_id: Optional[str] = Query(None, description="Filter by pipeline job ID"),
    limit: int = Query(100, description="Maximum number of jobs to return", ge=1, le=1000)
) -> JobListResponse:
    """
    List jobs with optional filtering.
    
    Args:
        service_name: Filter by service name
        status: Filter by job status
        pipeline_job_id: Filter by pipeline job ID
        limit: Maximum number of jobs to return
        
    Returns:
        List of jobs with metadata
    """
    try:
        # Convert string status to JobStatus enum
        job_status = None
        if status:
            try:
                job_status = JobStatus(status)
            except ValueError:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid status: {status}. Valid values: {[s.value for s in JobStatus]}"
                )
        
        # Get jobs from manager
        jobs = job_manager.list_jobs(
            service_name=service_name,
            status=job_status,
            pipeline_job_id=pipeline_job_id,
            limit=limit
        )
        
        # Convert to dictionaries
        job_dicts = [job.to_dict() for job in jobs]
        
        # Get total count (without filters)
        all_jobs = job_manager.list_jobs(limit=10000)  # Large limit to get all
        total_count = len(all_jobs)
        
        return JobListResponse(
            jobs=job_dicts,
            total_count=total_count,
            filtered_count=len(job_dicts)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list jobs: {str(e)}")


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str) -> JobStatusResponse:
    """
    Get detailed status of a specific job.
    
    Args:
        job_id: Unique job ID
        
    Returns:
        Detailed job status information
    """
    try:
        job_info = job_manager.get_job_status(job_id)
        
        if not job_info:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        return JobStatusResponse(
            job_id=job_info.job_id,
            status=job_info.status.value,
            progress=job_info.progress,
            created_at=job_info.created_at.isoformat(),
            started_at=job_info.started_at.isoformat() if job_info.started_at else None,
            completed_at=job_info.completed_at.isoformat() if job_info.completed_at else None,
            duration=job_info.duration,
            service_name=job_info.service_name,
            function_name=job_info.function_name,
            job_name=job_info.job_name,
            pipeline_job_id=job_info.pipeline_job_id,
            result=job_info.result,
            error=job_info.error,
            metadata=job_info.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")


@router.get("/{job_id}/result", response_model=Dict[str, Any])
async def get_job_result(job_id: str) -> Dict[str, Any]:
    """
    Get the result of a completed job.
    
    Args:
        job_id: Unique job ID
        
    Returns:
        Job result or error information
    """
    try:
        job_info = job_manager.get_job_status(job_id)
        
        if not job_info:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        if job_info.status == JobStatus.COMPLETED:
            return {
                "status": "completed",
                "job_id": job_id,
                "result": job_info.result
            }
        elif job_info.status == JobStatus.FAILED:
            return {
                "status": "failed",
                "job_id": job_id,
                "error": job_info.error
            }
        elif job_info.status == JobStatus.RUNNING:
            return {
                "status": "running",
                "job_id": job_id,
                "progress": job_info.progress,
                "message": "Job is still running"
            }
        else:
            return {
                "status": job_info.status.value,
                "job_id": job_id,
                "message": f"Job is in {job_info.status.value} state"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job result: {str(e)}")


@router.put("/{job_id}/progress", response_model=Dict[str, Any])
async def update_job_progress(job_id: str, request: JobProgressUpdateRequest) -> Dict[str, Any]:
    """
    Update job progress.
    
    Args:
        job_id: Unique job ID
        request: Progress update request
        
    Returns:
        Updated job status
    """
    try:
        job_info = job_manager.get_job_status(job_id)
        
        if not job_info:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        if job_info.status != JobStatus.RUNNING:
            raise HTTPException(
                status_code=400, 
                detail=f"Cannot update progress for job in {job_info.status.value} state"
            )
        
        # Validate progress value
        if not (0 <= request.progress <= 100):
            raise HTTPException(
                status_code=400, 
                detail="Progress must be between 0 and 100"
            )
        
        # Update progress
        job_manager.update_job_progress(
            job_id=job_id,
            progress=request.progress,
            metadata=request.metadata
        )
        
        # Get updated job info
        updated_job = job_manager.get_job_status(job_id)
        
        return {
            "status": "success",
            "job_id": job_id,
            "progress": updated_job.progress,
            "updated_at": updated_job.updated_at.isoformat() if updated_job.updated_at else None,
            "metadata": updated_job.metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update job progress: {str(e)}")


@router.post("/{job_id}/cancel", response_model=Dict[str, Any])
async def cancel_job(job_id: str, request: JobCancelRequest = None) -> Dict[str, Any]:
    """
    Cancel a job.
    
    Args:
        job_id: Unique job ID
        request: Optional cancellation request with reason
        
    Returns:
        Cancellation result
    """
    try:
        job_info = job_manager.get_job_status(job_id)
        
        if not job_info:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        if job_info.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            return {
                "status": "already_finished",
                "job_id": job_id,
                "current_status": job_info.status.value,
                "message": f"Job is already in {job_info.status.value} state"
            }
        
        # Attempt to cancel the job
        cancelled = job_manager.cancel_job(job_id)
        
        if cancelled:
            # Update metadata with cancellation reason
            if request and request.reason:
                job_manager.update_job_progress(
                    job_id=job_id,
                    progress=job_info.progress,
                    metadata={"cancellation_reason": request.reason}
                )
            
            return {
                "status": "cancelled",
                "job_id": job_id,
                "message": "Job cancelled successfully",
                "reason": request.reason if request else None
            }
        else:
            return {
                "status": "failed_to_cancel",
                "job_id": job_id,
                "message": "Failed to cancel job (may have already started execution)"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {str(e)}")


@router.get("/statistics/overview", response_model=JobStatisticsResponse)
async def get_job_statistics() -> JobStatisticsResponse:
    """
    Get overall job management statistics.
    
    Returns:
        Job statistics including counts, performance metrics, etc.
    """
    try:
        stats = job_manager.get_statistics()
        
        return JobStatisticsResponse(
            total_jobs=stats["total_jobs"],
            running_jobs=stats["running_jobs"],
            max_workers=stats["max_workers"],
            status_counts=stats["status_counts"],
            service_counts=stats["service_counts"],
            average_duration=stats["average_duration"],
            completed_jobs=stats["completed_jobs"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job statistics: {str(e)}")


@router.get("/services/{service_name}", response_model=JobListResponse)
async def list_jobs_by_service(
    service_name: str,
    status: Optional[str] = Query(None, description="Filter by job status"),
    limit: int = Query(100, description="Maximum number of jobs to return", ge=1, le=1000)
) -> JobListResponse:
    """
    List jobs for a specific service.
    
    Args:
        service_name: Name of the service
        status: Optional status filter
        limit: Maximum number of jobs to return
        
    Returns:
        List of jobs for the specified service
    """
    try:
        # Convert string status to JobStatus enum
        job_status = None
        if status:
            try:
                job_status = JobStatus(status)
            except ValueError:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid status: {status}. Valid values: {[s.value for s in JobStatus]}"
                )
        
        # Get jobs from manager
        jobs = job_manager.list_jobs(
            service_name=service_name,
            status=job_status,
            limit=limit
        )
        
        # Convert to dictionaries
        job_dicts = [job.to_dict() for job in jobs]
        
        # Get total count for this service (without status filter)
        all_service_jobs = job_manager.list_jobs(service_name=service_name, limit=10000)
        total_count = len(all_service_jobs)
        
        return JobListResponse(
            jobs=job_dicts,
            total_count=total_count,
            filtered_count=len(job_dicts)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list jobs for service {service_name}: {str(e)}")


@router.get("/pipeline/{pipeline_job_id}", response_model=JobListResponse)
async def list_jobs_by_pipeline(
    pipeline_job_id: str,
    status: Optional[str] = Query(None, description="Filter by job status"),
    limit: int = Query(100, description="Maximum number of jobs to return", ge=1, le=1000)
) -> JobListResponse:
    """
    List jobs for a specific pipeline.
    
    Args:
        pipeline_job_id: Pipeline job ID
        status: Optional status filter
        limit: Maximum number of jobs to return
        
    Returns:
        List of jobs for the specified pipeline
    """
    try:
        # Convert string status to JobStatus enum
        job_status = None
        if status:
            try:
                job_status = JobStatus(status)
            except ValueError:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid status: {status}. Valid values: {[s.value for s in JobStatus]}"
                )
        
        # Get jobs from manager
        jobs = job_manager.list_jobs(
            pipeline_job_id=pipeline_job_id,
            status=job_status,
            limit=limit
        )
        
        # Convert to dictionaries
        job_dicts = [job.to_dict() for job in jobs]
        
        # Get total count for this pipeline (without status filter)
        all_pipeline_jobs = job_manager.list_jobs(pipeline_job_id=pipeline_job_id, limit=10000)
        total_count = len(all_pipeline_jobs)
        
        return JobListResponse(
            jobs=job_dicts,
            total_count=total_count,
            filtered_count=len(job_dicts)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list jobs for pipeline {pipeline_job_id}: {str(e)}")


@router.delete("/cleanup", response_model=Dict[str, Any])
async def cleanup_old_jobs() -> Dict[str, Any]:
    """
    Manually trigger cleanup of old completed jobs.
    
    Returns:
        Cleanup result
    """
    try:
        # Get current stats
        stats_before = job_manager.get_statistics()
        
        # Trigger cleanup
        job_manager._cleanup_old_jobs()
        
        # Get updated stats
        stats_after = job_manager.get_statistics()
        
        cleaned_jobs = stats_before["total_jobs"] - stats_after["total_jobs"]
        
        return {
            "status": "success",
            "message": f"Cleanup completed. {cleaned_jobs} old jobs removed.",
            "jobs_before": stats_before["total_jobs"],
            "jobs_after": stats_after["total_jobs"],
            "cleaned_jobs": cleaned_jobs
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cleanup old jobs: {str(e)}")


@router.get("/health/check", response_model=Dict[str, Any])
async def job_manager_health_check() -> Dict[str, Any]:
    """
    Health check for the job management system.
    
    Returns:
        Health status and system information
    """
    try:
        stats = job_manager.get_statistics()
        
        # Determine health status
        health_status = "healthy"
        issues = []
        
        # Check for potential issues
        if stats["running_jobs"] >= stats["max_workers"]:
            health_status = "degraded"
            issues.append("All worker threads are busy")
        
        if stats["status_counts"].get("failed", 0) > stats["status_counts"].get("completed", 0):
            health_status = "warning"
            issues.append("More failed jobs than completed jobs")
        
        return {
            "status": health_status,
            "timestamp": datetime.now().isoformat(),
            "issues": issues,
            "statistics": stats,
            "system_info": {
                "max_workers": stats["max_workers"],
                "running_jobs": stats["running_jobs"],
                "total_jobs": stats["total_jobs"]
            }
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "issues": ["Job manager system error"]
        } 