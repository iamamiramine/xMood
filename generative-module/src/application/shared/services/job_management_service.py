import asyncio
import uuid
import json
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Union
from enum import Enum
from dataclasses import dataclass, asdict
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
import logging

from domain.exceptions.global_exceptions import ConfigurationError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Job status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(Enum):
    """Job priority enumeration."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class JobInfo:
    """Job information container."""
    job_id: str
    job_name: str
    service_name: str
    function_name: str
    status: JobStatus
    priority: JobPriority
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    progress: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    pipeline_job_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert JobInfo to dictionary."""
        data = asdict(self)
        # Convert datetime objects to ISO format
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
            elif isinstance(value, (JobStatus, JobPriority)):
                data[key] = value.value
        return data
    
    @property
    def duration(self) -> Optional[float]:
        """Calculate job duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        elif self.started_at:
            return (datetime.now() - self.started_at).total_seconds()
        return None


class JobManager:
    """
    Background job management system with monitoring capabilities.
    
    Features:
    - Background job execution with thread pool
    - Job status tracking and monitoring
    - Priority-based job queuing
    - Progress tracking and result storage
    - Error handling and recovery
    - Job cancellation and cleanup
    """
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.jobs: Dict[str, JobInfo] = {}
        self.running_jobs: Dict[str, Future] = {}
        self.job_lock = threading.Lock()
        self.cleanup_interval = 3600  # 1 hour cleanup interval
        self.max_completed_jobs = 100  # Keep max 100 completed jobs
        
        # Start background cleanup task
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start background cleanup task."""
        def cleanup_task():
            while True:
                try:
                    self._cleanup_old_jobs()
                    time.sleep(self.cleanup_interval)
                except Exception as e:
                    logger.error(f"Error in cleanup task: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
        cleanup_thread.start()
    
    def _cleanup_old_jobs(self):
        """Clean up old completed jobs."""
        with self.job_lock:
            completed_jobs = [
                job for job in self.jobs.values()
                if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]
            ]
            
            if len(completed_jobs) > self.max_completed_jobs:
                # Sort by completion time and remove oldest
                completed_jobs.sort(key=lambda x: x.completed_at or datetime.min)
                jobs_to_remove = completed_jobs[:-self.max_completed_jobs]
                
                for job in jobs_to_remove:
                    if job.job_id in self.jobs:
                        del self.jobs[job.job_id]
                        logger.info(f"Cleaned up old job: {job.job_id}")
    
    def submit_job(
        self,
        service_name: str,
        function_name: str,
        job_function: Callable,
        args: tuple = (),
        kwargs: dict = None,
        job_name: Optional[str] = None,
        priority: JobPriority = JobPriority.NORMAL,
        pipeline_job_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Submit a job for background execution.
        
        Args:
            service_name: Name of the service
            function_name: Name of the function being executed
            job_function: Function to execute
            args: Function arguments
            kwargs: Function keyword arguments
            job_name: Optional custom job name
            priority: Job priority
            pipeline_job_id: Optional pipeline job ID
            metadata: Optional job metadata
            
        Returns:
            Unique job ID
        """
        if kwargs is None:
            kwargs = {}
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Generate job name if not provided
        if not job_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            job_name = f"{service_name}_{function_name}_{timestamp}"
        
        # Create job info
        job_info = JobInfo(
            job_id=job_id,
            job_name=job_name,
            service_name=service_name,
            function_name=function_name,
            status=JobStatus.PENDING,
            priority=priority,
            created_at=datetime.now(),
            pipeline_job_id=pipeline_job_id,
            metadata=metadata or {}
        )
        
        # Store job info
        with self.job_lock:
            self.jobs[job_id] = job_info
        
        # Submit job to executor
        future = self.executor.submit(self._execute_job, job_id, job_function, args, kwargs)
        
        # Store running job reference
        with self.job_lock:
            self.running_jobs[job_id] = future
        
        logger.info(f"Job submitted: {job_id} ({service_name}.{function_name})")
        return job_id
    
    def _execute_job(self, job_id: str, job_function: Callable, args: tuple, kwargs: dict) -> Dict[str, Any]:
        """
        Execute a job with error handling and progress tracking.
        
        Args:
            job_id: Unique job ID
            job_function: Function to execute
            args: Function arguments
            kwargs: Function keyword arguments
            
        Returns:
            Job execution result
        """
        try:
            # Update job status to running
            self._update_job_status(job_id, JobStatus.RUNNING, started_at=datetime.now())
            
            # Execute the job function
            result = job_function(*args, **kwargs)
            
            # Update job status to completed
            self._update_job_status(
                job_id, 
                JobStatus.COMPLETED, 
                completed_at=datetime.now(), 
                result=result,
                progress=100.0
            )
            
            logger.info(f"Job completed successfully: {job_id}")
            return result
            
        except Exception as e:
            error_msg = f"Job failed: {str(e)}\n{traceback.format_exc()}"
            
            # Update job status to failed
            self._update_job_status(
                job_id, 
                JobStatus.FAILED, 
                completed_at=datetime.now(),
                error=error_msg
            )
            
            logger.error(f"Job failed: {job_id} - {error_msg}")
            return {"error": error_msg}
        
        finally:
            # Clean up running job reference
            with self.job_lock:
                if job_id in self.running_jobs:
                    del self.running_jobs[job_id]
    
    def _update_job_status(
        self, 
        job_id: str, 
        status: JobStatus, 
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        progress: Optional[float] = None
    ):
        """Update job status and metadata."""
        with self.job_lock:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                job.status = status
                job.updated_at = datetime.now()
                
                if started_at:
                    job.started_at = started_at
                if completed_at:
                    job.completed_at = completed_at
                if result is not None:
                    job.result = result
                if error:
                    job.error = error
                if progress is not None:
                    job.progress = progress
    
    def get_job_status(self, job_id: str) -> Optional[JobInfo]:
        """
        Get job status and information.
        
        Args:
            job_id: Unique job ID
            
        Returns:
            Job information or None if not found
        """
        with self.job_lock:
            return self.jobs.get(job_id)
    
    def list_jobs(
        self, 
        service_name: Optional[str] = None,
        status: Optional[JobStatus] = None,
        pipeline_job_id: Optional[str] = None,
        limit: int = 100
    ) -> List[JobInfo]:
        """
        List jobs with optional filtering.
        
        Args:
            service_name: Filter by service name
            status: Filter by job status
            pipeline_job_id: Filter by pipeline job ID
            limit: Maximum number of jobs to return
            
        Returns:
            List of job information
        """
        with self.job_lock:
            jobs = list(self.jobs.values())
        
        # Apply filters
        if service_name:
            jobs = [job for job in jobs if job.service_name == service_name]
        
        if status:
            jobs = [job for job in jobs if job.status == status]
        
        if pipeline_job_id:
            jobs = [job for job in jobs if job.pipeline_job_id == pipeline_job_id]
        
        # Sort by creation time (newest first)
        jobs.sort(key=lambda x: x.created_at, reverse=True)
        
        # Apply limit
        return jobs[:limit]
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a job.
        
        Args:
            job_id: Unique job ID
            
        Returns:
            True if job was cancelled, False otherwise
        """
        with self.job_lock:
            if job_id not in self.jobs:
                return False
            
            job = self.jobs[job_id]
            
            # Can only cancel pending or running jobs
            if job.status not in [JobStatus.PENDING, JobStatus.RUNNING]:
                return False
            
            # Try to cancel future if it's running
            if job_id in self.running_jobs:
                future = self.running_jobs[job_id]
                cancelled = future.cancel()
                if cancelled:
                    del self.running_jobs[job_id]
            else:
                cancelled = True
            
            if cancelled:
                job.status = JobStatus.CANCELLED
                job.completed_at = datetime.now()
                job.updated_at = datetime.now()
                logger.info(f"Job cancelled: {job_id}")
            
            return cancelled
    
    def get_job_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get job result.
        
        Args:
            job_id: Unique job ID
            
        Returns:
            Job result or None if not found/completed
        """
        job = self.get_job_status(job_id)
        if job and job.status == JobStatus.COMPLETED:
            return job.result
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get job management statistics.
        
        Returns:
            Statistics dictionary
        """
        with self.job_lock:
            jobs = list(self.jobs.values())
        
        total_jobs = len(jobs)
        status_counts = {}
        service_counts = {}
        
        for job in jobs:
            # Count by status
            status = job.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
            
            # Count by service
            service = job.service_name
            service_counts[service] = service_counts.get(service, 0) + 1
        
        # Calculate average duration for completed jobs
        completed_jobs = [job for job in jobs if job.status == JobStatus.COMPLETED and job.duration]
        avg_duration = sum(job.duration for job in completed_jobs) / len(completed_jobs) if completed_jobs else 0
        
        return {
            "total_jobs": total_jobs,
            "running_jobs": len(self.running_jobs),
            "max_workers": self.max_workers,
            "status_counts": status_counts,
            "service_counts": service_counts,
            "average_duration": avg_duration,
            "completed_jobs": len(completed_jobs)
        }
    
    def update_job_progress(self, job_id: str, progress: float, metadata: Optional[Dict[str, Any]] = None):
        """
        Update job progress.
        
        Args:
            job_id: Unique job ID
            progress: Progress percentage (0-100)
            metadata: Optional metadata to update
        """
        with self.job_lock:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                job.progress = min(100.0, max(0.0, progress))
                job.updated_at = datetime.now()
                
                if metadata:
                    if job.metadata is None:
                        job.metadata = {}
                    job.metadata.update(metadata)
    
    def shutdown(self):
        """Shutdown the job manager."""
        logger.info("Shutting down job manager...")
        
        # Cancel all running jobs
        with self.job_lock:
            for job_id in list(self.running_jobs.keys()):
                self.cancel_job(job_id)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        logger.info("Job manager shutdown complete")


# Global job manager instance
job_manager = JobManager(max_workers=4)


def submit_background_job(
    service_name: str,
    function_name: str,
    job_function: Callable,
    args: tuple = (),
    kwargs: dict = None,
    job_name: Optional[str] = None,
    priority: JobPriority = JobPriority.NORMAL,
    pipeline_job_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Convenience function to submit a background job.
    
    Args:
        service_name: Name of the service
        function_name: Name of the function being executed
        job_function: Function to execute
        args: Function arguments
        kwargs: Function keyword arguments
        job_name: Optional custom job name
        priority: Job priority
        pipeline_job_id: Optional pipeline job ID
        metadata: Optional job metadata
        
    Returns:
        Unique job ID
    """
    return job_manager.submit_job(
        service_name=service_name,
        function_name=function_name,
        job_function=job_function,
        args=args,
        kwargs=kwargs,
        job_name=job_name,
        priority=priority,
        pipeline_job_id=pipeline_job_id,
        metadata=metadata
    ) 