"""
Job Manager for Pipeline Background Jobs

This module provides a singleton job manager that handles the execution and tracking
of background jobs for pipeline services. It manages job states, handles async 
execution, and provides job monitoring capabilities.
"""

import asyncio
import threading
from typing import Dict, Optional, Any, Callable
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import logging

from domain.models.pipeline.pipeline_model import PipelineJobInfo, JobStatus


class JobManager:
    """Singleton job manager for handling background jobs"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._jobs: Dict[str, PipelineJobInfo] = {}
            self._executor = ThreadPoolExecutor(max_workers=4)
            self._running_tasks: Dict[str, asyncio.Task] = {}
            self._logger = logging.getLogger(__name__)
            self._initialized = True
    
    def create_job(
        self,
        service_name: str,
        function_name: str,
        job_name: Optional[str] = None
    ) -> PipelineJobInfo:
        """Create a new job entry"""
        
        if job_name is None:
            job_name = f"{service_name}_{function_name}"
        
        job_info = PipelineJobInfo(
            job_name=job_name,
            service_name=service_name,
            function_name=function_name
        )
        
        self._jobs[job_info.job_id] = job_info
        self._logger.info(f"Created job {job_info.job_id}: {job_name}")
        
        return job_info
    
    def get_job(self, job_id: str) -> Optional[PipelineJobInfo]:
        """Get job information by ID"""
        return self._jobs.get(job_id)
    
    def get_all_jobs(self) -> Dict[str, PipelineJobInfo]:
        """Get all jobs"""
        return self._jobs.copy()
    
    def get_jobs_by_status(self, status: JobStatus) -> Dict[str, PipelineJobInfo]:
        """Get jobs by status"""
        return {
            job_id: job_info 
            for job_id, job_info in self._jobs.items() 
            if job_info.status == status
        }
    
    def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        error_message: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        progress: Optional[float] = None
    ) -> bool:
        """Update job status and related information"""
        
        if job_id not in self._jobs:
            return False
        
        job_info = self._jobs[job_id]
        job_info.status = status
        
        if error_message:
            job_info.error_message = error_message
        
        if result:
            job_info.result = result
        
        if progress is not None:
            job_info.progress = progress
        
        # Update timestamps
        if status == JobStatus.RUNNING and job_info.started_at is None:
            job_info.started_at = datetime.utcnow()
        elif status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            job_info.completed_at = datetime.utcnow()
        
        self._logger.info(f"Updated job {job_id} status to {status}")
        return True
    
    async def submit_job(
        self,
        job_id: str,
        service_function: Callable,
        parameters: Any
    ) -> bool:
        """Submit a job for background execution"""
        
        if job_id not in self._jobs:
            self._logger.error(f"Job {job_id} not found")
            return False
        
        if job_id in self._running_tasks:
            self._logger.warning(f"Job {job_id} is already running")
            return False
        
        # Create and start the async task
        task = asyncio.create_task(
            self._execute_job(job_id, service_function, parameters)
        )
        self._running_tasks[job_id] = task
        
        self._logger.info(f"Submitted job {job_id} for execution")
        return True
    
    async def _execute_job(
        self,
        job_id: str,
        service_function: Callable,
        parameters: Any
    ) -> None:
        """Execute a job in the background"""
        
        try:
            # Update status to running
            self.update_job_status(job_id, JobStatus.RUNNING)
            
            # Execute the service function
            if asyncio.iscoroutinefunction(service_function):
                # If it's an async function, await it
                result = await service_function(parameters)
            else:
                # If it's a sync function, run it in the executor
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self._executor, service_function, parameters
                )
            
            # Update status to completed
            self.update_job_status(
                job_id, 
                JobStatus.COMPLETED, 
                result=result,
                progress=100.0
            )
            
        except Exception as e:
            self._logger.error(f"Job {job_id} failed: {str(e)}")
            self.update_job_status(
                job_id, 
                JobStatus.FAILED, 
                error_message=str(e)
            )
        
        finally:
            # Clean up the task reference
            if job_id in self._running_tasks:
                del self._running_tasks[job_id]
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job"""
        
        if job_id not in self._jobs:
            return False
        
        if job_id in self._running_tasks:
            task = self._running_tasks[job_id]
            task.cancel()
            del self._running_tasks[job_id]
            
            self.update_job_status(job_id, JobStatus.CANCELLED)
            self._logger.info(f"Cancelled job {job_id}")
            return True
        
        # If job is not running, just update status
        if self._jobs[job_id].status == JobStatus.PENDING:
            self.update_job_status(job_id, JobStatus.CANCELLED)
            return True
        
        return False
    
    def cleanup_completed_jobs(self, max_age_hours: int = 24) -> int:
        """Clean up completed jobs older than max_age_hours"""
        
        cutoff_time = datetime.utcnow().replace(
            hour=datetime.utcnow().hour - max_age_hours
        )
        
        jobs_to_remove = []
        for job_id, job_info in self._jobs.items():
            if (job_info.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED] 
                and job_info.completed_at 
                and job_info.completed_at < cutoff_time):
                jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            del self._jobs[job_id]
        
        self._logger.info(f"Cleaned up {len(jobs_to_remove)} completed jobs")
        return len(jobs_to_remove)
    
    def get_running_jobs_count(self) -> int:
        """Get count of currently running jobs"""
        return len(self.get_jobs_by_status(JobStatus.RUNNING))
    
    def shutdown(self):
        """Shutdown the job manager"""
        
        # Cancel all running tasks
        for job_id, task in self._running_tasks.items():
            task.cancel()
            self.update_job_status(job_id, JobStatus.CANCELLED)
        
        # Shutdown the executor
        self._executor.shutdown(wait=True)
        
        self._logger.info("Job manager shutdown complete")


# Global job manager instance
job_manager = JobManager() 