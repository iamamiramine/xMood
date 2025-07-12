"""
Job Management Service Adapter

This adapter implements the IJobManagementService interface for the existing
job management service, enabling dependency injection and reducing cross-service
dependencies.
"""

from typing import Dict, Any, Optional
import logging

from domain.interfaces.service_interfaces import IJobManagementService
from application.shared.services.job_management_service import (
    job_manager,
    JobPriority,
    JobStatus,
)

logger = logging.getLogger(__name__)


class JobManagementServiceAdapter(IJobManagementService):
    """
    Adapter for the job management service that implements the IJobManagementService interface.
    
    This adapter wraps the existing job management service to provide
    a clean interface for dependency injection.
    """
    
    def __init__(self):
        self._job_manager = job_manager
        logger.info("JobManagementServiceAdapter initialized")
    
    def submit_job(self, service_name: str, function_name: str, job_function: callable, 
                   args: tuple = (), kwargs: dict = None, job_name: Optional[str] = None,
                   priority: JobPriority = JobPriority.NORMAL, pipeline_job_id: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Submit a background job.
        
        Args:
            service_name: Name of the service
            function_name: Name of the function
            job_function: Function to execute
            args: Function arguments
            kwargs: Function keyword arguments
            job_name: Optional job name
            priority: Job priority
            pipeline_job_id: Optional pipeline job ID
            metadata: Optional job metadata
            
        Returns:
            Job ID
        """
        try:
            return self._job_manager.submit_job(
                service_name=service_name,
                function_name=function_name,
                job_function=job_function,
                args=args,
                kwargs=kwargs or {},
                job_name=job_name,
                priority=priority,
                pipeline_job_id=pipeline_job_id,
                metadata=metadata
            )
        except Exception as e:
            logger.error(f"Error submitting job {service_name}.{function_name}: {e}")
            raise
    
    def get_job_status(self, job_id: str) -> JobStatus:
        """
        Get job status.
        
        Args:
            job_id: Job ID
            
        Returns:
            Job status
        """
        try:
            job_info = self._job_manager.get_job_info(job_id)
            return job_info.status if job_info else JobStatus.FAILED
        except Exception as e:
            logger.error(f"Error getting job status for {job_id}: {e}")
            return JobStatus.FAILED
    
    def get_job_result(self, job_id: str) -> Dict[str, Any]:
        """
        Get job result.
        
        Args:
            job_id: Job ID
            
        Returns:
            Job result
        """
        try:
            job_info = self._job_manager.get_job_info(job_id)
            if job_info and job_info.result:
                return job_info.result
            return {"status": "error", "message": "Job not found or no result available"}
        except Exception as e:
            logger.error(f"Error getting job result for {job_id}: {e}")
            return {"status": "error", "message": str(e)}
    
    def cancel_job(self, job_id: str, reason: Optional[str] = None) -> bool:
        """
        Cancel a job.
        
        Args:
            job_id: Job ID
            reason: Optional cancellation reason
            
        Returns:
            True if job was cancelled, False otherwise
        """
        try:
            return self._job_manager.cancel_job(job_id, reason)
        except Exception as e:
            logger.error(f"Error cancelling job {job_id}: {e}")
            return False 