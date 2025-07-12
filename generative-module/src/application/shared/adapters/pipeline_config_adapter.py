"""
Pipeline Config Service Adapter

This adapter implements the IPipelineConfigService interface for the existing
pipeline config service, enabling dependency injection and reducing cross-service
dependencies.
"""

from typing import Dict, Any, Optional, List
import logging

from domain.interfaces.service_interfaces import IPipelineConfigService
from application.pipeline_config.services.pipeline_config_service import pipeline_config_service

logger = logging.getLogger(__name__)


class PipelineConfigServiceAdapter(IPipelineConfigService):
    """
    Adapter for the pipeline config service that implements the IPipelineConfigService interface.
    
    This adapter wraps the existing pipeline config service to provide
    a clean interface for dependency injection.
    """
    
    def __init__(self):
        self._pipeline_config_service = pipeline_config_service
        logger.info("PipelineConfigServiceAdapter initialized")
    
    def generate_pipeline_config(self, services: List[str], environment: str = "production",
                                 job_name: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate pipeline configuration.
        
        Args:
            services: List of services to include
            environment: Environment name
            job_name: Optional job name
            overrides: Optional configuration overrides
            
        Returns:
            Generated pipeline configuration
        """
        try:
            return self._pipeline_config_service.generate_pipeline_config(
                services=services,
                environment=environment,
                job_name=job_name,
                overrides=overrides
            )
        except Exception as e:
            logger.error(f"Error generating pipeline config: {e}")
            return {}
    
    def get_pipeline_config(self, pipeline_job_id: str) -> Dict[str, Any]:
        """
        Get pipeline configuration.
        
        Args:
            pipeline_job_id: Pipeline job ID
            
        Returns:
            Pipeline configuration
        """
        try:
            return self._pipeline_config_service.get_pipeline_config(pipeline_job_id)
        except Exception as e:
            logger.error(f"Error getting pipeline config for {pipeline_job_id}: {e}")
            return {}
    
    def update_pipeline_status(self, pipeline_job_id: str, status: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update pipeline status.
        
        Args:
            pipeline_job_id: Pipeline job ID
            status: New status
            metadata: Optional metadata
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            self._pipeline_config_service.update_pipeline_status(
                pipeline_job_id=pipeline_job_id,
                status=status,
                metadata=metadata
            )
            return True
        except Exception as e:
            logger.error(f"Error updating pipeline status for {pipeline_job_id}: {e}")
            return False
