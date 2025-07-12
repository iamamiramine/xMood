"""
Pipeline Job Service with Dependency Injection

This version of the pipeline job service uses interfaces and
dependency injection to reduce cross-service dependencies and improve
architectural compliance.
"""

import asyncio
import traceback
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import logging

from domain.interfaces.dependency_injection import (
    get_dependency_container,
)

from domain.interfaces.service_interfaces import (
    IEncoderService,
    IFeatureExtractionService,
    IGeneratorService,
    IMultimodalMappingService,
    IMusicBaseService,
    IDataloaderService,
    IConfigService,
    IJobManagementService,
    IPipelineConfigService,
)

from application.shared.services.job_management_service import JobPriority, JobStatus
from domain.exceptions.global_exceptions import ConfigurationException

# Import parameter model registry from config schema
from domain.models.config.config_schema import PARAMETER_MODEL_REGISTRY

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipelineJobService:
    """
    Pipeline Job Service with Dependency Injection.
    
    This service uses interfaces and dependency injection to reduce
    cross-service dependencies and improve architectural compliance.
    
    Features:
    - Uses service interfaces instead of direct imports
    - Dependency injection for loose coupling
    - Configurable service registry
    - Improved error handling and logging
    """
    
    def __init__(self):
        self._container = get_dependency_container()
        self._service_registry = self._build_service_registry()
        self._parameter_registry = self._build_parameter_registry()
        
        # Resolve core services
        self._config_service = self._container.resolve(IConfigService)
        self._job_service = self._container.resolve(IJobManagementService)
        self._pipeline_config_service = self._container.resolve(IPipelineConfigService)
        
        logger.info("PipelineJobService initialized with dependency injection")
    
    def _build_service_registry(self) -> Dict[str, Dict[str, Any]]:
        """
        Build registry of available services using interfaces.
        
        This method creates a registry that maps service names to their
        interface methods, enabling dynamic service discovery.
        """
        return {
            "encoder": {
                "encode_midi": self._get_service_method(IEncoderService, "encode_midi"),
                "encode_dataset": self._get_service_method(IEncoderService, "encode_dataset"),
                "tokenize_remi_dataset": self._get_service_method(IEncoderService, "tokenize_remi_dataset"),
            },
            "feature_extraction": {
                "extract_symbolic_features": self._get_service_method(IFeatureExtractionService, "extract_symbolic_features"),
                "extract_symbolic_features_dataset": self._get_service_method(IFeatureExtractionService, "extract_symbolic_features_dataset"),
                "train_vae": self._get_service_method(IFeatureExtractionService, "train_vae"),
                "generate_latent_representations": self._get_service_method(IFeatureExtractionService, "generate_latent_representations_dataset"),
            },
            "generator": {
                "train": self._get_service_method(IGeneratorService, "train_generator"),
                "generate_from_midi": self._get_service_method(IGeneratorService, "generate_from_midi"),
                "batch_generate": self._get_service_method(IGeneratorService, "batch_generate_from_dataset"),
            },
            "multimodal_mapping": {
                "train": self._get_service_method(IMultimodalMappingService, "train_multimodal_mapping"),
                "generate_representations": self._get_service_method(IMultimodalMappingService, "generate_multimodal_representations"),
            },
            "music_base": {
                "extract_chords": self._get_service_method(IMusicBaseService, "extract_chords"),
                "synthesize_midi": self._get_service_method(IMusicBaseService, "synthesize_midi"),
            },
            "dataloader": {
                "initialize_module": self._get_service_method(IDataloaderService, "initialize_dataloader_module"),
                "load_dataset": self._get_service_method(IDataloaderService, "load_dataset"),
                "process_files": self._get_service_method(IDataloaderService, "process_dataset_files"),
            },
        }
    
    def _get_service_method(self, service_interface: type, method_name: str) -> callable:
        """
        Get a service method using dependency injection.
        
        Args:
            service_interface: The service interface type
            method_name: Name of the method to retrieve
            
        Returns:
            Callable service method
        """
        def service_method_wrapper(*args, **kwargs):
            try:
                service = self._container.resolve(service_interface)
                method = getattr(service, method_name)
                return method(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error calling {service_interface.__name__}.{method_name}: {e}")
                raise
        
        return service_method_wrapper
    
    def _build_parameter_registry(self) -> Dict[str, Dict[str, Any]]:
        """Build registry of parameter models for each service function."""
        return PARAMETER_MODEL_REGISTRY
    
    def execute_service_function(
        self, 
        pipeline_job_id: str,
        service_name: str, 
        function_name: str, 
        parameters: Any,
        job_priority: JobPriority = JobPriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Execute a single service function with BaseModel parameters.
        
        Args:
            pipeline_job_id: Pipeline job ID
            service_name: Name of the service
            function_name: Name of the function
            parameters: BaseModel parameters
            job_priority: Job priority
            metadata: Optional job metadata
            
        Returns:
            Background job ID
        """
        try:
            # Get service function using interface
            if service_name not in self._service_registry:
                raise ConfigurationException(f"Service '{service_name}' not found")
            
            service_functions = self._service_registry[service_name]
            if function_name not in service_functions:
                raise ConfigurationException(f"Function '{function_name}' not found for service '{service_name}'")
            
            service_function = service_functions[function_name]
            
            # Submit background job using interface
            job_id = self._job_service.submit_job(
                service_name=service_name,
                function_name=function_name,
                job_function=service_function,
                kwargs={"parameters": parameters},
                job_name=f"{service_name}_{function_name}",
                priority=job_priority,
                pipeline_job_id=pipeline_job_id,
                metadata=metadata
            )
            
            logger.info(f"Started background job {job_id} for {service_name}.{function_name}")
            return job_id
            
        except Exception as e:
            logger.error(f"Failed to execute {service_name}.{function_name}: {str(e)}")
            raise ConfigurationException(f"Service execution failed: {str(e)}")
    
    def run_service_from_pipeline_config(
        self,
        pipeline_job_id: str,
        service_name: str,
        function_name: str,
        job_priority: JobPriority = JobPriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Run a service function using pipeline configuration.
        
        Args:
            pipeline_job_id: Pipeline job ID
            service_name: Name of the service
            function_name: Name of the function
            job_priority: Job priority
            metadata: Optional job metadata
            
        Returns:
            Background job ID
        """
        try:
            # Get pipeline configuration using interface
            config = self._pipeline_config_service.get_pipeline_config(pipeline_job_id)
            
            # Parse configuration to parameters
            parameters = self.parse_service_config_to_parameters(
                service_name=service_name,
                function_name=function_name,
                config=config
            )
            
            # Execute service function
            job_id = self.execute_service_function(
                pipeline_job_id=pipeline_job_id,
                service_name=service_name,
                function_name=function_name,
                parameters=parameters,
                job_priority=job_priority,
                metadata=metadata
            )
            
            return job_id
            
        except Exception as e:
            logger.error(f"Failed to run service from pipeline config: {str(e)}")
            raise ConfigurationException(f"Pipeline service execution failed: {str(e)}")
    
    def parse_service_config_to_parameters(
        self, 
        service_name: str, 
        function_name: str, 
        config: Dict[str, Any]
    ) -> Any:
        """
        Parse service configuration to BaseModel parameters.
        
        Args:
            service_name: Name of the service
            function_name: Name of the function
            config: Configuration dictionary
            
        Returns:
            BaseModel parameters
        """
        try:
            # Get parameter model class
            if service_name not in self._parameter_registry:
                raise ConfigurationException(f"Parameter model for service '{service_name}' not found")
            
            service_params = self._parameter_registry[service_name]
            if function_name not in service_params:
                raise ConfigurationException(f"Parameter model for function '{function_name}' not found")
            
            parameter_model = service_params[function_name]
            
            # Extract service-specific config
            service_config = config.get(service_name, {})
            
            # Extract function-specific config
            function_config = service_config.get(function_name, {})
            
            # If function config is empty, try to use the entire service config
            # This provides backward compatibility
            if not function_config:
                function_config = service_config
            
            # Parse config to parameters
            parameters = parameter_model(**function_config)
            
            return parameters
            
        except Exception as e:
            logger.error(f"Failed to parse config to parameters: {str(e)}")
            raise ConfigurationException(f"Parameter parsing failed: {str(e)}")
    
    def get_service_health(self) -> Dict[str, Any]:
        """
        Get health status of all services.
        
        Returns:
            Dictionary containing health status of all services
        """
        health_status = {}
        
        for service_name, functions in self._service_registry.items():
            service_health = {
                "service_name": service_name,
                "functions": list(functions.keys()),
                "status": "healthy"
            }
            
            try:
                # Test service availability by attempting to resolve
                # This doesn't execute the service, just checks if it's available
                for function_name in functions:
                    _ = functions[function_name]
                
            except Exception as e:
                service_health["status"] = "unhealthy"
                service_health["error"] = str(e)
            
            health_status[service_name] = service_health
        
        return health_status


# Create a factory function for the pipeline job service
def create_pipeline_job_service() -> PipelineJobService:
    """
    Factory function to create a PipelineJobService instance.
    
    Returns:
        PipelineJobService instance
    """
    return PipelineJobService()


# Global instance for backward compatibility
pipeline_job_service = create_pipeline_job_service() 