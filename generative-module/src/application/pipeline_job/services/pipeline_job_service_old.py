import asyncio
import traceback
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import logging

from application.pipeline_config.services.pipeline_config_service import pipeline_config_service
from application.shared.services.job_management_service import (
    submit_background_job,
    job_manager,
    JobPriority,
    JobStatus
)
from domain.exceptions.global_exceptions import ConfigurationException

# Import all service functions and their parameter models
from application.encoder.services.encoder_service import (
    encode_dataset,
    tokenize_remi_dataset
)
from application.feature_extraction.services.feature_extraction_service import (
    extract_symbolic_features,
    train_vae,
    generate_latent_representations_dataset
)
from application.generator.services.generator_service import (
    train_generator,
    generate_from_midi
)
from application.multimodal_mapping.services.multimodal_mapping_service import (
    train_multimodal_mapping,
    generate_multimodal_representations
)
from application.music_base.services.music_base_service import (
    extract_chords,
    synthesize_midi
)
from application.dataloader.services.dataloader_service import (
    initialize_dataloader_module,
    load_dataset,
    process_dataset_files
)

# Import parameter models from domain layer
from domain.models.encoder.encoder_model import EncodeParameters, EncodeDatasetParameters, TokenizeRemiDatasetParameters
from domain.models.feature_extraction.feature_extraction_model import (
    SymbolicFeaturesParameters,
    SymbolicFeaturesDatasetParameters,
    VaeTrainingParameters,
    LatentRepresentationParameters,
)
from domain.models.generator_model import (
    GenerateFromMIDIParameters,
    GeneratorTrainingParameters,
)
from domain.models.multimodal_mapping_model import (
    MultimodalMappingParameters,
    MultimodalTrainingParameters,
)
from domain.models.music_base.music_base_model import (
    MusicBaseParameters,
)
from domain.models.dataloader.dataloader_model import (
    DataloaderModuleParameters,
    DatasetLoadParameters,
    DataloaderParameters
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipelineJobService:
    """
    Pipeline Job Service for orchestrating pipeline execution.
    
    Features:
    - Parse pipeline configurations and convert to BaseModel parameters
    - Execute individual services or complete pipelines
    - Run services in parallel using background job management
    - Monitor pipeline progress and handle errors
    - Support for different pipeline types and configurations
    """
    
    def __init__(self):
        self.service_registry = self._build_service_registry()
        self.parameter_registry = self._build_parameter_registry()
    
    def _build_service_registry(self) -> Dict[str, Dict[str, Any]]:
        """Build registry of available services and their functions."""
        return {
            "encoder": {
                "encode_dataset": encode_dataset,
                "tokenize_remi_dataset": tokenize_remi_dataset,
            },
            "feature_extraction": {
                "extract_symbolic_features": extract_symbolic_features,
                "train_vae": train_vae,
                "generate_latent_representations": generate_latent_representations_dataset,
            },
            "generator": {
                "train": train_generator,
                "generate_from_midi": generate_from_midi,
            },
            "multimodal_mapping": {
                "train": train_multimodal_mapping,
                "generate_representations": generate_multimodal_representations,
            },
            "music_base": {
                "extract_chords": extract_chords,
                "synthesize_midi": synthesize_midi,
            },
            "dataloader": {
                "initialize_module": initialize_dataloader_module,
                "load_dataset": load_dataset,
                "process_files": process_dataset_files,
            },
        }
    
    def _build_parameter_registry(self) -> Dict[str, Dict[str, Any]]:
        """Build registry of parameter models for each service function."""
        return {
            "encoder": {
                "encode_midi": EncodeParameters,
                "encode_dataset": EncodeDatasetParameters,
                "tokenize_remi_dataset": TokenizeRemiDatasetParameters,
            },
            "feature_extraction": {
                "extract_symbolic_features": SymbolicFeaturesParameters,
                "extract_symbolic_features_dataset": SymbolicFeaturesDatasetParameters,
                "train_vae": VaeTrainingParameters,
                "generate_latent_representations": LatentRepresentationParameters,
            },
            "generator": {
                "train": GeneratorTrainingParameters,
                "generate_from_midi": GenerateFromMIDIParameters,
            },
            "multimodal_mapping": {
                "train": MultimodalTrainingParameters,
                "generate_representations": MultimodalMappingParameters,
            },
            "music_base": {
                "extract_chords": MusicBaseParameters,
                "synthesize_midi": MusicBaseParameters,
            },
            "dataloader": {
                "initialize_module": DataloaderModuleParameters,
                "load_dataset": DatasetLoadParameters,
                "process_files": DataloaderParameters,
            },
        }
    
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
            BaseModel parameters instance
        """
        try:
            # Get parameter model class
            if service_name not in self.parameter_registry:
                raise ConfigurationException(f"Service '{service_name}' not found in parameter registry")
            
            service_params = self.parameter_registry[service_name]
            if function_name not in service_params:
                raise ConfigurationException(f"Function '{function_name}' not found for service '{service_name}'")
            
            parameter_model = service_params[function_name]
            
            # Extract service-specific config
            service_config = config.get(service_name, {})
            global_config = config.get('global', {})
            data_config = config.get('data', {})
            
            # Merge configurations (service-specific overrides global)
            merged_config = {**global_config, **data_config, **service_config}
            
            # Create parameter instance
            parameters = parameter_model(**merged_config)
            
            logger.info(f"Parsed configuration for {service_name}.{function_name}")
            return parameters
            
        except Exception as e:
            logger.error(f"Failed to parse config for {service_name}.{function_name}: {str(e)}")
            raise ConfigurationException(f"Configuration parsing failed: {str(e)}")
    
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
            # Get service function
            if service_name not in self.service_registry:
                raise ConfigurationException(f"Service '{service_name}' not found")
            
            service_functions = self.service_registry[service_name]
            if function_name not in service_functions:
                raise ConfigurationException(f"Function '{function_name}' not found for service '{service_name}'")
            
            service_function = service_functions[function_name]
            
            # Submit background job
            job_id = submit_background_job(
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
            # Get pipeline configuration
            config = pipeline_config_service.get_pipeline_config(pipeline_job_id)
            
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
    
    def run_pipeline(
        self,
        pipeline_job_id: str,
        pipeline_definition: Optional[List[Dict[str, Any]]] = None,
        parallel_execution: bool = False,
        job_priority: JobPriority = JobPriority.NORMAL
    ) -> Dict[str, Any]:
        """
        Run a complete pipeline with multiple services.
        
        Args:
            pipeline_job_id: Pipeline job ID
            pipeline_definition: Custom pipeline definition (if not provided, runs all services)
            parallel_execution: Whether to run services in parallel
            job_priority: Job priority for all jobs
            
        Returns:
            Pipeline execution result with job IDs
        """
        try:
            # Update pipeline status
            pipeline_config_service.update_pipeline_status(pipeline_job_id, "running")
            
            # Get pipeline configuration
            config = pipeline_config_service.get_pipeline_config(pipeline_job_id)
            
            # Define default pipeline if not provided
            if not pipeline_definition:
                pipeline_definition = self._get_default_pipeline_definition(config)
            
            # Execute pipeline
            if parallel_execution:
                result = self._run_pipeline_parallel(
                    pipeline_job_id=pipeline_job_id,
                    pipeline_definition=pipeline_definition,
                    config=config,
                    job_priority=job_priority
                )
            else:
                result = self._run_pipeline_sequential(
                    pipeline_job_id=pipeline_job_id,
                    pipeline_definition=pipeline_definition,
                    config=config,
                    job_priority=job_priority
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            pipeline_config_service.update_pipeline_status(
                pipeline_job_id, 
                "failed",
                {"error": str(e)}
            )
            raise ConfigurationException(f"Pipeline execution failed: {str(e)}")
    
    def _get_default_pipeline_definition(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get default pipeline definition based on available services in config."""
        pipeline_steps = []
        
        # Define common pipeline order
        service_order = [
            ("dataloader", "initialize_module"),
            ("dataloader", "process_files"),
            ("encoder", "encode_dataset"),
            ("feature_extraction", "extract_symbolic_features"),
            ("music_base", "extract_chords"),
            ("feature_extraction", "train_vae"),
            ("feature_extraction", "generate_latent_representations"),
            ("multimodal_mapping", "train"),
            ("generator", "train"),
        ]
        
        # Add steps that have corresponding config sections
        for service_name, function_name in service_order:
            if service_name in config:
                pipeline_steps.append({
                    "service": service_name,
                    "function": function_name,
                    "dependencies": []  # Can be extended for complex dependencies
                })
        
        return pipeline_steps
    
    def _run_pipeline_parallel(
        self,
        pipeline_job_id: str,
        pipeline_definition: List[Dict[str, Any]],
        config: Dict[str, Any],
        job_priority: JobPriority
    ) -> Dict[str, Any]:
        """Run pipeline with parallel execution."""
        job_ids = []
        
        # Submit all jobs in parallel
        for step in pipeline_definition:
            try:
                job_id = self.run_service_from_pipeline_config(
                    pipeline_job_id=pipeline_job_id,
                    service_name=step["service"],
                    function_name=step["function"],
                    job_priority=job_priority,
                    metadata={
                        "pipeline_step": step,
                        "execution_mode": "parallel"
                    }
                )
                job_ids.append({
                    "service": step["service"],
                    "function": step["function"],
                    "job_id": job_id
                })
            except Exception as e:
                logger.error(f"Failed to submit job for {step['service']}.{step['function']}: {str(e)}")
        
        return {
            "pipeline_job_id": pipeline_job_id,
            "execution_mode": "parallel",
            "submitted_jobs": job_ids,
            "total_jobs": len(job_ids),
            "status": "running"
        }
    
    def _run_pipeline_sequential(
        self,
        pipeline_job_id: str,
        pipeline_definition: List[Dict[str, Any]],
        config: Dict[str, Any],
        job_priority: JobPriority
    ) -> Dict[str, Any]:
        """Run pipeline with sequential execution."""
        job_ids = []
        
        # Submit jobs sequentially (can be extended to wait for completion)
        for step in pipeline_definition:
            try:
                job_id = self.run_service_from_pipeline_config(
                    pipeline_job_id=pipeline_job_id,
                    service_name=step["service"],
                    function_name=step["function"],
                    job_priority=job_priority,
                    metadata={
                        "pipeline_step": step,
                        "execution_mode": "sequential"
                    }
                )
                job_ids.append({
                    "service": step["service"],
                    "function": step["function"],
                    "job_id": job_id
                })
            except Exception as e:
                logger.error(f"Failed to submit job for {step['service']}.{step['function']}: {str(e)}")
                break  # Stop on first failure in sequential mode
        
        return {
            "pipeline_job_id": pipeline_job_id,
            "execution_mode": "sequential",
            "submitted_jobs": job_ids,
            "total_jobs": len(job_ids),
            "status": "running"
        }
    
    def get_pipeline_status(self, pipeline_job_id: str) -> Dict[str, Any]:
        """
        Get comprehensive pipeline status including all sub-jobs.
        
        Args:
            pipeline_job_id: Pipeline job ID
            
        Returns:
            Pipeline status with sub-job details
        """
        try:
            # Get pipeline configuration and status
            config = pipeline_config_service.get_pipeline_config(pipeline_job_id)
            pipeline_metadata = config.get('pipeline_metadata', {})
            
            # Get all jobs for this pipeline
            jobs = job_manager.list_jobs(pipeline_job_id=pipeline_job_id)
            
            # Calculate overall status
            total_jobs = len(jobs)
            completed_jobs = sum(1 for job in jobs if job.status == JobStatus.COMPLETED)
            failed_jobs = sum(1 for job in jobs if job.status == JobStatus.FAILED)
            running_jobs = sum(1 for job in jobs if job.status == JobStatus.RUNNING)
            
            # Determine overall pipeline status
            if total_jobs == 0:
                overall_status = "created"
            elif failed_jobs > 0:
                overall_status = "failed"
            elif completed_jobs == total_jobs:
                overall_status = "completed"
            elif running_jobs > 0:
                overall_status = "running"
            else:
                overall_status = "pending"
            
            # Calculate progress
            progress = (completed_jobs / total_jobs * 100) if total_jobs > 0 else 0
            
            return {
                "pipeline_job_id": pipeline_job_id,
                "pipeline_name": pipeline_metadata.get('job_name', 'Unknown'),
                "overall_status": overall_status,
                "progress": progress,
                "total_jobs": total_jobs,
                "completed_jobs": completed_jobs,
                "failed_jobs": failed_jobs,
                "running_jobs": running_jobs,
                "created_at": pipeline_metadata.get('created_at'),
                "updated_at": pipeline_metadata.get('updated_at'),
                "services": pipeline_metadata.get('services', []),
                "environment": pipeline_metadata.get('environment', 'unknown'),
                "jobs": [job.to_dict() for job in jobs]
            }
            
        except Exception as e:
            logger.error(f"Failed to get pipeline status: {str(e)}")
            raise ConfigurationException(f"Pipeline status retrieval failed: {str(e)}")
    
    def cancel_pipeline(self, pipeline_job_id: str) -> Dict[str, Any]:
        """
        Cancel a pipeline and all its sub-jobs.
        
        Args:
            pipeline_job_id: Pipeline job ID
            
        Returns:
            Cancellation result
        """
        try:
            # Get all jobs for this pipeline
            jobs = job_manager.list_jobs(pipeline_job_id=pipeline_job_id)
            
            # Cancel all jobs
            cancelled_jobs = []
            for job in jobs:
                if job.status in [JobStatus.PENDING, JobStatus.RUNNING]:
                    success = job_manager.cancel_job(job.job_id)
                    if success:
                        cancelled_jobs.append(job.job_id)
            
            # Update pipeline status
            pipeline_config_service.update_pipeline_status(
                pipeline_job_id, 
                "cancelled",
                {"cancelled_jobs": cancelled_jobs}
            )
            
            return {
                "pipeline_job_id": pipeline_job_id,
                "status": "cancelled",
                "cancelled_jobs": cancelled_jobs,
                "total_jobs": len(jobs),
                "message": f"Pipeline cancelled. {len(cancelled_jobs)} jobs were cancelled."
            }
            
        except Exception as e:
            logger.error(f"Failed to cancel pipeline: {str(e)}")
            raise ConfigurationException(f"Pipeline cancellation failed: {str(e)}")
    
    def get_available_services(self) -> Dict[str, List[str]]:
        """Get available services and their functions."""
        return {
            service: list(functions.keys())
            for service, functions in self.service_registry.items()
        }
    
    def validate_pipeline_definition(self, pipeline_definition: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate a pipeline definition.
        
        Args:
            pipeline_definition: Pipeline definition to validate
            
        Returns:
            Validation result
        """
        errors = []
        warnings = []
        
        for i, step in enumerate(pipeline_definition):
            step_errors = []
            
            # Check required fields
            if "service" not in step:
                step_errors.append("Missing 'service' field")
            if "function" not in step:
                step_errors.append("Missing 'function' field")
            
            # Check if service exists
            if "service" in step and step["service"] not in self.service_registry:
                step_errors.append(f"Unknown service: {step['service']}")
            
            # Check if function exists
            if ("service" in step and "function" in step and 
                step["service"] in self.service_registry and
                step["function"] not in self.service_registry[step["service"]]):
                step_errors.append(f"Unknown function: {step['function']} for service {step['service']}")
            
            if step_errors:
                errors.append(f"Step {i}: {', '.join(step_errors)}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "total_steps": len(pipeline_definition)
        }


# Global pipeline job service instance
pipeline_job_service = PipelineJobService() 