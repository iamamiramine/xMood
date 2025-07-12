import os
import yaml
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List

from domain.constants.paths_constants import CONFIG_PATH
from domain.exceptions.global_exceptions import ConfigurationException

# Import BaseModel configuration classes
from domain.models.config.config_schema import FullConfig

# Global registry for tracking pipeline jobs
_pipeline_jobs_registry = {}

class PipelineConfigService:
    """
    Enhanced Pipeline Configuration Service for generating unique configs for each pipeline job.
    Supports job tracking, unique ID generation, and pipeline-specific configurations.
    """
    
    def __init__(self):
        self.config_cache = {}
        self.pipeline_jobs_path = os.path.join(CONFIG_PATH, "pipeline_jobs")
        os.makedirs(self.pipeline_jobs_path, exist_ok=True)
    
    def generate_pipeline_config(
        self, 
        services: List[str],
        environment: str = 'production',
        job_name: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a unique pipeline configuration for a specific job.
        
        Args:
            services: List of services to include in the pipeline
            environment: Environment (development, staging, production)
            job_name: Optional custom job name
            overrides: Configuration overrides in the format:
                      {
                          'global': {...},
                          'service_name': {
                              'function_name': {...}
                          }
                      }
            
        Returns:
            Dictionary containing pipeline config with unique job ID
        """
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Generate job name if not provided
        if not job_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            job_name = f"pipeline_job_{timestamp}"
        
        # Create base configuration using BaseModel classes
        base_config = self._create_base_config()
        
        # Create pipeline job metadata
        pipeline_metadata = {
            'job_id': job_id,
            'job_name': job_name,
            'created_at': datetime.now().isoformat(),
            'environment': environment,
            'services': services,
            'status': 'created',
            'config_version': '1.0',
            'generated_by': 'PipelineConfigService.generate_pipeline_config'
        }
        
        # Filter services if specified
        filtered_config = {
            'pipeline_metadata': pipeline_metadata,
            'global': base_config.global_config.dict(),
            'paths': base_config.paths.dict(),
            'midi': base_config.midi.dict()
        }
        
        # Add only requested services
        for service in services:
            service_config = base_config.get_config_for_service(service)
            if service_config:
                filtered_config[service] = service_config.dict()
        
        # Apply overrides if provided
        if overrides:
            # Validate overrides before applying
            validation_result = self.validate_overrides(overrides, services)
            if not validation_result['valid']:
                raise ConfigurationException(f"Invalid overrides: {validation_result['errors']}")
            
            filtered_config = self._apply_overrides(filtered_config, overrides)
        
        # Save configuration to unique file
        config_file_path = os.path.join(self.pipeline_jobs_path, f"{job_id}.yaml")
        with open(config_file_path, 'w', encoding='utf-8') as f:
            yaml.dump(filtered_config, f, default_flow_style=False, sort_keys=False, indent=2)
        
        # Register pipeline job
        _pipeline_jobs_registry[job_id] = {
            'job_id': job_id,
            'job_name': job_name,
            'config_path': config_file_path,
            'services': services,
            'environment': environment,
            'status': 'created',
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        
        # Add config path to the returned config
        filtered_config['pipeline_metadata']['config_path'] = config_file_path
        
        return filtered_config
    
    def _apply_overrides(self, config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply configuration overrides to the base configuration.
        
        Args:
            config: Base configuration dictionary
            overrides: Override values to apply
            
        Returns:
            Configuration with overrides applied
        """
        def deep_merge(base_dict: Dict[str, Any], override_dict: Dict[str, Any]) -> Dict[str, Any]:
            """Recursively merge override values into base dictionary."""
            result = base_dict.copy()
            
            for key, value in override_dict.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    # Recursively merge nested dictionaries
                    result[key] = deep_merge(result[key], value)
                else:
                    # Override the value
                    result[key] = value
            
            return result
        
        return deep_merge(config, overrides)
    
    def validate_overrides(self, overrides: Dict[str, Any], services: List[str]) -> Dict[str, Any]:
        """
        Validate that overrides are properly formatted and contain valid service names.
        
        Args:
            overrides: Override configuration to validate
            services: List of services being configured
            
        Returns:
            Validation result with status and any errors
        """
        valid_sections = ['global', 'paths', 'midi', 'pipeline_metadata'] + services
        errors = []
        
        for section in overrides.keys():
            if section not in valid_sections:
                errors.append(f"Invalid section '{section}'. Valid sections are: {valid_sections}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    def _create_base_config(self) -> FullConfig:
        """
        Create a base configuration using BaseModel classes.
        
        Returns:
            FullConfig instance with all service configurations
        """
        return FullConfig()

    
    def get_pipeline_config(self, job_id: str) -> Dict[str, Any]:
        """
        Get pipeline configuration by job ID.
        
        Args:
            job_id: Unique job ID
            
        Returns:
            Pipeline configuration dictionary
        """
        if job_id not in _pipeline_jobs_registry:
            raise ConfigurationException(f"Pipeline job {job_id} not found")
        
        job_info = _pipeline_jobs_registry[job_id]
        config_path = job_info['config_path']
        
        if not os.path.exists(config_path):
            raise ConfigurationException(f"Configuration file {config_path} not found")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def update_pipeline_status(self, job_id: str, status: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Update pipeline job status.
        
        Args:
            job_id: Unique job ID
            status: New status (created, running, completed, failed, cancelled)
            metadata: Optional additional metadata
            
        Returns:
            Updated job information
        """
        if job_id not in _pipeline_jobs_registry:
            raise ConfigurationException(f"Pipeline job {job_id} not found")
        
        job_info = _pipeline_jobs_registry[job_id]
        job_info['status'] = status
        job_info['updated_at'] = datetime.now().isoformat()
        
        if metadata:
            job_info.update(metadata)
        
        # Also update the config file
        config = self.get_pipeline_config(job_id)
        config['pipeline_metadata']['status'] = status
        config['pipeline_metadata']['updated_at'] = datetime.now().isoformat()
        
        if metadata:
            config['pipeline_metadata'].update(metadata)
        
        with open(job_info['config_path'], 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, indent=2)
        
        return job_info
    
    def list_pipeline_jobs(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all pipeline jobs, optionally filtered by status.
        
        Args:
            status: Optional status filter
            
        Returns:
            List of pipeline job information
        """
        jobs = list(_pipeline_jobs_registry.values())
        
        if status:
            jobs = [job for job in jobs if job['status'] == status]
        
        return jobs
    
    def delete_pipeline_job(self, job_id: str) -> Dict[str, Any]:
        """
        Delete a pipeline job and its configuration.
        
        Args:
            job_id: Unique job ID
            
        Returns:
            Deletion confirmation
        """
        if job_id not in _pipeline_jobs_registry:
            raise ConfigurationException(f"Pipeline job {job_id} not found")
        
        job_info = _pipeline_jobs_registry[job_id]
        config_path = job_info['config_path']
        
        # Delete configuration file
        if os.path.exists(config_path):
            os.remove(config_path)
        
        # Remove from registry
        del _pipeline_jobs_registry[job_id]
        
        return {
            'job_id': job_id,
            'status': 'deleted',
            'message': f"Pipeline job {job_id} deleted successfully"
        }
    
    def get_service_config_from_pipeline(self, job_id: str, service_name: str) -> Dict[str, Any]:
        """
        Extract specific service configuration from pipeline config.
        
        Args:
            job_id: Unique job ID
            service_name: Name of the service
            
        Returns:
            Service-specific configuration
        """
        pipeline_config = self.get_pipeline_config(job_id)
        
        if service_name not in pipeline_config:
            raise ConfigurationException(f"Service {service_name} not found in pipeline job {job_id}")
        
        # Include global, paths, and midi sections along with service-specific config
        service_config = {
            'global': pipeline_config.get('global', {}),
            'paths': pipeline_config.get('paths', {}),
            'midi': pipeline_config.get('midi', {}),
            service_name: pipeline_config[service_name],
            'pipeline_metadata': pipeline_config.get('pipeline_metadata', {})
        }
        
        return service_config
    
    def get_overridable_parameters(self, service_name: str) -> Dict[str, Any]:
        """
        Get a template showing what parameters can be overridden for a specific service.
        
        Args:
            service_name: Name of the service
            
        Returns:
            Dictionary showing overridable parameters with their current values
        """
        base_config = self._create_base_config()
        service_config = base_config.get_config_for_service(service_name)
        
        if service_config is None:
            raise ConfigurationException(f"Service '{service_name}' not found")
        
        return {
            'service': service_name,
            'overridable_parameters': service_config.dict(),
            'example_override': {
                service_name: {
                    'batch_size': 64,  # Example override
                    'num_workers': 8   # Example override
                }
            }
        }


# Initialize global service instance
pipeline_config_service = PipelineConfigService()


def get_service_config_template(service_name: str) -> Dict[str, Any]:
    """
    Get a configuration template for a specific service.
    
    Args:
        service_name: Name of the service
        
    Returns:
        Service configuration template
    """
    base_config = FullConfig()
    service_config = base_config.get_config_for_service(service_name)
    
    if service_config is None:
        raise ConfigurationException(f"Service '{service_name}' not found in base configuration")
    
    return {
        'metadata': {
            'service': service_name,
            'template_version': '1.0'
        },
        service_name: service_config.dict()
    }


def list_available_services() -> List[str]:
    """
    List all available services in the configuration.
    
    Returns:
        List of available service names
    """
    # Define the available services based on the FullConfig class
    available_services = [
        'generator',
        'multimodal_mapping', 
        'vae',
        'training',
        'data',
        'evaluation',
        'dataloader'
    ]
    
    return sorted(available_services) 