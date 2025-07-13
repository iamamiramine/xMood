"""
Pipeline Orchestration Service

This module provides the main pipeline service that orchestrates background jobs
for all PA-AI services. It handles YAML configuration parsing, parameter conversion
using the parameter model registry, and job execution management.
"""

import yaml
import asyncio
import logging
import uuid
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from datetime import datetime

from domain.schemas.config.config_schema import (
    FullConfig, 
    PARAMETER_MODEL_REGISTRY,
    GlobalConfig,
    PathsConfig,
    MidiConfig,
    PipelineMetadata,
    SymbolicServiceConfig,
    LatentServiceConfig,
    GeneratorServiceConfig,
    MultimodalServiceConfig
)
from domain.models.pipeline.pipeline_model import (
    PipelineJobRequest, 
    PipelineJobInfo, 
    PipelineExecutionRequest,
    PipelineExecutionResponse,
    ServiceType,
    ParameterSetupRequest,
    ParameterSetupResponse,
    CommonParameters
)
from application.pipeline.models.job_manager import job_manager

# Import all service functions
from application.symbolic.services.symbolic_service import (
    synthesize_midi as music_base_synthesize_midi,
    encode_midi as encoder_encode_midi,
    encode_dataset as encoder_encode_dataset,
    tokenize_remi_dataset as encoder_tokenize_remi_dataset,
)
from application.latent.services.latent_service import (
    train_vae as feature_train_vae,
    generate_latent_representations_dataset as feature_generate_latent_representations,
)
from application.generator.services.generator_service import (
    train_generator as generator_train,
    generate_from_midi as generator_generate_from_midi,
    batch_generate_from_dataset as generator_batch_generate,
)
from application.multimodal.services.multimodal_service import (
    train_multimodal_mapping as multimodal_train,
    generate_multimodal_representations as multimodal_generate_representations,
)


class PipelineService:
    """Main pipeline orchestration service"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Service function registry
        self.service_functions: Dict[str, Dict[str, Callable]] = {
            "symbolic": {
                "synthesize_midi": music_base_synthesize_midi,
                "encode_midi": encoder_encode_midi,
                "encode_dataset": encoder_encode_dataset,
                "tokenize_remi_dataset": encoder_tokenize_remi_dataset,
            },
            "latent": {
                "train_vae": feature_train_vae,
                "generate_latent_representations": feature_generate_latent_representations,
            },
            "generator": {
                "train": generator_train,
                "generate_from_midi": generator_generate_from_midi,
                "batch_generate": generator_batch_generate,
            },
            "multimodal": {
                "train": multimodal_train,
                "generate_representations": multimodal_generate_representations,
            },
        }
    
    def setup_parameters(self, request: ParameterSetupRequest) -> ParameterSetupResponse:
        """
        Set up parameters for pipeline configuration and export to YAML file.
        
        This method takes common parameters and service-specific parameters,
        creates a FullConfig object, and exports it as a YAML file with a unique ID.
        """
        try:
            # Generate unique configuration ID
            config_id = str(uuid.uuid4())
            config_name = request.config_name or f"config_{config_id[:8]}"
            
            # Create timestamp for configuration
            created_at = datetime.utcnow()
            
            # Start with common parameters
            common_params = request.common_parameters.model_dump()
            
            # Create global configuration
            global_config_data = {
                "device": common_params["device"],
                "seed": 42,
                "debug": False,
                "output_root": "output/",
                "checkpoints_root": "checkpoints/",
                "logs_root": "logs/"
            }
            
            # Override with any provided global config
            if request.global_config:
                global_config_data.update(request.global_config)
            
            # Create paths configuration
            paths_config_data = {
                "ROOT_OUTPUT": "output",
                "DATASETS_PATH": "datasets",
                "DATASET_NAME": common_params["dataset_name"]
            }
            
            # Override with any provided paths config
            if request.paths_config:
                paths_config_data.update(request.paths_config)
            
            # Create MIDI configuration
            midi_config_data = {
                "pos_per_quarter": 12,
                "resolution": 480,
                "max_bar_length": 3,
                "max_bars": common_params["max_bars"],
                "pad_idx": 338
            }
            
            # Override with any provided MIDI config
            if request.midi_config:
                midi_config_data.update(request.midi_config)
            
            # Create pipeline metadata
            metadata = {
                "version": "1.0",
                "schema": "pa-ai-unified-config-v1",
                "description": request.description or f"Configuration {config_name} created at {created_at}",
                "generated_by": "PipelineParameterSetupService"
            }
            
            # Create service configurations
            service_configs = self._create_service_configurations(common_params, request.service_parameters)
            
            # Create the full configuration
            full_config_data = {
                "metadata": metadata,
                "global": global_config_data,
                "paths": paths_config_data,
                "midi": midi_config_data,
                **service_configs
            }
            
            # Validate the configuration by creating FullConfig instance
            full_config = FullConfig.model_validate(full_config_data)
            
            # Export to YAML file
            config_file_path = self._export_config_to_yaml(config_id, config_name, full_config)
            
            # Create response
            response = ParameterSetupResponse(
                config_id=config_id,
                config_name=config_name,
                config_file_path=config_file_path,
                created_at=created_at,
                common_parameters_applied=common_params,
                service_parameters_applied=request.service_parameters.model_dump(),
                validation_passed=True,
                validation_warnings=[]
            )
            
            self.logger.info(f"Successfully created configuration {config_name} with ID {config_id}")
            return response
            
        except Exception as e:
            self.logger.error(f"Failed to setup parameters: {str(e)}")
            raise
    
    def _create_service_configurations(
        self, 
        common_params: Dict[str, Any], 
        service_params: Any
    ) -> Dict[str, Any]:
        """Create service-specific configurations by merging common and specific parameters"""
        
        service_configs = {}
        
        # Symbolic service configuration
        symbolic_config = {}
        if service_params.symbolic:
            # Create function-specific configs
            for func_name in ["synthesize_midi", "encode_midi", "encode_dataset", "tokenize_remi_dataset"]:
                if func_name in service_params.symbolic:
                    func_config = self._merge_common_with_specific(
                        common_params, 
                        service_params.symbolic[func_name],
                        func_name
                    )
                    symbolic_config[func_name] = func_config
        
        service_configs["symbolic_service"] = symbolic_config
        
        # Latent service configuration
        latent_config = {}
        if service_params.latent:
            for func_name in ["train_vae", "generate_latent_representations"]:
                if func_name in service_params.latent:
                    func_config = self._merge_common_with_specific(
                        common_params,
                        service_params.latent[func_name],
                        func_name
                    )
                    latent_config[func_name] = func_config
        
        service_configs["latent_service"] = latent_config
        
        # Generator service configuration
        generator_config = {}
        if service_params.generator:
            for func_name in ["train", "generate_from_midi", "batch_generate"]:
                if func_name in service_params.generator:
                    func_config = self._merge_common_with_specific(
                        common_params,
                        service_params.generator[func_name],
                        func_name
                    )
                    generator_config[func_name] = func_config
        
        service_configs["generator_service"] = generator_config
        
        # Multimodal service configuration
        multimodal_config = {}
        if service_params.multimodal:
            for func_name in ["train", "generate_representations"]:
                if func_name in service_params.multimodal:
                    func_config = self._merge_common_with_specific(
                        common_params,
                        service_params.multimodal[func_name],
                        func_name
                    )
                    multimodal_config[func_name] = func_config
        
        service_configs["multimodal_service"] = multimodal_config
        
        return service_configs
    
    def _merge_common_with_specific(
        self, 
        common_params: Dict[str, Any], 
        specific_params: Dict[str, Any],
        function_name: str
    ) -> Dict[str, Any]:
        """Merge common parameters with function-specific parameters"""
        
        # Start with common parameters
        merged_config = common_params.copy()
        
        # Add function-specific parameters (these override common ones)
        if specific_params:
            merged_config.update(specific_params)
        
        # Add any missing parameters that are required for specific functions
        self._add_function_specific_defaults(merged_config, function_name)
        
        return merged_config
    
    def _add_function_specific_defaults(self, config: Dict[str, Any], function_name: str):
        """Add function-specific default parameters that might be missing"""
        
        # Add defaults based on function requirements
        if function_name in ["train_vae", "train", "train_multimodal"]:
            # Training-specific defaults
            config.setdefault("max_epochs", 100)
            config.setdefault("gpus", 1)
            config.setdefault("accumulate_grad_batches", 1)
            config.setdefault("val_check_interval", 1.0)
            config.setdefault("log_every_n_steps", 50)
            config.setdefault("save_top_k", 3)
            config.setdefault("limit_val_batches", 100)
            config.setdefault("num_sanity_val_steps", 2)
            config.setdefault("every_n_train_steps", 1000)
            config.setdefault("checkpoint_dir", f"output/checkpoints/{function_name}")
            config.setdefault("weight_decay", 0.01)
            config.setdefault("lr_schedule", "cosine")
        
        elif function_name in ["generate_from_midi", "generate_representations"]:
            # Generation-specific defaults
            config.setdefault("temperature", 1.0)
            config.setdefault("max_n_tokens", 1024)
            config.setdefault("output_folder", "output/generated")
            config.setdefault("output_name", "generated_output")
        
        elif function_name in ["encode_dataset", "tokenize_remi_dataset"]:
            # Dataset processing defaults
            config.setdefault("max_files", None)
            config.setdefault("resume_from", None)
            config.setdefault("overwrite_existing", False)
            config.setdefault("validate_output", True)
            config.setdefault("skip_invalid", True)
            config.setdefault("save", True)
        
        # VAE-specific defaults
        if function_name == "train_vae":
            config.setdefault("n_codes", 1024)
            config.setdefault("n_groups", 4)
            config.setdefault("encoder_layers", 6)
            config.setdefault("decoder_layers", 6)
            config.setdefault("encoder_ffn_dim", 2048)
            config.setdefault("decoder_ffn_dim", 2048)
            config.setdefault("beta", 0.25)
            config.setdefault("disable_vq", False)
            config.setdefault("use_wandb", False)
        
        # Multimodal-specific defaults
        if function_name == "train" and "multimodal" in function_name:
            config.setdefault("image_dim", 512)
            config.setdefault("text_dim", 768)
            config.setdefault("fusion_dim", 512)
            config.setdefault("fusion_type", "linear_concat")
            config.setdefault("num_layers", 4)
            config.setdefault("num_heads", 8)
            config.setdefault("use_kl_annealing", True)
            config.setdefault("load_images", True)
    
    def _export_config_to_yaml(self, config_id: str, config_name: str, full_config: FullConfig) -> str:
        """Export FullConfig to YAML file with unique ID"""
        
        # Create config directory if it doesn't exist
        config_dir = Path("shared/config")
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with unique ID
        filename = f"{config_name}_{config_id[:8]}.yaml"
        config_file_path = config_dir / filename
        
        # Convert to dictionary and export to YAML
        config_dict = full_config.model_dump(by_alias=True, exclude_none=True)
        
        with open(config_file_path, 'w', encoding='utf-8') as file:
            yaml.dump(
                config_dict, 
                file, 
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
                indent=2
            )
        
        self.logger.info(f"Configuration exported to {config_file_path}")
        return str(config_file_path)
    
    def parse_yaml_config(self, config_path: str) -> FullConfig:
        """Parse YAML configuration file into FullConfig model"""
        
        try:
            config_path = Path(config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
            with open(config_path, 'r', encoding='utf-8') as file:
                yaml_data = yaml.safe_load(file)
            
            # Parse into FullConfig model
            full_config = FullConfig.model_validate(yaml_data)
            
            self.logger.info(f"Successfully parsed configuration from {config_path}")
            return full_config
            
        except Exception as e:
            self.logger.error(f"Failed to parse configuration: {str(e)}")
            raise
    
    def parse_config_data(self, config_data: Dict[str, Any]) -> FullConfig:
        """Parse configuration data dictionary into FullConfig model"""
        
        try:
            full_config = FullConfig.model_validate(config_data)
            self.logger.info("Successfully parsed configuration data")
            return full_config
            
        except Exception as e:
            self.logger.error(f"Failed to parse configuration data: {str(e)}")
            raise
    
    def get_service_parameters(
        self, 
        service_name: str, 
        function_name: str, 
        config: FullConfig
    ) -> Any:
        """Extract and convert service parameters from configuration"""
        
        try:
            # Get the parameter model class for this service function
            if service_name not in PARAMETER_MODEL_REGISTRY:
                raise ValueError(f"Unknown service: {service_name}")
            
            service_registry = PARAMETER_MODEL_REGISTRY[service_name]
            if function_name not in service_registry:
                raise ValueError(f"Unknown function {function_name} for service {service_name}")
            
            parameter_model_class = service_registry[function_name]
            
            # Get the service configuration
            service_config = config.get_config_for_service(service_name)
            if service_config is None:
                raise ValueError(f"No configuration found for service: {service_name}")
            
            # Extract function-specific configuration
            function_config = getattr(service_config, function_name, None)
            if function_config is None:
                raise ValueError(f"No configuration found for function: {function_name}")
            
            # Convert to parameter model
            parameters = parameter_model_class.model_validate(function_config)
            
            self.logger.info(f"Successfully extracted parameters for {service_name}.{function_name}")
            return parameters
            
        except Exception as e:
            self.logger.error(f"Failed to extract parameters: {str(e)}")
            raise
    
    def get_service_function(self, service_name: str, function_name: str) -> Callable:
        """Get service function by name"""
        
        if service_name not in self.service_functions:
            raise ValueError(f"Unknown service: {service_name}")
        
        service_funcs = self.service_functions[service_name]
        if function_name not in service_funcs:
            raise ValueError(f"Unknown function {function_name} for service {service_name}")
        
        return service_funcs[function_name]
    
    async def execute_job(self, request: PipelineJobRequest) -> PipelineJobInfo:
        """Execute a single pipeline job using the globally set configuration"""
        
        try:
            # Get the globally set configuration
            from domain.constants.paths_constants import get_config_file_path
            config_path = get_config_file_path()
            
            if not config_path:
                raise ValueError("No configuration file is set. Please use /configs/set to set a configuration file first.")
            
            # Parse configuration
            config = self.parse_yaml_config(config_path)
            
            # Get service function and parameters
            service_function = self.get_service_function(
                request.service_name, 
                request.function_name
            )
            parameters = self.get_service_parameters(
                request.service_name, 
                request.function_name, 
                config
            )
            
            # Create job
            job_info = job_manager.create_job(
                service_name=request.service_name,
                function_name=request.function_name,
                job_name=request.job_name
            )
            
            # Submit job for background execution
            await job_manager.submit_job(
                job_id=job_info.job_id,
                service_function=service_function,
                parameters=parameters
            )
            
            return job_info
            
        except Exception as e:
            self.logger.error(f"Failed to execute job: {str(e)}")
            raise
    
    async def execute_pipeline(self, request: PipelineExecutionRequest) -> PipelineExecutionResponse:
        """Execute a complete pipeline with multiple services using the globally set configuration"""
        
        try:
            # Get the globally set configuration
            from domain.constants.paths_constants import get_config_file_path
            config_path = get_config_file_path()
            
            if not config_path:
                raise ValueError("No configuration file is set. Please use /configs/set to set a configuration file first.")
            
            # Parse configuration
            config = self.parse_yaml_config(config_path)
            
            # Determine which services to execute
            services_to_execute = request.services if request.services else []
            
            # If no specific services provided, execute all configured services
            if not services_to_execute:
                services_to_execute = []
                for service_name in ServiceType:
                    service_config = config.get_config_for_service(service_name.value)
                    if service_config and self._has_configured_functions(service_config):
                        services_to_execute.append(service_name.value)
            
            # Create job execution plan
            job_ids = []
            execution_tasks = []
            
            for service_name in services_to_execute:
                service_config = config.get_config_for_service(service_name)
                if service_config is None:
                    continue
                
                # Find configured functions for this service
                configured_functions = self._get_configured_functions(service_config)
                
                for function_name in configured_functions:
                    # Create job request (no need for config_data since we use global config)
                    job_request = PipelineJobRequest(
                        service_name=ServiceType(service_name),
                        function_name=function_name,
                        job_name=f"{request.pipeline_name}_{service_name}_{function_name}"
                    )
                    
                    if request.parallel_execution:
                        # Execute in parallel
                        task = asyncio.create_task(self.execute_job(job_request))
                        execution_tasks.append(task)
                    else:
                        # Execute sequentially
                        job_info = await self.execute_job(job_request)
                        job_ids.append(job_info.job_id)
            
            # Wait for parallel tasks to complete
            if execution_tasks:
                job_infos = await asyncio.gather(*execution_tasks)
                job_ids.extend([job_info.job_id for job_info in job_infos])
            
            # Create pipeline execution response
            response = PipelineExecutionResponse(
                pipeline_name=request.pipeline_name,
                job_ids=job_ids
            )
            
            self.logger.info(f"Started pipeline '{request.pipeline_name}' with {len(job_ids)} jobs")
            return response
            
        except Exception as e:
            self.logger.error(f"Failed to execute pipeline: {str(e)}")
            raise
    
    def _has_configured_functions(self, service_config: Any) -> bool:
        """Check if service config has any configured functions"""
        
        # Check if any of the function configurations are not None
        for attr_name in dir(service_config):
            if not attr_name.startswith('_'):
                attr_value = getattr(service_config, attr_name)
                if attr_value is not None and isinstance(attr_value, dict):
                    return True
        return False
    
    def _get_configured_functions(self, service_config: Any) -> List[str]:
        """Get list of configured function names for a service"""
        
        configured_functions = []
        for attr_name in dir(service_config):
            if not attr_name.startswith('_'):
                attr_value = getattr(service_config, attr_name)
                if attr_value is not None and isinstance(attr_value, dict):
                    configured_functions.append(attr_name)
        return configured_functions
    
    def get_job_status(self, job_id: str) -> Optional[PipelineJobInfo]:
        """Get job status by ID"""
        return job_manager.get_job(job_id)
    
    def get_all_jobs(self) -> List[PipelineJobInfo]:
        """Get all pipeline jobs"""
        jobs = job_manager.get_all_jobs()
        return list(jobs.values())
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job"""
        return job_manager.cancel_job(job_id)
    
    def cleanup_completed_jobs(self, max_age_hours: int = 24) -> int:
        """Clean up completed jobs"""
        return job_manager.cleanup_completed_jobs(max_age_hours)


# Global pipeline service instance
pipeline_service = PipelineService() 