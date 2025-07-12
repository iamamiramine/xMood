import os
import yaml
import uuid
import json
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

from application.shared.services.config_service import config_service
from domain.constants.paths_constants import CONFIG_PATH
from domain.constants.model_constants import ModelConstants
from domain.exceptions.global_exceptions import ConfigurationError

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
            overrides: Configuration overrides
            
        Returns:
            Dictionary containing pipeline config with unique job ID
        """
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Generate job name if not provided
        if not job_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            job_name = f"pipeline_job_{timestamp}"
        
        # Create base configuration
        base_config = create_base_config()
        
        # Apply environment-specific overrides
        env_config = create_environment_config(environment)
        if env_config:
            base_config = config_service._deep_merge(base_config, env_config)
        
        # Apply custom overrides
        if overrides:
            base_config = config_service._deep_merge(base_config, overrides)
        
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
            'global': base_config['global'],
            'data': base_config['data']
        }
        
        # Add only requested services
        for service in services:
            if service in base_config:
                filtered_config[service] = base_config[service]
        
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
    
    def get_pipeline_config(self, job_id: str) -> Dict[str, Any]:
        """
        Get pipeline configuration by job ID.
        
        Args:
            job_id: Unique job ID
            
        Returns:
            Pipeline configuration dictionary
        """
        if job_id not in _pipeline_jobs_registry:
            raise ConfigurationError(f"Pipeline job {job_id} not found")
        
        job_info = _pipeline_jobs_registry[job_id]
        config_path = job_info['config_path']
        
        if not os.path.exists(config_path):
            raise ConfigurationError(f"Configuration file {config_path} not found")
        
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
            raise ConfigurationError(f"Pipeline job {job_id} not found")
        
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
            raise ConfigurationError(f"Pipeline job {job_id} not found")
        
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
            raise ConfigurationError(f"Service {service_name} not found in pipeline job {job_id}")
        
        # Include global and data sections along with service-specific config
        service_config = {
            'global': pipeline_config['global'],
            'data': pipeline_config['data'],
            service_name: pipeline_config[service_name],
            'pipeline_metadata': pipeline_config['pipeline_metadata']
        }
        
        return service_config


# Initialize global service instance
pipeline_config_service = PipelineConfigService()

def create_base_config() -> Dict[str, Any]:
    """
    Create a base configuration with standardized parameters for all services.
    
    Returns:
        Base configuration dictionary with standardized parameter names
    """
    base_config = {
        'metadata': {
            'version': '1.0',
            'schema': 'pa-ai-unified-config-v1',
            'description': 'Unified configuration for PA-AI-2 generative module',
            'generated_by': 'pipeline_config_service'
        },
        
        'global': {
            'device': 'cuda',
            'seed': 42,
            'debug': False,
            'output_root': 'output/',
            'checkpoints_root': 'checkpoints/',
            'logs_root': 'logs/'
        },
        
        'data': {
            'dataset_name': 'ReMIDICaps',
            'context_size': ModelConstants.DEFAULT_CONTEXT_SIZE,
            'max_positions': ModelConstants.DEFAULT_MAX_POSITIONS,
            'max_bars': ModelConstants.DEFAULT_MAX_BARS,
            'max_bars_per_context': -1,
            'max_contexts_per_file': -1,
            'bar_token_mask': None,
            'bar_token_idx': 2,
            'batch_size': ModelConstants.DEFAULT_BATCH_SIZE,
            'num_workers': 2,
            'pin_memory': True,
            'load_latent': True,
            'load_symb': True,
            'load_emotions': True,
            'load_global_features': True,
            'load_text_prompts': True,
            'train_val_test_split': [0.7, 0.2, 0.1]
        },
        
        'classifier': {
            'd_model': ModelConstants.DEFAULT_D_MODEL,
            'context_size': ModelConstants.DEFAULT_CONTEXT_SIZE,
            'batch_size': ModelConstants.DEFAULT_BATCH_SIZE,
            'num_attention_heads': ModelConstants.DEFAULT_NUM_ATTENTION_HEADS,
            'cls_type': 'MOOD',
            'num_of_dim': ModelConstants.MOOD_DIMENSIONS,
            'lstm_hidden_dim': 128,
            'vocab_size': ModelConstants.DEFAULT_VOCAB_SIZE,
            'r': 14,
            'da': 128,
            'dropout': ModelConstants.DEFAULT_DROPOUT
        },
        
        'generator': {
            'd_model': ModelConstants.DEFAULT_D_MODEL,
            'context_size': ModelConstants.DEFAULT_CONTEXT_SIZE,
            'batch_size': ModelConstants.DEFAULT_BATCH_SIZE,
            'num_attention_heads': ModelConstants.DEFAULT_NUM_ATTENTION_HEADS,
            'd_latent': ModelConstants.DEFAULT_D_LATENT,
            'max_bars': ModelConstants.DEFAULT_MAX_BARS,
            'max_positions': ModelConstants.DEFAULT_MAX_POSITIONS,
            'encoder_layers': ModelConstants.DEFAULT_ENCODER_LAYERS,
            'decoder_layers': ModelConstants.DEFAULT_DECODER_LAYERS,
            'intermediate_size': ModelConstants.DEFAULT_INTERMEDIATE_SIZE,
            'vocab_size': ModelConstants.DEFAULT_VOCAB_SIZE,
            'lr': ModelConstants.DEFAULT_LEARNING_RATE,
            'lr_schedule': 'sqrt_decay',
            'warmup_steps': ModelConstants.DEFAULT_WARMUP_STEPS,
            'max_steps': ModelConstants.DEFAULT_MAX_STEPS,
            'dropout': ModelConstants.DEFAULT_DROPOUT
        },
        
        'multimodal_mapping': {
            'd_model': ModelConstants.DEFAULT_D_MODEL,
            'context_size': ModelConstants.DEFAULT_CONTEXT_SIZE,
            'batch_size': ModelConstants.DEFAULT_BATCH_SIZE,
            'num_attention_heads': ModelConstants.DEFAULT_NUM_ATTENTION_HEADS,
            'image_dim': ModelConstants.DEFAULT_IMAGE_DIM,
            'text_dim': ModelConstants.DEFAULT_TEXT_DIM,
            'global_feature_dim': ModelConstants.DEFAULT_GLOBAL_FEATURE_DIM,
            'global_feature_out_dim': ModelConstants.DEFAULT_GLOBAL_FEATURE_OUT_DIM,
            'mood_dim': ModelConstants.DEFAULT_MOOD_DIM,
            'mood_out_dim': ModelConstants.DEFAULT_MOOD_OUT_DIM,
            'fusion_dim': ModelConstants.DEFAULT_FUSION_DIM,
            'latent_dim': ModelConstants.DEFAULT_LATENT_DIM,
            'output_dim': ModelConstants.DEFAULT_OUTPUT_DIM,
            'num_layers': 4,
            'fusion_heads': 8,
            'fusion_type': 'linear_concat',
            'dropout': ModelConstants.DEFAULT_DROPOUT,
            'lr': 5e-5,
            'lr_schedule': 'cosine',
            'warmup_steps': 2000,
            'max_steps': 100000,
            'use_kl_annealing': True,
            'kl_start': 0.0,
            'kl_end': 1.0,
            'kl_anneal_steps': 10000
        },
        
        'vae': {
            'd_model': ModelConstants.DEFAULT_D_MODEL,
            'context_size': ModelConstants.DEFAULT_CONTEXT_SIZE,
            'batch_size': ModelConstants.DEFAULT_BATCH_SIZE,
            'num_attention_heads': ModelConstants.DEFAULT_NUM_ATTENTION_HEADS,
            'n_codes': ModelConstants.DEFAULT_N_CODES,
            'n_groups': ModelConstants.DEFAULT_N_GROUPS,
            'd_latent': ModelConstants.DEFAULT_D_LATENT,
            'encoder_layers': ModelConstants.DEFAULT_ENCODER_LAYERS,
            'decoder_layers': ModelConstants.DEFAULT_DECODER_LAYERS,
            'encoder_ffn_dim': ModelConstants.DEFAULT_INTERMEDIATE_SIZE,
            'decoder_ffn_dim': ModelConstants.DEFAULT_INTERMEDIATE_SIZE,
            'lr': ModelConstants.DEFAULT_LEARNING_RATE,
            'lr_schedule': 'sqrt_decay',
            'warmup_steps': ModelConstants.DEFAULT_WARMUP_STEPS,
            'max_steps': ModelConstants.DEFAULT_MAX_STEPS,
            'windowed_attention_pr': 0.0,
            'max_lookahead': 4,
            'disable_vq': False,
            'beta': 0.02,
            'cycle_length': 2000,
            'decay': ModelConstants.DEFAULT_VQ_DECAY,
            'eps': ModelConstants.DEFAULT_VQ_EPS,
            'restart_threshold': ModelConstants.DEFAULT_VQ_RESTART_THRESHOLD,
            'dropout': ModelConstants.DEFAULT_DROPOUT
        },
        
        'training': {
            'epochs': 100,
            'lr': ModelConstants.DEFAULT_LEARNING_RATE,
            'weight_decay': ModelConstants.DEFAULT_WEIGHT_DECAY,
            'warmup_steps': ModelConstants.DEFAULT_WARMUP_STEPS,
            'max_steps': ModelConstants.DEFAULT_MAX_STEPS,
            'val_check_interval': ModelConstants.DEFAULT_VAL_CHECK_INTERVAL,
            'log_every_n_steps': ModelConstants.DEFAULT_LOG_EVERY_N_STEPS,
            'checkpoint_every_n_steps': ModelConstants.DEFAULT_CHECKPOINT_EVERY_N_STEPS,
            'save_top_k': ModelConstants.DEFAULT_SAVE_TOP_K,
            'limit_val_batches': ModelConstants.DEFAULT_VALIDATION_BATCHES,
            'num_sanity_val_steps': ModelConstants.DEFAULT_SANITY_VAL_STEPS,
            'lr_schedule': 'cosine'
        },
        
        'evaluation': {
            'output_dir': 'output/evaluation/',
            'metrics': [
                'macro_overlapping_area',
                'f1_score',
                'standard_accuracy',
                'nrmse',
                'kl_divergence'
            ],
            'batch_size': 10,
            'calculate_aggregates': True,
            'parameters': {
                'measure_resolution': 4,
                'threshold': 2,
                'max_bars': 16,
                'max_time': 60.0
            }
        },
        
        'dataloader': {
            'dataset_name': 'ReMIDICaps',
            'context_size': ModelConstants.DEFAULT_CONTEXT_SIZE,
            'max_positions': ModelConstants.DEFAULT_MAX_POSITIONS,
            'max_bars': ModelConstants.DEFAULT_MAX_BARS,
            'max_bars_per_context': -1,
            'max_contexts_per_file': -1,
            'bar_token_mask': None,
            'bar_token_idx': 2,
            'batch_size': ModelConstants.DEFAULT_BATCH_SIZE,
            'num_workers': 2,
            'pin_memory': True,
            'train_val_test_split': [0.7, 0.2, 0.1],
            'load_latent': True,
            'load_symb': True,
            'load_emotions': True,
            'load_global_features': False,
            'load_text_prompts': False,
            'encode': False,
            'caption': False
        },
        
        'pseudo_labeller': {
            'device': 'cuda',
            'model_name': 'ViT-B/32',
            'batch_size': 8,
            'valid_extensions': ['.png', '.jpg', '.jpeg', '.bmp', '.gif'],
            'mood_categories': [
                'relaxing', 'christmas', 'dramatic', 'meditative',
                'energetic', 'happy', 'motivational', 'dark', 'love'
            ],
            'return_probabilities': True,
            'return_top_k': 3,
            'num_workers': 2,
            'save_individual': False
        }
    }
    
    return base_config


def create_environment_config(environment: str) -> Dict[str, Any]:
    """
    Create environment-specific configuration overrides.
    
    Args:
        environment: Environment name (development, staging, production)
        
    Returns:
        Environment-specific configuration dictionary
    """
    env_configs = {
        'development': {
            'global': {
                'device': 'cpu',
                'debug': True
            },
            'data': {
                'batch_size': 2,
                'num_workers': 1
            },
            'training': {
                'epochs': 5,
                'max_steps': 1000,
                'val_check_interval': 100,
                'log_every_n_steps': 10
            }
        },
        
        'staging': {
            'global': {
                'device': 'cuda',
                'debug': False
            },
            'data': {
                'batch_size': 8,
                'num_workers': 4
            },
            'training': {
                'epochs': 50,
                'max_steps': 50000,
                'val_check_interval': 500,
                'log_every_n_steps': 50
            }
        },
        
        'production': {
            'global': {
                'device': 'cuda',
                'debug': False
            },
            'data': {
                'batch_size': 32,
                'num_workers': 8
            },
            'training': {
                'epochs': 100,
                'max_steps': 100000,
                'val_check_interval': 1000,
                'log_every_n_steps': 100
            }
        }
    }
    
    return env_configs.get(environment, {})


def generate_unified_config(
    output_path: str,
    environment: str = 'production',
    services: Optional[List[str]] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate a unified YAML configuration file for all services.
    
    Args:
        output_path: Path to save the unified configuration
        environment: Environment name (development, staging, production)
        services: List of services to include (default: all services)
        overrides: Additional configuration overrides
        
    Returns:
        Generated unified configuration dictionary
    """
    # Create base configuration
    base_config = create_base_config()
    
    # Apply environment-specific overrides
    env_config = create_environment_config(environment)
    if env_config:
        base_config = config_service._deep_merge(base_config, env_config)
    
    # Apply custom overrides
    if overrides:
        base_config = config_service._deep_merge(base_config, overrides)
    
    # Filter services if specified
    if services:
        filtered_config = {
            'metadata': base_config['metadata'],
            'global': base_config['global']
        }
        
        # Add only requested services
        for service in services:
            if service in base_config:
                filtered_config[service] = base_config[service]
        
        base_config = filtered_config
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save unified configuration
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(base_config, f, default_flow_style=False, sort_keys=False, indent=2)
    
    return base_config


def migrate_json_to_yaml(
    json_config_path: str,
    yaml_output_path: str,
    apply_standardization: bool = True
) -> Dict[str, Any]:
    """
    Migrate existing JSON configuration to unified YAML format.
    
    Args:
        json_config_path: Path to existing JSON configuration
        yaml_output_path: Path to save the migrated YAML configuration
        apply_standardization: Whether to apply parameter name standardization
        
    Returns:
        Migrated configuration dictionary
    """
    import json
    
    # Load existing JSON configuration
    with open(json_config_path, 'r', encoding='utf-8') as f:
        json_config = json.load(f)
    
    # Apply standardization if requested
    if apply_standardization:
        json_config = standardize_parameter_names(json_config)
    
    # Create base configuration and merge with JSON config
    base_config = create_base_config()
    migrated_config = config_service._deep_merge(base_config, json_config)
    
    # Add migration metadata
    migrated_config['metadata'].update({
        'migrated_from': json_config_path,
        'migration_date': str(Path(json_config_path).stat().st_mtime),
        'standardization_applied': apply_standardization
    })
    
    # Save migrated configuration
    os.makedirs(os.path.dirname(yaml_output_path), exist_ok=True)
    with open(yaml_output_path, 'w', encoding='utf-8') as f:
        yaml.dump(migrated_config, f, default_flow_style=False, sort_keys=False, indent=2)
    
    return migrated_config


def standardize_parameter_names(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply parameter name standardization to configuration.
    
    Args:
        config: Configuration dictionary to standardize
        
    Returns:
        Configuration dictionary with standardized parameter names
    """
    # Parameter name mappings
    parameter_mappings = {
        'hidden_dim': 'd_model',
        'embedding_size': 'd_model',
        'max_length': 'context_size',
        'max_context_size': 'context_size',
        'target_batch_size': 'batch_size'
    }
    
    def apply_mappings(obj):
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                # Apply mapping if key exists in mappings
                new_key = parameter_mappings.get(key, key)
                result[new_key] = apply_mappings(value)
            return result
        elif isinstance(obj, list):
            return [apply_mappings(item) for item in obj]
        else:
            return obj
    
    return apply_mappings(config)


def validate_unified_config(config_path: str) -> Dict[str, Any]:
    """
    Validate a unified configuration file.
    
    Args:
        config_path: Path to configuration file to validate
        
    Returns:
        Validation results dictionary
    """
    try:
        # Load and validate configuration
        config = config_service.load_config(config_path, validate=True)
        
        return {
            'valid': True,
            'message': 'Configuration is valid',
            'config_path': config_path,
            'services': list(config.keys())
        }
    except ConfigurationError as e:
        return {
            'valid': False,
            'message': str(e),
            'config_path': config_path
        }


def get_service_config_template(service_name: str) -> Dict[str, Any]:
    """
    Get a configuration template for a specific service.
    
    Args:
        service_name: Name of the service
        
    Returns:
        Service configuration template
    """
    base_config = create_base_config()
    
    if service_name not in base_config:
        raise ConfigurationError(f"Service '{service_name}' not found in base configuration")
    
    return {
        'metadata': {
            'service': service_name,
            'template_version': '1.0'
        },
        service_name: base_config[service_name]
    }


def list_available_services() -> List[str]:
    """
    List all available services in the configuration.
    
    Returns:
        List of available service names
    """
    base_config = create_base_config()
    
    # Filter out metadata, global, data, training, and evaluation sections
    services = [key for key in base_config.keys() 
               if key not in ['metadata', 'global', 'data', 'training', 'evaluation']]
    
    return sorted(services) 