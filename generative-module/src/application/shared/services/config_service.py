import os
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path
from pydantic import ValidationError

from domain.models.config.config_schema import (
    FullConfig, 
    ClassifierConfig, 
    GeneratorConfig, 
    MultimodalConfig, 
    VAEConfig,
    TrainingConfig,
    DataConfig
)
from domain.constants.paths_constants import CONFIG_PATH
from domain.exceptions.global_exceptions import ConfigurationError


class ConfigService:
    """
    Unified configuration service for loading and validating YAML configurations.
    Supports configuration inheritance (base + override) and schema validation.
    """
    
    def __init__(self):
        self.base_config_path = os.path.join(CONFIG_PATH, "base_config.yaml")
        self.env_config_path = os.path.join(CONFIG_PATH, "env_config.yaml")
        self._config_cache: Dict[str, Dict] = {}
    
    def load_config(self, config_path: str, validate: bool = True) -> Dict[str, Any]:
        """
        Load and validate configuration from YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
            validate: Whether to validate the configuration against schema
            
        Returns:
            Validated configuration dictionary
            
        Raises:
            ConfigurationError: If configuration is invalid or file not found
        """
        try:
            # Check cache first
            if config_path in self._config_cache:
                return self._config_cache[config_path]
            
            # Load YAML configuration
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            if config_data is None:
                raise ConfigurationError(f"Configuration file {config_path} is empty")
            
            # Apply inheritance if specified
            if 'inherit_from' in config_data:
                config_data = self._apply_inheritance(config_data, config_path)
            
            # Validate configuration if requested
            if validate:
                config_data = self._validate_config(config_data)
            
            # Cache the result
            self._config_cache[config_path] = config_data
            
            return config_data
            
        except FileNotFoundError:
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML syntax in {config_path}: {e}")
        except ValidationError as e:
            raise ConfigurationError(f"Configuration validation failed: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration: {e}")
    
    def _apply_inheritance(self, config: Dict[str, Any], current_path: str) -> Dict[str, Any]:
        """
        Apply configuration inheritance by merging base configuration with current config.
        
        Args:
            config: Current configuration dictionary
            current_path: Path to current configuration file
            
        Returns:
            Merged configuration dictionary
        """
        inherit_from = config.pop('inherit_from')
        
        # Resolve inheritance path relative to current config file
        base_path = os.path.join(os.path.dirname(current_path), inherit_from)
        
        # Load base configuration (without validation to avoid circular dependency)
        base_config = self.load_config(base_path, validate=False)
        
        # Deep merge base config with current config
        merged_config = self._deep_merge(base_config, config)
        
        return merged_config
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries, with override taking precedence.
        
        Args:
            base: Base dictionary
            override: Override dictionary
            
        Returns:
            Merged dictionary
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration against Pydantic schema.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Validated configuration dictionary
        """
        try:
            # Validate against full configuration schema
            validated_config = FullConfig(**config)
            return validated_config.dict()
        except ValidationError as e:
            # If full validation fails, try to validate individual components
            validated_sections = {}
            
            for section_name, section_config in config.items():
                try:
                    if section_name == 'classifier':
                        validated_sections[section_name] = ClassifierConfig(**section_config).dict()
                    elif section_name == 'generator':
                        validated_sections[section_name] = GeneratorConfig(**section_config).dict()
                    elif section_name == 'multimodal_mapping':
                        validated_sections[section_name] = MultimodalConfig(**section_config).dict()
                    elif section_name == 'vae':
                        validated_sections[section_name] = VAEConfig(**section_config).dict()
                    elif section_name == 'training':
                        validated_sections[section_name] = TrainingConfig(**section_config).dict()
                    elif section_name == 'data':
                        validated_sections[section_name] = DataConfig(**section_config).dict()
                    else:
                        # Keep unvalidated sections as-is
                        validated_sections[section_name] = section_config
                except ValidationError as section_error:
                    # Log warning but continue with unvalidated section
                    print(f"Warning: Could not validate {section_name} section: {section_error}")
                    validated_sections[section_name] = section_config
            
            return validated_sections
    
    def get_service_config(self, service_name: str, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Get configuration for a specific service.
        
        Args:
            service_name: Name of the service (e.g., 'classifier', 'generator')
            config_path: Optional path to configuration file
            
        Returns:
            Service-specific configuration dictionary
        """
        if config_path is None:
            config_path = self.base_config_path
        
        full_config = self.load_config(config_path)
        
        if service_name not in full_config:
            raise ConfigurationError(f"Service '{service_name}' not found in configuration")
        
        return full_config[service_name]
    
    def create_unified_config(self, output_path: str, services: Optional[list] = None) -> Dict[str, Any]:
        """
        Create a unified configuration file from multiple service configurations.
        
        Args:
            output_path: Path to save the unified configuration
            services: List of services to include (default: all services)
            
        Returns:
            Unified configuration dictionary
        """
        if services is None:
            services = ['classifier', 'generator', 'multimodal_mapping', 'vae', 'training', 'data']
        
        unified_config = {}
        
        # Add metadata
        unified_config['metadata'] = {
            'version': '1.0',
            'schema': 'pa-ai-unified-config-v1',
            'generated_by': 'ConfigService.create_unified_config'
        }
        
        # Load and merge service configurations
        for service in services:
            try:
                service_config = self.get_service_config(service)
                unified_config[service] = service_config
            except ConfigurationError as e:
                print(f"Warning: Could not load {service} configuration: {e}")
        
        # Add global settings
        unified_config['global'] = {
            'device': 'cuda',
            'seed': 42,
            'debug': False
        }
        
        # Save unified configuration
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(unified_config, f, default_flow_style=False, sort_keys=False, indent=2)
        
        return unified_config
    
    def validate_config_file(self, config_path: str) -> bool:
        """
        Validate a configuration file without loading it into cache.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            True if valid, False otherwise
        """
        try:
            self.load_config(config_path, validate=True)
            return True
        except ConfigurationError:
            return False
    
    def clear_cache(self):
        """Clear the configuration cache."""
        self._config_cache.clear()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached configurations."""
        return {
            'cached_files': list(self._config_cache.keys()),
            'cache_size': len(self._config_cache)
        }


# Global configuration service instance
config_service = ConfigService() 