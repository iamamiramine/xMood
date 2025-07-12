"""
Unified Configuration Management Service

This service provides centralized configuration management with support for:
- Environment-specific configurations
- Configuration inheritance and merging
- Schema validation
- Hot reload capabilities
- Configuration caching
- Secure configuration handling
"""

import os
import yaml
import json
from typing import Dict, Any, Optional, List, Union, Set
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime
import threading
import hashlib
from contextlib import contextmanager

from domain.exceptions.global_exceptions import ConfigurationError
from domain.constants.paths_constants import get_config_value

logger = logging.getLogger(__name__)


class ConfigurationEnvironment(Enum):
    """Configuration environment enumeration."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class ConfigurationScope(Enum):
    """Configuration scope enumeration."""
    GLOBAL = "global"
    SERVICE = "service"
    PIPELINE = "pipeline"
    USER = "user"


@dataclass
class ConfigurationSource:
    """Configuration source information."""
    path: str
    environment: ConfigurationEnvironment
    scope: ConfigurationScope
    priority: int = 0
    last_modified: Optional[datetime] = None
    checksum: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfigurationValidationResult:
    """Configuration validation result."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    validated_config: Optional[Dict[str, Any]] = None


class UnifiedConfigService:
    """
    Unified Configuration Management Service.
    
    Provides centralized configuration management with support for multiple
    environments, configuration inheritance, validation, and hot reloading.
    """
    
    def __init__(self, base_config_dir: str = "shared/config"):
        self.base_config_dir = Path(base_config_dir)
        self.current_environment = ConfigurationEnvironment.PRODUCTION
        self._config_cache: Dict[str, ConfigurationSource] = {}
        self._merged_config: Dict[str, Any] = {}
        self._validation_schemas: Dict[str, Dict[str, Any]] = {}
        self._config_watchers: Set[str] = set()
        self._lock = threading.RLock()
        self._subscribers: List[callable] = []
        
        # Initialize default configuration paths
        self._config_paths = self._discover_config_files()
        
        logger.info(f"UnifiedConfigService initialized with base directory: {base_config_dir}")
    
    def set_environment(self, environment: Union[str, ConfigurationEnvironment]) -> None:
        """
        Set the current configuration environment.
        
        Args:
            environment: Environment name or ConfigurationEnvironment enum
        """
        if isinstance(environment, str):
            try:
                environment = ConfigurationEnvironment(environment.lower())
            except ValueError:
                logger.warning(f"Invalid environment '{environment}', using PRODUCTION")
                environment = ConfigurationEnvironment.PRODUCTION
        
        with self._lock:
            self.current_environment = environment
            # Clear cache to force reload with new environment
            self._config_cache.clear()
            self._merged_config.clear()
            
        logger.info(f"Environment set to: {self.current_environment.value}")
    
    def _discover_config_files(self) -> Dict[str, List[str]]:
        """
        Discover all configuration files in the base directory.
        
        Returns:
            Dictionary mapping configuration types to file paths
        """
        config_paths = {
            "base": [],
            "environment": [],
            "service": [],
            "override": []
        }
        
        if not self.base_config_dir.exists():
            logger.warning(f"Base configuration directory does not exist: {self.base_config_dir}")
            return config_paths
        
        try:
            # Discover base configurations
            for pattern in ["config.yaml", "config.yml", "base.yaml", "base.yml"]:
                for file_path in self.base_config_dir.glob(pattern):
                    config_paths["base"].append(str(file_path))
            
            # Discover environment configurations
            for env in ConfigurationEnvironment:
                for pattern in [f"{env.value}.yaml", f"{env.value}.yml", f"env_{env.value}.yaml"]:
                    for file_path in self.base_config_dir.glob(pattern):
                        config_paths["environment"].append(str(file_path))
            
            # Discover service configurations
            for pattern in ["services.yaml", "services.yml", "service_*.yaml"]:
                for file_path in self.base_config_dir.glob(pattern):
                    config_paths["service"].append(str(file_path))
            
            # Discover override configurations
            for pattern in ["override.yaml", "override.yml", "local.yaml", "local.yml"]:
                for file_path in self.base_config_dir.glob(pattern):
                    config_paths["override"].append(str(file_path))
                    
        except Exception as e:
            logger.error(f"Error discovering configuration files: {e}")
        
        return config_paths
    
    def set_environment(self, environment: Union[str, ConfigurationEnvironment]) -> None:
        """
        Set the current configuration environment.
        
        Args:
            environment: Environment to set (string or enum)
        """
        if isinstance(environment, str):
            try:
                environment = ConfigurationEnvironment(environment.lower())
            except ValueError:
                raise ConfigurationError(f"Invalid environment: {environment}")
        
        with self._lock:
            old_environment = self.current_environment
            self.current_environment = environment
            
            if old_environment != environment:
                logger.info(f"Configuration environment changed from {old_environment.value} to {environment.value}")
                self._invalidate_cache()
                self._notify_subscribers("environment_changed", {
                    "old_environment": old_environment.value,
                    "new_environment": environment.value
                })
    
    def load_configuration(self, force_reload: bool = False) -> Dict[str, Any]:
        """
        Load and merge all configurations for the current environment.
        
        Args:
            force_reload: Force reload from disk even if cached
            
        Returns:
            Merged configuration dictionary
        """
        with self._lock:
            cache_key = f"merged_{self.current_environment.value}"
            
            if not force_reload and cache_key in self._config_cache:
                return self._config_cache[cache_key].data
            
            try:
                # Load configurations in priority order
                configurations = []
                
                # 1. Load base configurations
                for config_path in self._config_paths["base"]:
                    config_source = self._load_config_file(
                        config_path, 
                        self.current_environment, 
                        ConfigurationScope.GLOBAL, 
                        priority=1
                    )
                    if config_source:
                        configurations.append(config_source)
                
                # 2. Load environment-specific configurations
                for config_path in self._config_paths["environment"]:
                    if self.current_environment.value in config_path:
                        config_source = self._load_config_file(
                            config_path, 
                            self.current_environment, 
                            ConfigurationScope.GLOBAL, 
                            priority=2
                        )
                        if config_source:
                            configurations.append(config_source)
                
                # 3. Load service configurations
                for config_path in self._config_paths["service"]:
                    config_source = self._load_config_file(
                        config_path, 
                        self.current_environment, 
                        ConfigurationScope.SERVICE, 
                        priority=3
                    )
                    if config_source:
                        configurations.append(config_source)
                
                # 4. Load override configurations
                for config_path in self._config_paths["override"]:
                    config_source = self._load_config_file(
                        config_path, 
                        self.current_environment, 
                        ConfigurationScope.USER, 
                        priority=4
                    )
                    if config_source:
                        configurations.append(config_source)
                
                # Sort by priority and merge
                configurations.sort(key=lambda x: x.priority)
                merged_config = self._merge_configurations(configurations)
                
                # Apply environment variable overrides
                merged_config = self._apply_environment_overrides(merged_config)
                
                # Cache the merged configuration
                merged_source = ConfigurationSource(
                    path=cache_key,
                    environment=self.current_environment,
                    scope=ConfigurationScope.GLOBAL,
                    priority=999,
                    last_modified=datetime.now(),
                    data=merged_config
                )
                self._config_cache[cache_key] = merged_source
                self._merged_config = merged_config
                
                logger.info(f"Successfully loaded configuration for environment: {self.current_environment.value}")
                return merged_config
                
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
                raise ConfigurationError(f"Failed to load configuration: {e}")
    
    def _load_config_file(self, file_path: str, environment: ConfigurationEnvironment, 
                         scope: ConfigurationScope, priority: int) -> Optional[ConfigurationSource]:
        """
        Load a single configuration file.
        
        Args:
            file_path: Path to configuration file
            environment: Configuration environment
            scope: Configuration scope
            priority: Priority for merging
            
        Returns:
            ConfigurationSource or None if loading failed
        """
        try:
            if not os.path.exists(file_path):
                return None
            
            # Check cache first
            file_stat = os.stat(file_path)
            last_modified = datetime.fromtimestamp(file_stat.st_mtime)
            
            if file_path in self._config_cache:
                cached_source = self._config_cache[file_path]
                if cached_source.last_modified and cached_source.last_modified >= last_modified:
                    return cached_source
            
            # Load file content
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith(('.yaml', '.yml')):
                    data = yaml.safe_load(f) or {}
                elif file_path.endswith('.json'):
                    data = json.load(f) or {}
                else:
                    logger.warning(f"Unsupported configuration file format: {file_path}")
                    return None
            
            # Calculate checksum
            checksum = self._calculate_checksum(data)
            
            # Create configuration source
            config_source = ConfigurationSource(
                path=file_path,
                environment=environment,
                scope=scope,
                priority=priority,
                last_modified=last_modified,
                checksum=checksum,
                data=data
            )
            
            # Cache the configuration
            self._config_cache[file_path] = config_source
            
            logger.debug(f"Loaded configuration from: {file_path}")
            return config_source
            
        except Exception as e:
            logger.error(f"Error loading configuration file {file_path}: {e}")
            return None
    
    def _merge_configurations(self, configurations: List[ConfigurationSource]) -> Dict[str, Any]:
        """
        Merge multiple configuration sources.
        
        Args:
            configurations: List of configuration sources to merge
            
        Returns:
            Merged configuration dictionary
        """
        merged = {}
        
        for config_source in configurations:
            merged = self._deep_merge(merged, config_source.data)
        
        return merged
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries.
        
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
    
    def _apply_environment_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply environment variable overrides to configuration.
        
        Args:
            config: Base configuration
            
        Returns:
            Configuration with environment overrides applied
        """
        # Apply environment variables with prefix PA_AI_CONFIG_
        env_prefix = "PA_AI_CONFIG_"
        
        for env_var, env_value in os.environ.items():
            if env_var.startswith(env_prefix):
                # Convert environment variable name to config key
                config_key = env_var[len(env_prefix):].lower().replace('_', '.')
                
                # Parse value (try JSON first, then string)
                try:
                    parsed_value = json.loads(env_value)
                except json.JSONDecodeError:
                    parsed_value = env_value
                
                # Set nested key
                self._set_nested_key(config, config_key, parsed_value)
        
        return config
    
    def _set_nested_key(self, config: Dict[str, Any], key_path: str, value: Any) -> None:
        """
        Set a nested key in configuration dictionary.
        
        Args:
            config: Configuration dictionary
            key_path: Dot-separated key path
            value: Value to set
        """
        keys = key_path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _calculate_checksum(self, data: Dict[str, Any]) -> str:
        """
        Calculate checksum for configuration data.
        
        Args:
            data: Configuration data
            
        Returns:
            Checksum string
        """
        serialized = json.dumps(data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]
    
    def _invalidate_cache(self) -> None:
        """Invalidate configuration cache."""
        self._config_cache.clear()
        self._merged_config.clear()
        logger.debug("Configuration cache invalidated")
    
    def _notify_subscribers(self, event: str, data: Dict[str, Any]) -> None:
        """
        Notify configuration change subscribers.
        
        Args:
            event: Event type
            data: Event data
        """
        for subscriber in self._subscribers:
            try:
                subscriber(event, data)
            except Exception as e:
                logger.error(f"Error notifying configuration subscriber: {e}")
    
    def get_service_config(self, service_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific service.
        
        Args:
            service_name: Name of the service
            
        Returns:
            Service configuration
        """
        config = self.load_configuration()
        return config.get("services", {}).get(service_name, {})
    
    def get_config_value(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value by key path.
        
        Args:
            key_path: Dot-separated key path
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        config = self.load_configuration()
        
        keys = key_path.split('.')
        current = config
        
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default
    
    def subscribe_to_changes(self, callback: callable) -> None:
        """
        Subscribe to configuration changes.
        
        Args:
            callback: Callback function to call on changes
        """
        self._subscribers.append(callback)
    
    def get_configuration_info(self) -> Dict[str, Any]:
        """
        Get information about loaded configurations.
        
        Returns:
            Configuration information dictionary
        """
        return {
            "current_environment": self.current_environment.value,
            "loaded_sources": [
                {
                    "path": source.path,
                    "environment": source.environment.value,
                    "scope": source.scope.value,
                    "priority": source.priority,
                    "last_modified": source.last_modified.isoformat() if source.last_modified else None,
                    "checksum": source.checksum
                }
                for source in self._config_cache.values()
            ],
            "cache_size": len(self._config_cache),
            "subscribers": len(self._subscribers)
        }


# Global unified configuration service instance
_unified_config_service: Optional[UnifiedConfigService] = None


def get_unified_config_service() -> UnifiedConfigService:
    """Get the global unified configuration service instance."""
    global _unified_config_service
    if _unified_config_service is None:
        _unified_config_service = UnifiedConfigService()
    return _unified_config_service 