"""
Config Service Adapter

This adapter implements the IConfigService interface for the existing
config service, enabling dependency injection and reducing cross-service
dependencies.
"""

from typing import Dict, Any, Optional
import logging

from domain.interfaces.service_interfaces import IConfigService
from application.shared.services.config_service import config_service

logger = logging.getLogger(__name__)


class ConfigServiceAdapter(IConfigService):
    """
    Adapter for the config service that implements the IConfigService interface.
    
    This adapter wraps the existing config service to provide
    a clean interface for dependency injection.
    """
    
    def __init__(self):
        self._config_service = config_service
        logger.info("ConfigServiceAdapter initialized")
    
    def load_config(self, config_path: str, validate: bool = True) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            validate: Whether to validate configuration
            
        Returns:
            Configuration dictionary
        """
        try:
            return self._config_service.load_config(config_path, validate)
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            return {}
    
    def get_service_config(self, service_name: str, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Get service-specific configuration.
        
        Args:
            service_name: Name of the service
            config_path: Optional path to configuration file
            
        Returns:
            Service-specific configuration
        """
        try:
            return self._config_service.get_service_config(service_name, config_path)
        except Exception as e:
            logger.error(f"Error getting service config for {service_name}: {e}")
            return {}
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            validated = self._config_service._validate_config(config)
            return validated is not None
        except Exception as e:
            logger.error(f"Error validating config: {e}")
            return False 