"""
Service Initialization Module

This module configures all service adapters with the dependency injection
container, enabling proper service discovery and dependency resolution.
"""

import logging
from typing import Dict, Any

from domain.interfaces import (
    get_dependency_container,
    ServiceScope,
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
from domain.interfaces.service_registry import get_enhanced_service_registry, ServiceStatus, CircuitBreakerConfig

from application.shared.adapters import (
    EncoderServiceAdapter,
    FeatureExtractionServiceAdapter,
    GeneratorServiceAdapter,
    MultimodalMappingServiceAdapter,
    MusicBaseServiceAdapter,
    DataloaderServiceAdapter,
    ConfigServiceAdapter,
    JobManagementServiceAdapter,
    PipelineConfigServiceAdapter,
)

logger = logging.getLogger(__name__)


def initialize_services() -> None:
    """
    Initialize all service adapters with unified service registry architecture.
    
    This function registers all service implementations with their interfaces
    in the enhanced service registry. The dependency injection container
    automatically uses the enhanced registry via an adapter, providing
    automatic dependency resolution, service discovery, health monitoring,
    and circuit breaker patterns.
    """
    logger.info("Initializing services with unified service registry architecture...")
    
    container = get_dependency_container()
    enhanced_registry = get_enhanced_service_registry()
    
    try:
        # Define service configurations with health checks
        service_configs = [
            {
                "name": "job_management_service",
                "interface": IJobManagementService,
                "adapter": JobManagementServiceAdapter,
                "description": "Job management and scheduling service",
                "tags": {"core", "jobs"},
                "dependencies": ["config_service"],
                "health_check": lambda: _check_job_management_service_health(),
                "circuit_breaker_config": CircuitBreakerConfig(failure_threshold=5, recovery_timeout=60)
            },
            {
                "name": "pipeline_config_service",
                "interface": IPipelineConfigService,
                "adapter": PipelineConfigServiceAdapter,
                "description": "Pipeline configuration service",
                "tags": {"core", "pipeline"},
                "dependencies": ["config_service"],
                "health_check": lambda: _check_pipeline_config_service_health(),
                "circuit_breaker_config": CircuitBreakerConfig(failure_threshold=3, recovery_timeout=30)
            },
            {
                "name": "encoder_service",
                "interface": IEncoderService,
                "adapter": EncoderServiceAdapter,
                "description": "Music encoding service",
                "tags": {"ml", "encoder"},
                "dependencies": ["config_service"],
                "health_check": lambda: _check_encoder_service_health(),
                "circuit_breaker_config": CircuitBreakerConfig(failure_threshold=5, recovery_timeout=120)
            },
            {
                "name": "feature_extraction_service",
                "interface": IFeatureExtractionService,
                "adapter": FeatureExtractionServiceAdapter,
                "description": "Feature extraction service",
                "tags": {"ml", "features"},
                "dependencies": ["config_service"],
                "health_check": lambda: _check_feature_extraction_service_health(),
                "circuit_breaker_config": CircuitBreakerConfig(failure_threshold=5, recovery_timeout=120)
            },
            {
                "name": "generator_service",
                "interface": IGeneratorService,
                "adapter": GeneratorServiceAdapter,
                "description": "Music generation service",
                "tags": {"ml", "generator"},
                "dependencies": ["config_service", "encoder_service"],
                "health_check": lambda: _check_generator_service_health(),
                "circuit_breaker_config": CircuitBreakerConfig(failure_threshold=5, recovery_timeout=120)
            },
            {
                "name": "multimodal_mapping_service",
                "interface": IMultimodalMappingService,
                "adapter": MultimodalMappingServiceAdapter,
                "description": "Multimodal mapping service",
                "tags": {"ml", "multimodal"},
                "dependencies": ["config_service"],
                "health_check": lambda: _check_multimodal_mapping_service_health(),
                "circuit_breaker_config": CircuitBreakerConfig(failure_threshold=5, recovery_timeout=120)
            },
            {
                "name": "music_base_service",
                "interface": IMusicBaseService,
                "adapter": MusicBaseServiceAdapter,
                "description": "Music base processing service",
                "tags": {"utility", "music"},
                "dependencies": ["config_service"],
                "health_check": lambda: _check_music_base_service_health(),
                "circuit_breaker_config": CircuitBreakerConfig(failure_threshold=3, recovery_timeout=60)
            },
            {
                "name": "dataloader_service",
                "interface": IDataloaderService,
                "adapter": DataloaderServiceAdapter,
                "description": "Data loading service",
                "tags": {"utility", "data"},
                "dependencies": ["config_service"],
                "health_check": lambda: _check_dataloader_service_health(),
                "circuit_breaker_config": CircuitBreakerConfig(failure_threshold=5, recovery_timeout=60)
            },

        ]
        
        # Register services in enhanced registry
        # Note: Dependency injection container automatically uses enhanced registry via adapter
        for config in service_configs:
            # Register with enhanced service registry
            enhanced_registry.register_service(
                service_name=config["name"],
                service_type=config["interface"],
                factory=config["adapter"],
                scope=ServiceScope.SINGLETON,
                version="1.0.0",
                description=config["description"],
                tags=config["tags"],
                dependencies=config.get("dependencies", []),
                health_check=config.get("health_check"),
                health_check_interval=60,
                circuit_breaker_enabled=True,
                circuit_breaker_config=config.get("circuit_breaker_config")
            )
            
            logger.info(f"Registered service: {config['name']}")
        
        logger.info("Successfully initialized all services with unified service registry architecture")
        
        # Log service statistics
        stats = container.get_resolution_statistics()
        registry_stats = enhanced_registry.get_registry_stats()
        logger.info(f"Service container statistics: {stats}")
        logger.info(f"Enhanced registry statistics: {registry_stats}")
        
    except Exception as e:
        logger.error(f"Error initializing services: {e}")
        raise


def get_service_health_check() -> Dict[str, Any]:
    """
    Perform a health check on all registered services using both DI container and enhanced service registry.
    
    Returns:
        Dictionary containing health check results for all services
    """
    container = get_dependency_container()
    enhanced_registry = get_enhanced_service_registry()
    health_results = {}
    
    # Get health status from enhanced service registry
    registry_services = enhanced_registry.get_all_services_info()
    
    for service_info in registry_services:
        service_name = service_info["name"]
        
        # Combine information from both systems
        health_results[service_name] = {
            "status": service_info["status"],
            "registered": True,
            "type": service_info["type"],
            "version": service_info["version"],
            "description": service_info["description"],
            "tags": service_info["tags"],
            "dependencies": service_info["dependencies"],
            "circuit_breaker_state": service_info["circuit_breaker_state"],
            "failure_count": service_info["failure_count"],
            "metrics": service_info["metrics"],
            "last_accessed": service_info["last_accessed"],
            "last_health_check": service_info["last_health_check"]
        }
        
        # Try to resolve from DI container as well
        try:
            service_interfaces = {
                "config_service": IConfigService,
                "job_management_service": IJobManagementService,
                "pipeline_config_service": IPipelineConfigService,
                "encoder_service": IEncoderService,
                "feature_extraction_service": IFeatureExtractionService,
                "generator_service": IGeneratorService,
                "multimodal_mapping_service": IMultimodalMappingService,
                "music_base_service": IMusicBaseService,
                "dataloader_service": IDataloaderService,
            }
            
            if service_name in service_interfaces:
                service = container.try_resolve(service_interfaces[service_name])
                if service:
                    health_results[service_name]["di_container_status"] = "available"
                    health_results[service_name]["instance_type"] = type(service).__name__
                else:
                    health_results[service_name]["di_container_status"] = "unavailable"
                    
        except Exception as e:
            health_results[service_name]["di_container_status"] = "error"
            health_results[service_name]["di_container_error"] = str(e)
    
    # Calculate overall health
    healthy_count = sum(1 for r in health_results.values() if r["status"] == "healthy")
    total_count = len(health_results)
    overall_status = "healthy" if healthy_count == total_count else "degraded" if healthy_count > 0 else "unhealthy"
    
    return {
        "overall_status": overall_status,
        "healthy_services": healthy_count,
        "total_services": total_count,
        "services": health_results,
        "container_stats": container.get_resolution_statistics(),
        "registry_stats": enhanced_registry.get_registry_stats()
    }


def reset_services() -> None:
    """
    Reset all service registrations and reinitialize.
    
    This function clears all service registrations and reinitializes
    both the dependency injection container and enhanced service registry.
    Useful for testing or configuration changes.
    """
    logger.info("Resetting services...")
    
    container = get_dependency_container()
    enhanced_registry = get_enhanced_service_registry()
    
    # Clear both registries
    container._registry.clear()
    enhanced_registry.clear()
    
    # Reinitialize
    initialize_services()
    
    logger.info("Services reset and reinitialized successfully")


def get_registered_services() -> Dict[str, Any]:
    """
    Get information about all registered services.
    
    Returns:
        Dictionary containing information about all registered services
    """
    container = get_dependency_container()
    
    services_info = {}
    for service_type in container._registry.list_services():
        registration = container._registry.get_registration_info(service_type)
        services_info[service_type.__name__] = {
            "interface": service_type.__name__,
            "implementation": registration.implementation.__name__ if registration.implementation else None,
            "scope": registration.scope.value if registration.scope else None,
            "dependencies": [dep.__name__ for dep in (registration.dependencies or [])],
            "metadata": registration.metadata
        }
    
    return services_info


def _check_job_management_service_health() -> bool:
    """Check if job management service is healthy."""
    try:
        from application.shared.services.job_management_service import job_management_service
        # Check if the service can return its status
        job_management_service.get_job_status("test_job_id")
        return True
    except Exception:
        return False


def _check_pipeline_config_service_health() -> bool:
    """Check if pipeline config service is healthy."""
    try:
        from application.pipeline_config.services.pipeline_config_service import pipeline_config_service
        # Check if the service can create a basic config
        pipeline_config_service.create_base_config()
        return True
    except Exception:
        return False


def _check_encoder_service_health() -> bool:
    """Check if encoder service is healthy."""
    try:
        from application.encoder.services.encoder_service import encoder_service
        # Check if the service can load vocabulary
        encoder_service.load_vocab("shared/conf/ReMIDICaps/midi/remi.yaml")
        return True
    except Exception:
        return False


def _check_feature_extraction_service_health() -> bool:
    """Check if feature extraction service is healthy."""
    try:
        from application.feature_extraction.services.feature_extraction_service import feature_extraction_service
        # Check if the service can initialize
        feature_extraction_service.initialize_vae_model({})
        return True
    except Exception:
        return False


def _check_generator_service_health() -> bool:
    """Check if generator service is healthy."""
    try:
        from application.generator.services.generator_service import generator_service
        # Check if the service can initialize
        generator_service.initialize_model({})
        return True
    except Exception:
        return False


def _check_multimodal_mapping_service_health() -> bool:
    """Check if multimodal mapping service is healthy."""
    try:
        from application.multimodal_mapping.services.multimodal_mapping_service import multimodal_mapping_service
        # Check if the service can initialize
        multimodal_mapping_service.initialize_model({})
        return True
    except Exception:
        return False


def _check_music_base_service_health() -> bool:
    """Check if music base service is healthy."""
    try:
        from application.music_base.services.music_base_service import music_base_service
        # Check if the service can process basic functions
        music_base_service.extract_chord_progression("test")
        return True
    except Exception:
        return False


def _check_dataloader_service_health() -> bool:
    """Check if dataloader service is healthy."""
    try:
        from application.dataloader.services.dataloader_service import dataloader_service
        # Check if the service can create a basic dataloader
        dataloader_service.create_dataloader({})
        return True
    except Exception:
        return False
