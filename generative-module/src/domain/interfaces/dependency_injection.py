"""
Dependency Injection Container for PA-AI-2 Generative Module

This module provides a dependency injection container that manages service
lifetimes, dependency resolution, and service discovery. It builds on top
of the service registry to provide a more convenient API for dependency
management.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Type, TypeVar, List, Callable, Set
from contextlib import contextmanager
import logging
import inspect
from dataclasses import dataclass

from .service_registry import EnhancedServiceRegistry, ServiceScope, get_enhanced_service_registry
from .service_interfaces import (
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

# Import service adapters
from application.shared.adapters.config_adapter import ConfigServiceAdapter
from application.shared.adapters.job_management_adapter import JobManagementServiceAdapter
from application.shared.adapters.pipeline_config_adapter import PipelineConfigServiceAdapter
from application.shared.adapters.encoder_adapter import EncoderServiceAdapter
from application.shared.adapters.feature_extraction_adapter import FeatureExtractionServiceAdapter
from application.shared.adapters.generator_adapter import GeneratorServiceAdapter
from application.shared.adapters.multimodal_mapping_adapter import MultimodalMappingServiceAdapter
from application.shared.adapters.music_base_adapter import MusicBaseServiceAdapter
from application.shared.adapters.dataloader_adapter import DataloaderServiceAdapter

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ServiceRegistryAdapter:
    """
    Adapter to bridge DependencyContainer API with EnhancedServiceRegistry API.
    
    This adapter translates Type-based service keys to string-based service names
    and provides the interface that DependencyContainer expects.
    """
    
    def __init__(self, enhanced_registry: EnhancedServiceRegistry):
        self._enhanced_registry = enhanced_registry
        self._type_to_name: Dict[Type, str] = {}
        self._name_to_type: Dict[str, Type] = {}
    
    def register(self, service_type: Type, implementation: Any = None, 
                 scope: ServiceScope = ServiceScope.SINGLETON, 
                 dependencies: List[Type] = None) -> None:
        """Register a service with type-based key."""
        service_name = self._get_service_name(service_type)
        
        # Convert dependency types to names
        dependency_names = []
        if dependencies:
            dependency_names = [self._get_service_name(dep) for dep in dependencies]
        
        self._enhanced_registry.register_service(
            service_name=service_name,
            service_type=service_type,
            implementation=implementation,
            scope=scope,
            dependencies=dependency_names
        )
    
    def register_instance(self, service_type: Type, instance: Any) -> None:
        """Register a service instance."""
        service_name = self._get_service_name(service_type)
        self._enhanced_registry.register_service(
            service_name=service_name,
            service_type=service_type,
            implementation=instance,
            scope=ServiceScope.SINGLETON
        )
    
    def register_factory(self, service_type: Type, factory: Callable, 
                        scope: ServiceScope = ServiceScope.SINGLETON) -> None:
        """Register a service factory."""
        service_name = self._get_service_name(service_type)
        self._enhanced_registry.register_service(
            service_name=service_name,
            service_type=service_type,
            factory=factory,
            scope=scope
        )
    
    def get_service(self, service_type: Type) -> Any:
        """Get service by type."""
        service_name = self._get_service_name(service_type)
        return self._enhanced_registry.get_service(service_name)
    
    def is_registered(self, service_type: Type) -> bool:
        """Check if service is registered."""
        service_name = self._get_service_name(service_type)
        return service_name in self._enhanced_registry._services
    
    def get_registration_info(self, service_type: Type) -> Optional[Any]:
        """Get registration info for a service."""
        service_name = self._get_service_name(service_type)
        if service_name in self._enhanced_registry._services:
            registration = self._enhanced_registry._services[service_name]
            # Return an object with the expected attributes
            class RegistrationInfo:
                def __init__(self, reg):
                    self.scope = reg.scope
                    self.dependencies = [self._name_to_type.get(dep, dep) for dep in reg.dependencies]
                    self.service_type = reg.service_type
                    self.implementation = reg.implementation
                    self.factory = reg.factory
            return RegistrationInfo(registration)
        return None
    
    def _get_service_name(self, service_type: Type) -> str:
        """Convert service type to service name using standard naming convention."""
        if service_type not in self._type_to_name:
            # Map interface types to their standard service names
            type_to_name_mapping = {
                'IConfigService': 'config_service',
                'IJobManagementService': 'job_management_service',
                'IPipelineConfigService': 'pipeline_config_service',
                'IEncoderService': 'encoder_service',
                'IFeatureExtractionService': 'feature_extraction_service',
                'IGeneratorService': 'generator_service',
                'IMultimodalMappingService': 'multimodal_mapping_service',
                'IMusicBaseService': 'music_base_service',
                'IDataloaderService': 'dataloader_service'
            }
            
            type_name = service_type.__name__
            if type_name in type_to_name_mapping:
                service_name = type_to_name_mapping[type_name]
            else:
                # Fallback: convert IServiceName to service_name
                service_name = type_name.lower().replace('i', '', 1) if type_name.startswith('I') else type_name.lower()
                service_name = service_name + '_service' if not service_name.endswith('service') else service_name
            
            self._type_to_name[service_type] = service_name
            self._name_to_type[service_name] = service_type
        return self._type_to_name[service_type]


@dataclass
class DependencyResolution:
    """Information about dependency resolution."""
    service_type: Type
    instance: Any
    resolved_dependencies: List[Type]
    resolution_time: float
    scope: ServiceScope


class IDependencyContainer(ABC):
    """Interface for dependency injection container."""
    
    @abstractmethod
    def register(self, service_type: Type[T], implementation: Type[T], 
                 scope: ServiceScope = ServiceScope.SINGLETON) -> None:
        """Register a service with its implementation."""
        pass
    
    @abstractmethod
    def register_instance(self, service_type: Type[T], instance: T) -> None:
        """Register a service instance."""
        pass
    
    @abstractmethod
    def register_factory(self, service_type: Type[T], factory: Callable[[], T],
                        scope: ServiceScope = ServiceScope.SINGLETON) -> None:
        """Register a service factory."""
        pass
    
    @abstractmethod
    def resolve(self, service_type: Type[T]) -> T:
        """Resolve a service and its dependencies."""
        pass
    
    @abstractmethod
    def try_resolve(self, service_type: Type[T]) -> Optional[T]:
        """Try to resolve a service, returning None if not registered."""
        pass
    
    @abstractmethod
    def create_scope(self) -> 'IDependencyContainer':
        """Create a new dependency scope."""
        pass
    
    @abstractmethod
    def is_registered(self, service_type: Type) -> bool:
        """Check if a service is registered."""
        pass
    
    @abstractmethod
    def get_registration_info(self, service_type: Type) -> Optional[DependencyResolution]:
        """Get dependency resolution information."""
        pass


class DependencyContainer(IDependencyContainer):
    """
    Dependency injection container implementation.
    
    Features:
    - Automatic dependency resolution
    - Service lifetime management
    - Circular dependency detection
    - Service scoping
    - Performance tracking
    """
    
    def __init__(self, registry: Optional[EnhancedServiceRegistry] = None):
        self._enhanced_registry = registry or get_enhanced_service_registry()
        self._registry = ServiceRegistryAdapter(self._enhanced_registry)
        self._scoped_instances: Dict[Type, Any] = {}
        self._resolution_stack: Set[Type] = set()
        self._resolution_history: List[DependencyResolution] = []
        self._parent_container: Optional['DependencyContainer'] = None
    
    def register(self, service_type: Type[T], implementation: Type[T], 
                 scope: ServiceScope = ServiceScope.SINGLETON) -> None:
        """
        Register a service with its implementation.
        
        Args:
            service_type: The service interface type
            implementation: The concrete implementation type
            scope: Service scope (singleton, transient, scoped)
        """
        # Automatically detect dependencies from constructor
        dependencies = self._get_constructor_dependencies(implementation)
        
        self._registry.register(
            service_type=service_type,
            implementation=implementation,
            scope=scope,
            dependencies=dependencies
        )
        
        logger.info(f"Registered {service_type.__name__} with {len(dependencies)} dependencies")
    
    def register_instance(self, service_type: Type[T], instance: T) -> None:
        """
        Register a service instance.
        
        Args:
            service_type: The service interface type
            instance: The service instance
        """
        self._registry.register_instance(service_type, instance)
    
    def register_factory(self, service_type: Type[T], factory: Callable[[], T],
                        scope: ServiceScope = ServiceScope.SINGLETON) -> None:
        """
        Register a service factory.
        
        Args:
            service_type: The service interface type
            factory: Factory function that creates service instances
            scope: Service scope
        """
        self._registry.register_factory(service_type, factory, scope)
    
    def resolve(self, service_type: Type[T]) -> T:
        """
        Resolve a service and its dependencies.
        
        Args:
            service_type: The service interface type
            
        Returns:
            Service instance with resolved dependencies
            
        Raises:
            ValueError: If service is not registered or circular dependency detected
        """
        import time
        start_time = time.time()
        
        # Check for circular dependencies
        if service_type in self._resolution_stack:
            raise ValueError(f"Circular dependency detected for {service_type.__name__}")
        
        try:
            self._resolution_stack.add(service_type)
            
            # Check scoped instances first
            if service_type in self._scoped_instances:
                return self._scoped_instances[service_type]
            
            # Check parent container for scoped services
            if self._parent_container and service_type in self._parent_container._scoped_instances:
                return self._parent_container._scoped_instances[service_type]
            
            # Resolve from registry
            instance = self._registry.get_service(service_type)
            
            # Track scoped instances
            registration = self._registry.get_registration_info(service_type)
            if registration and registration.scope == ServiceScope.SCOPED:
                self._scoped_instances[service_type] = instance
            
            # Record resolution
            resolution_time = time.time() - start_time
            resolved_deps = self._get_resolved_dependencies(service_type)
            
            self._resolution_history.append(DependencyResolution(
                service_type=service_type,
                instance=instance,
                resolved_dependencies=resolved_deps,
                resolution_time=resolution_time,
                scope=registration.scope if registration else ServiceScope.SINGLETON
            ))
            
            return instance
            
        finally:
            self._resolution_stack.discard(service_type)
    
    def try_resolve(self, service_type: Type[T]) -> Optional[T]:
        """
        Try to resolve a service, returning None if not registered.
        
        Args:
            service_type: The service interface type
            
        Returns:
            Service instance or None if not registered
        """
        try:
            return self.resolve(service_type)
        except ValueError:
            return None
    
    def create_scope(self) -> 'DependencyContainer':
        """
        Create a new dependency scope.
        
        Returns:
            New scoped dependency container
        """
        scoped_container = DependencyContainer(self._enhanced_registry)
        scoped_container._parent_container = self
        return scoped_container
    
    def is_registered(self, service_type: Type) -> bool:
        """Check if a service is registered."""
        return self._registry.is_registered(service_type)
    
    def get_registration_info(self, service_type: Type) -> Optional[DependencyResolution]:
        """Get dependency resolution information."""
        for resolution in reversed(self._resolution_history):
            if resolution.service_type == service_type:
                return resolution
        return None
    
    def get_resolution_statistics(self) -> Dict[str, Any]:
        """Get dependency resolution statistics."""
        if not self._resolution_history:
            return {"total_resolutions": 0, "average_resolution_time": 0}
        
        total_time = sum(r.resolution_time for r in self._resolution_history)
        avg_time = total_time / len(self._resolution_history)
        
        service_counts = {}
        for resolution in self._resolution_history:
            service_name = resolution.service_type.__name__
            service_counts[service_name] = service_counts.get(service_name, 0) + 1
        
        return {
            "total_resolutions": len(self._resolution_history),
            "average_resolution_time": avg_time,
            "total_resolution_time": total_time,
            "service_resolution_counts": service_counts,
            "most_resolved_service": max(service_counts.items(), key=lambda x: x[1])[0] if service_counts else None
        }
    
    def _get_constructor_dependencies(self, implementation: Type) -> List[Type]:
        """Get constructor dependencies from type hints."""
        try:
            constructor = implementation.__init__
            signature = inspect.signature(constructor)
            dependencies = []
            
            for param_name, param in signature.parameters.items():
                if param_name == 'self':
                    continue
                
                if param.annotation != inspect.Parameter.empty:
                    # Check if annotation is a service interface
                    if self._is_service_interface(param.annotation):
                        dependencies.append(param.annotation)
            
            return dependencies
        except Exception as e:
            logger.warning(f"Failed to extract dependencies from {implementation.__name__}: {e}")
            return []
    
    def _is_service_interface(self, annotation: Type) -> bool:
        """Check if a type annotation is a service interface."""
        service_interfaces = [
            IEncoderService,
            IFeatureExtractionService,
            IGeneratorService,
            IMultimodalMappingService,
            IMusicBaseService,
            IDataloaderService,
            IConfigService,
            IJobManagementService,
            IPipelineConfigService,
        ]
        
        return annotation in service_interfaces or any(
            issubclass(annotation, interface) for interface in service_interfaces 
            if inspect.isclass(annotation)
        )
    
    def _get_resolved_dependencies(self, service_type: Type) -> List[Type]:
        """Get the dependencies that were resolved for a service."""
        registration = self._registry.get_registration_info(service_type)
        if registration and registration.dependencies:
            return registration.dependencies
        return []
    
    @contextmanager
    def scope(self):
        """Context manager for dependency scoping."""
        scoped_container = self.create_scope()
        try:
            yield scoped_container
        finally:
            scoped_container._scoped_instances.clear()


# Global dependency container instance
_global_container: Optional[DependencyContainer] = None


def get_dependency_container() -> DependencyContainer:
    """Get the global dependency container instance."""
    global _global_container
    if _global_container is None:
        _global_container = DependencyContainer()
    return _global_container


def initialize_dependency_container() -> DependencyContainer:
    """
    Initialize the dependency container.
    
    Note: Services are configured via service_initialization.py to avoid
    duplicate registrations between the dependency container and enhanced registry.
    """
    container = get_dependency_container()
    logger.info("Dependency container initialized - services will be resolved from enhanced registry")
    return container


def configure_services(container: DependencyContainer) -> None:
    """
    Configure services in the dependency container.
    
    Note: This function is now deprecated in favor of the centralized 
    service initialization in service_initialization.py to avoid
    duplicate registrations.
    """
    logger.info("Service configuration is now handled by service_initialization.py")
    logger.info("Dependency container will automatically resolve services from enhanced registry")


def inject_dependencies(func: Callable) -> Callable:
    """
    Decorator for automatic dependency injection.
    
    Args:
        func: Function to inject dependencies into
        
    Returns:
        Decorated function with injected dependencies
    """
    def wrapper(*args, **kwargs):
        container = get_dependency_container()
        signature = inspect.signature(func)
        
        # Resolve dependencies from type hints
        for param_name, param in signature.parameters.items():
            if param_name not in kwargs and param.annotation != inspect.Parameter.empty:
                if container.is_registered(param.annotation):
                    kwargs[param_name] = container.resolve(param.annotation)
        
        return func(*args, **kwargs)
    
    return wrapper 