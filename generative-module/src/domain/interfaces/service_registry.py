"""
Enhanced Service Registry

This module provides a comprehensive service registry pattern implementation with:
- Service discovery and registration
- Health monitoring and circuit breaker patterns
- Dynamic service management and versioning
- Load balancing and failover mechanisms
- Service metadata and documentation
- Performance monitoring and metrics collection
"""

from typing import Dict, Any, Optional, List, Type, Union, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
import logging
import traceback
from datetime import datetime, timedelta
from contextlib import contextmanager
from abc import ABC, abstractmethod
import inspect
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class ServiceScope(Enum):
    """Service instance scope enumeration."""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"


class ServiceStatus(Enum):
    """Service status enumeration."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"
    DISABLED = "disabled"


class CircuitBreakerState(Enum):
    """Circuit breaker state enumeration."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class ServiceMetrics:
    """Service performance metrics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    last_request_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    last_failure_time: Optional[datetime] = None
    
    def success_rate(self) -> float:
        """Calculate service success rate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    def failure_rate(self) -> float:
        """Calculate service failure rate."""
        return 1.0 - self.success_rate()


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    recovery_timeout: int = 60  # seconds
    request_timeout: int = 30  # seconds
    half_open_max_calls: int = 3


@dataclass
class ServiceRegistration:
    """Enhanced service registration with metadata and health monitoring."""
    service_name: str
    service_type: Type
    implementation: Optional[Any] = None
    factory: Optional[Callable] = None
    scope: ServiceScope = ServiceScope.SINGLETON
    version: str = "1.0.0"
    description: str = ""
    tags: Set[str] = field(default_factory=set)
    dependencies: List[str] = field(default_factory=list)
    
    # Health monitoring
    health_check: Optional[Callable] = None
    health_check_interval: int = 60  # seconds
    last_health_check: Optional[datetime] = None
    status: ServiceStatus = ServiceStatus.UNKNOWN
    
    # Circuit breaker
    circuit_breaker_enabled: bool = True
    circuit_breaker_config: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    circuit_breaker_state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    
    # Performance metrics
    metrics: ServiceMetrics = field(default_factory=ServiceMetrics)
    
    # Instance management
    instance: Optional[Any] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: Optional[datetime] = None


class ServiceDiscovery:
    """Service discovery interface."""
    
    def find_services_by_type(self, service_type: Type) -> List[ServiceRegistration]:
        """Find services by type."""
        pass
    
    def find_services_by_tag(self, tag: str) -> List[ServiceRegistration]:
        """Find services by tag."""
        pass
    
    def find_healthy_services(self) -> List[ServiceRegistration]:
        """Find all healthy services."""
        pass


class EnhancedServiceRegistry:
    """
    Enhanced service registry with comprehensive service management capabilities.
    
    Features:
    - Service discovery and registration
    - Health monitoring and circuit breaker patterns
    - Dynamic service management
    - Performance metrics collection
    - Load balancing and failover
    """
    
    def __init__(self):
        self._services: Dict[str, ServiceRegistration] = {}
        self._instances: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._health_monitor_thread: Optional[threading.Thread] = None
        self._health_monitor_running = False
        self._metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._event_listeners: Dict[str, List[Callable]] = defaultdict(list)
        
        # Performance tracking
        self._resolution_stats = {
            'total_resolutions': 0,
            'successful_resolutions': 0,
            'failed_resolutions': 0,
            'average_resolution_time': 0.0
        }
        
        logger.info("Enhanced service registry initialized")
    
    def register_service(self,
                        service_name: str,
                        service_type: Type,
                        implementation: Optional[Any] = None,
                        factory: Optional[Callable] = None,
                        scope: ServiceScope = ServiceScope.SINGLETON,
                        version: str = "1.0.0",
                        description: str = "",
                        tags: Optional[Set[str]] = None,
                        dependencies: Optional[List[str]] = None,
                        health_check: Optional[Callable] = None,
                        health_check_interval: int = 60,
                        circuit_breaker_enabled: bool = True,
                        circuit_breaker_config: Optional[CircuitBreakerConfig] = None) -> None:
        """
        Register a service with enhanced metadata and monitoring.
        
        Args:
            service_name: Unique service name
            service_type: Service interface type
            implementation: Service implementation instance
            factory: Factory function for creating instances
            scope: Service instance scope
            version: Service version
            description: Service description
            tags: Service tags for discovery
            dependencies: List of dependent service names
            health_check: Health check function
            health_check_interval: Health check interval in seconds
            circuit_breaker_enabled: Enable circuit breaker pattern
            circuit_breaker_config: Circuit breaker configuration
        """
        with self._lock:
            if service_name in self._services:
                logger.warning(f"Service {service_name} already registered, updating...")
            
            registration = ServiceRegistration(
                service_name=service_name,
                service_type=service_type,
                implementation=implementation,
                factory=factory,
                scope=scope,
                version=version,
                description=description,
                tags=tags or set(),
                dependencies=dependencies or [],
                health_check=health_check,
                health_check_interval=health_check_interval,
                circuit_breaker_enabled=circuit_breaker_enabled,
                circuit_breaker_config=circuit_breaker_config or CircuitBreakerConfig()
            )
            
            self._services[service_name] = registration
            
            # Start health monitoring if not already running
            if not self._health_monitor_running:
                self._start_health_monitoring()
            
            # Emit registration event
            self._emit_event("service_registered", service_name, registration)
            
            logger.info(f"Service {service_name} registered successfully")
    
    def get_service(self, service_name: str) -> Optional[Any]:
        """
        Get service instance with circuit breaker and metrics tracking.
        
        Args:
            service_name: Service name
            
        Returns:
            Service instance or None if not available
        """
        start_time = time.time()
        
        try:
            with self._lock:
                if service_name not in self._services:
                    self._resolution_stats['failed_resolutions'] += 1
                    return None
                
                registration = self._services[service_name]
                
                # Check circuit breaker
                if not self._is_circuit_breaker_closed(registration):
                    logger.warning(f"Circuit breaker open for service {service_name}")
                    self._resolution_stats['failed_resolutions'] += 1
                    return None
                
                # Get or create instance
                instance = self._get_or_create_instance(registration)
                
                if instance is not None:
                    # Update metrics
                    registration.metrics.total_requests += 1
                    registration.metrics.successful_requests += 1
                    registration.metrics.last_request_time = datetime.now()
                    registration.metrics.last_success_time = datetime.now()
                    registration.last_accessed = datetime.now()
                    
                    # Update resolution stats
                    self._resolution_stats['successful_resolutions'] += 1
                    resolution_time = time.time() - start_time
                    self._update_average_resolution_time(resolution_time)
                    
                    # Emit access event
                    self._emit_event("service_accessed", service_name, registration)
                    
                    return instance
                else:
                    # Mark as failure
                    self._record_service_failure(registration)
                    self._resolution_stats['failed_resolutions'] += 1
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting service {service_name}: {e}")
            if service_name in self._services:
                self._record_service_failure(self._services[service_name])
            self._resolution_stats['failed_resolutions'] += 1
            return None
        finally:
            self._resolution_stats['total_resolutions'] += 1
    
    def _get_or_create_instance(self, registration: ServiceRegistration) -> Optional[Any]:
        """Get or create service instance based on scope."""
        try:
            if registration.scope == ServiceScope.SINGLETON:
                if registration.instance is None:
                    if registration.implementation is not None:
                        registration.instance = registration.implementation
                    elif registration.factory is not None:
                        registration.instance = registration.factory()
                    else:
                        logger.error(f"No implementation or factory for singleton service {registration.service_name}")
                        return None
                return registration.instance
            
            elif registration.scope == ServiceScope.TRANSIENT:
                if registration.factory is not None:
                    return registration.factory()
                elif registration.implementation is not None:
                    # For transient, create new instance if implementation is a class
                    if inspect.isclass(registration.implementation):
                        return registration.implementation()
                    else:
                        return registration.implementation
                else:
                    logger.error(f"No factory or class implementation for transient service {registration.service_name}")
                    return None
            
            elif registration.scope == ServiceScope.SCOPED:
                # For scoped services, use thread-local storage
                thread_id = threading.get_ident()
                instance_key = f"{registration.service_name}_{thread_id}"
                
                if instance_key not in self._instances:
                    if registration.factory is not None:
                        self._instances[instance_key] = registration.factory()
                    elif registration.implementation is not None:
                        if inspect.isclass(registration.implementation):
                            self._instances[instance_key] = registration.implementation()
                        else:
                            self._instances[instance_key] = registration.implementation
                    else:
                        logger.error(f"No factory or implementation for scoped service {registration.service_name}")
                        return None
                
                return self._instances[instance_key]
                
        except Exception as e:
            logger.error(f"Error creating instance for service {registration.service_name}: {e}")
            return None
    
    def _is_circuit_breaker_closed(self, registration: ServiceRegistration) -> bool:
        """Check if circuit breaker is closed (service is available)."""
        if not registration.circuit_breaker_enabled:
            return True
        
        now = datetime.now()
        config = registration.circuit_breaker_config
        
        if registration.circuit_breaker_state == CircuitBreakerState.CLOSED:
            return True
        
        elif registration.circuit_breaker_state == CircuitBreakerState.OPEN:
            # Check if recovery timeout has passed
            if (registration.last_failure_time and 
                now - registration.last_failure_time > timedelta(seconds=config.recovery_timeout)):
                registration.circuit_breaker_state = CircuitBreakerState.HALF_OPEN
                registration.failure_count = 0
                logger.info(f"Circuit breaker for {registration.service_name} moved to HALF_OPEN")
                return True
            return False
        
        elif registration.circuit_breaker_state == CircuitBreakerState.HALF_OPEN:
            # Allow limited requests in half-open state
            return registration.failure_count < config.half_open_max_calls
        
        return False
    
    def _record_service_failure(self, registration: ServiceRegistration) -> None:
        """Record service failure and update circuit breaker state."""
        now = datetime.now()
        registration.failure_count += 1
        registration.last_failure_time = now
        registration.metrics.failed_requests += 1
        registration.metrics.last_failure_time = now
        
        if registration.circuit_breaker_enabled:
            config = registration.circuit_breaker_config
            
            if registration.circuit_breaker_state == CircuitBreakerState.HALF_OPEN:
                # Move back to OPEN on failure in half-open state
                registration.circuit_breaker_state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker for {registration.service_name} moved to OPEN (failure in half-open)")
            
            elif (registration.circuit_breaker_state == CircuitBreakerState.CLOSED and 
                  registration.failure_count >= config.failure_threshold):
                # Move to OPEN if failure threshold exceeded
                registration.circuit_breaker_state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker for {registration.service_name} moved to OPEN (threshold exceeded)")
        
        # Emit failure event
        self._emit_event("service_failure", registration.service_name, registration)
    
    def discover_services(self, **criteria) -> List[ServiceRegistration]:
        """
        Discover services based on criteria.
        
        Args:
            **criteria: Discovery criteria (type, tag, status, etc.)
            
        Returns:
            List of matching service registrations
        """
        with self._lock:
            services = list(self._services.values())
            
            # Filter by type
            if 'type' in criteria:
                services = [s for s in services if s.service_type == criteria['type']]
            
            # Filter by tag
            if 'tag' in criteria:
                services = [s for s in services if criteria['tag'] in s.tags]
            
            # Filter by status
            if 'status' in criteria:
                services = [s for s in services if s.status == criteria['status']]
            
            # Filter by healthy services
            if criteria.get('healthy_only', False):
                services = [s for s in services if s.status == ServiceStatus.HEALTHY]
            
            return services
    
    def _start_health_monitoring(self) -> None:
        """Start background health monitoring."""
        if self._health_monitor_running:
            return
        
        self._health_monitor_running = True
        self._health_monitor_thread = threading.Thread(target=self._health_monitor_loop, daemon=True)
        self._health_monitor_thread.start()
        logger.info("Health monitoring started")
    
    def _health_monitor_loop(self) -> None:
        """Background health monitoring loop."""
        while self._health_monitor_running:
            try:
                with self._lock:
                    for service_name, registration in self._services.items():
                        if registration.health_check is not None:
                            self._check_service_health(registration)
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in health monitor loop: {e}")
                time.sleep(10)  # Wait longer on error
    
    def _check_service_health(self, registration: ServiceRegistration) -> None:
        """Check service health."""
        now = datetime.now()
        
        # Check if it's time for health check
        if (registration.last_health_check and 
            now - registration.last_health_check < timedelta(seconds=registration.health_check_interval)):
            return
        
        try:
            # Call health check function
            is_healthy = registration.health_check()
            
            # Update status
            previous_status = registration.status
            registration.status = ServiceStatus.HEALTHY if is_healthy else ServiceStatus.UNHEALTHY
            registration.last_health_check = now
            
            # Emit status change event
            if previous_status != registration.status:
                self._emit_event("service_status_changed", registration.service_name, registration)
                logger.info(f"Service {registration.service_name} status changed: {previous_status} -> {registration.status}")
            
            # Reset failure count if healthy
            if is_healthy and registration.circuit_breaker_state == CircuitBreakerState.HALF_OPEN:
                registration.circuit_breaker_state = CircuitBreakerState.CLOSED
                registration.failure_count = 0
                logger.info(f"Circuit breaker for {registration.service_name} moved to CLOSED (health restored)")
                
        except Exception as e:
            logger.error(f"Health check failed for service {registration.service_name}: {e}")
            registration.status = ServiceStatus.UNHEALTHY
            registration.last_health_check = now
    
    def _emit_event(self, event_type: str, service_name: str, registration: ServiceRegistration) -> None:
        """Emit service event to listeners."""
        if event_type in self._event_listeners:
            for listener in self._event_listeners[event_type]:
                try:
                    listener(service_name, registration)
                except Exception as e:
                    logger.error(f"Error calling event listener: {e}")
    
    def add_event_listener(self, event_type: str, listener: Callable) -> None:
        """Add event listener for service events."""
        self._event_listeners[event_type].append(listener)
    
    def _update_average_resolution_time(self, resolution_time: float) -> None:
        """Update average resolution time."""
        current_avg = self._resolution_stats['average_resolution_time']
        total_resolutions = self._resolution_stats['total_resolutions']
        
        if total_resolutions == 0:
            self._resolution_stats['average_resolution_time'] = resolution_time
        else:
            self._resolution_stats['average_resolution_time'] = (
                (current_avg * (total_resolutions - 1) + resolution_time) / total_resolutions
            )
    
    def get_service_info(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed service information."""
        with self._lock:
            if service_name not in self._services:
                return None
            
            registration = self._services[service_name]
            return {
                'name': registration.service_name,
                'type': registration.service_type.__name__,
                'version': registration.version,
                'description': registration.description,
                'tags': list(registration.tags),
                'dependencies': registration.dependencies,
                'scope': registration.scope.value,
                'status': registration.status.value,
                'circuit_breaker_state': registration.circuit_breaker_state.value,
                'failure_count': registration.failure_count,
                'metrics': {
                    'total_requests': registration.metrics.total_requests,
                    'successful_requests': registration.metrics.successful_requests,
                    'failed_requests': registration.metrics.failed_requests,
                    'success_rate': registration.metrics.success_rate(),
                    'average_response_time': registration.metrics.average_response_time,
                    'last_request_time': registration.metrics.last_request_time.isoformat() if registration.metrics.last_request_time else None,
                    'last_success_time': registration.metrics.last_success_time.isoformat() if registration.metrics.last_success_time else None,
                    'last_failure_time': registration.metrics.last_failure_time.isoformat() if registration.metrics.last_failure_time else None
                },
                'created_at': registration.created_at.isoformat(),
                'last_accessed': registration.last_accessed.isoformat() if registration.last_accessed else None,
                'last_health_check': registration.last_health_check.isoformat() if registration.last_health_check else None
            }
    
    def get_all_services_info(self) -> List[Dict[str, Any]]:
        """Get information about all registered services."""
        with self._lock:
            return [self.get_service_info(name) for name in self._services.keys()]
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        with self._lock:
            return {
                'total_services': len(self._services),
                'healthy_services': len([s for s in self._services.values() if s.status == ServiceStatus.HEALTHY]),
                'unhealthy_services': len([s for s in self._services.values() if s.status == ServiceStatus.UNHEALTHY]),
                'resolution_stats': self._resolution_stats.copy(),
                'active_instances': len(self._instances)
            }
    
    def unregister_service(self, service_name: str) -> bool:
        """Unregister a service."""
        with self._lock:
            if service_name in self._services:
                registration = self._services.pop(service_name)
                
                # Clean up instances
                keys_to_remove = [key for key in self._instances.keys() if key.startswith(service_name)]
                for key in keys_to_remove:
                    del self._instances[key]
                
                # Emit unregistration event
                self._emit_event("service_unregistered", service_name, registration)
                
                logger.info(f"Service {service_name} unregistered")
                return True
            return False
    
    def clear(self) -> None:
        """Clear all services and stop monitoring."""
        with self._lock:
            self._health_monitor_running = False
            self._services.clear()
            self._instances.clear()
            self._metrics_history.clear()
            self._event_listeners.clear()
            
            # Reset stats
            self._resolution_stats = {
                'total_resolutions': 0,
                'successful_resolutions': 0,
                'failed_resolutions': 0,
                'average_resolution_time': 0.0
            }
            
            logger.info("Service registry cleared")


# Global enhanced service registry instance
_enhanced_service_registry: Optional[EnhancedServiceRegistry] = None


def get_enhanced_service_registry() -> EnhancedServiceRegistry:
    """Get the global enhanced service registry instance."""
    global _enhanced_service_registry
    if _enhanced_service_registry is None:
        _enhanced_service_registry = EnhancedServiceRegistry()
    return _enhanced_service_registry


# Backward compatibility - keep existing simple registry
@dataclass
class ServiceRegistration:
    """Simple service registration for backward compatibility."""
    service_type: Type
    implementation: Optional[Any] = None
    factory: Optional[Callable] = None
    scope: ServiceScope = ServiceScope.SINGLETON
    priority: int = 0
    instance: Optional[Any] = None


class ServiceRegistry:
    """Simple service registry for backward compatibility."""
    
    def __init__(self):
        self._services: Dict[str, ServiceRegistration] = {}
        self._instances: Dict[str, Any] = {}
        self._lock = threading.RLock()
    
    def register(self, service_name: str, service_type: Type, implementation: Optional[Any] = None, 
                factory: Optional[Callable] = None, scope: ServiceScope = ServiceScope.SINGLETON) -> None:
        """Register a service."""
        with self._lock:
            self._services[service_name] = ServiceRegistration(
                service_type=service_type,
                implementation=implementation,
                factory=factory,
                scope=scope
            )
    
    def get(self, service_name: str) -> Optional[Any]:
        """Get a service instance."""
        with self._lock:
            if service_name not in self._services:
                return None
            
            registration = self._services[service_name]
            
            if registration.scope == ServiceScope.SINGLETON:
                if registration.instance is None:
                    if registration.implementation is not None:
                        registration.instance = registration.implementation
                    elif registration.factory is not None:
                        registration.instance = registration.factory()
                return registration.instance
            
            elif registration.scope == ServiceScope.TRANSIENT:
                if registration.factory is not None:
                    return registration.factory()
                elif registration.implementation is not None:
                    if inspect.isclass(registration.implementation):
                        return registration.implementation()
                    else:
                        return registration.implementation
            
            return None
    
    def clear(self) -> None:
        """Clear all services."""
        with self._lock:
            self._services.clear()
            self._instances.clear()


# Global simple service registry instance
_service_registry: Optional[ServiceRegistry] = None


def get_service_registry() -> ServiceRegistry:
    """Get the global service registry instance."""
    global _service_registry
    if _service_registry is None:
        _service_registry = ServiceRegistry()
    return _service_registry 