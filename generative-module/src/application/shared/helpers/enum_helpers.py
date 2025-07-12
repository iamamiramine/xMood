"""
Helper utilities for working with enums and enum-like data structures.
"""
from typing import Dict, Any
from application.shared.services.job_management_service import JobPriority


def parse_job_priority(priority_str: str) -> JobPriority:
    """
    Parse job priority string to JobPriority enum.
    
    Args:
        priority_str: String representation of job priority
        
    Returns:
        JobPriority enum value
        
    Raises:
        ValueError: If priority string is not valid
    """
    priority_map = {
        "low": JobPriority.LOW,
        "normal": JobPriority.NORMAL,
        "high": JobPriority.HIGH,
        "urgent": JobPriority.URGENT
    }
    
    if priority_str.lower() not in priority_map:
        raise ValueError(f"Invalid priority: {priority_str}. Valid values: {list(priority_map.keys())}")
    
    return priority_map[priority_str.lower()]


def validate_parameters_defaults(parameters: Any, defaults: Dict[str, Any]) -> None:
    """
    Apply default values to parameters if they are missing or None.
    
    Args:
        parameters: Parameter object to validate and set defaults
        defaults: Dictionary of default values to apply
    """
    for key, default_value in defaults.items():
        if not hasattr(parameters, key) or getattr(parameters, key) is None:
            setattr(parameters, key, default_value) 