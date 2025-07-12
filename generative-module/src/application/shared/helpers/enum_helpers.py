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
