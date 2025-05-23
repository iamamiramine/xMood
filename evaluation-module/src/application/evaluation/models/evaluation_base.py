import muspy
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List


class EvaluationBase(ABC):
    """
    Base class for evaluation metrics.
    
    All metric classes should inherit from this base class and implement
    the evaluate method according to their specific evaluation logic.
    """
    
    def __init__(self, name: str, description: str, category: str, parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize the evaluation metric.
        
        Args:
            name: Name of the metric (e.g., "n_pitches_used")
            description: Description of what the metric measures
            category: Category of the metric (e.g., "pitch", "rhythm")
            parameters: Optional parameters required for the metric calculation
        """
        self.name = name
        self.description = description
        self.category = category
        self.parameters = parameters or {}
    
    @abstractmethod
    def evaluate(self, 
                 midi_path: str, 
                 reference_midi_path: Optional[str] = None, 
                 max_time: float = 0) -> Dict[str, Any]:
        """
        Evaluate a MIDI file using this metric.
        
        Args:
            midi_path: Path to the MIDI file to evaluate
            reference_midi_path: Optional path to a reference MIDI file (for comparison metrics)
            max_time: Maximum time in seconds to consider (0 for no limit)
            
        Returns:
            Dictionary containing the metric results
        """
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about this metric.
        
        Returns:
            Dictionary with metric metadata
        """
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "parameters": self.parameters
        }
    
    def __str__(self) -> str:
        """String representation of the metric."""
        return f"{self.name} ({self.category})"


class MusPyMixin:
    """
    Mixin class that provides MusPy-specific functionality.
    
    This should only be used by metrics that directly work with MusPy objects.
    """
    
    def load_midi(self, midi_path: str, max_time: float = 0) -> Optional[muspy.Music]:
        """
        Load a MIDI file into a MusPy Music object, with optional time truncation.
        
        Args:
            midi_path: Path to the MIDI file
            max_time: Maximum time in seconds to keep (0 for no limit)
            
        Returns:
            MusPy Music object or None if loading fails
        """
        try:
            music = muspy.read_midi(midi_path)
            
            # Trim the MIDI to specified maximum time if needed
            if max_time > 0 and music is not None:
                # Convert max_time from seconds to time steps (ticks)
                max_ticks = int(max_time * music.resolution * 4)  # Assuming 4 beats per second by default
                
                # Find all notes that start within the max time
                for track in music.tracks:
                    filtered_notes = []
                    for note in track.notes:
                        if note.time < max_ticks:
                            # If note extends beyond max_time, truncate it
                            if note.time + note.duration > max_ticks:
                                note.duration = max_ticks - note.time
                            filtered_notes.append(note)
                    track.notes = filtered_notes
                
                # Update the MIDI time information
                music.adjust_time(lambda time: min(time, max_ticks))
            
            return music
        except Exception as e:
            print(f"Failed to load MIDI file {midi_path}: {e}")
            return None 