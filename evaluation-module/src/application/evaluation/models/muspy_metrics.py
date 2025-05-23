import muspy
from typing import Dict, Any, Optional
import inspect
import numpy as np

from application.evaluation.models.evaluation_base import EvaluationBase, MusPyMixin


class MusPyMetric(EvaluationBase, MusPyMixin):
    """Base class for metrics implemented in the MusPy library."""
    
    def __init__(self, name: str, description: str, category: str, 
                parameters: Optional[Dict[str, Any]] = None,
                muspy_func = None):
        """
        Initialize a MusPy metric.
        
        Args:
            name: Name of the metric
            description: Description of what the metric measures
            category: Category of the metric
            parameters: Optional parameters required for the metric
            muspy_func: The MusPy function to call for calculation
        """
        super().__init__(name, description, category, parameters)
        self.muspy_func = muspy_func if muspy_func else getattr(muspy.metrics, name, None)
        
        if self.muspy_func is None:
            raise ValueError(f"MusPy function not found for metric: {name}")
    
    def evaluate(self, 
                midi_path: str, 
                reference_midi_path: Optional[str] = None, 
                max_time: float = 0) -> Dict[str, Any]:
        """
        Evaluate a MIDI file using a MusPy metric.
        
        Args:
            midi_path: Path to the MIDI file to evaluate
            reference_midi_path: Not used for basic MusPy metrics
            max_time: Maximum time in seconds to consider
            
        Returns:
            Dictionary containing the metric result
        """
        # Load the MIDI file
        music = self.load_midi(midi_path, max_time)
        
        if music is None:
            return {"error": f"Failed to load MIDI file: {midi_path}"}
        
        # Get the parameter names expected by the function
        sig = inspect.signature(self.muspy_func)
        expected_params = set(sig.parameters.keys())
        expected_params.discard('music')  # Remove the 'music' parameter
        
        # Filter the parameters to only include those expected by the function
        filtered_params = {k: v for k, v in self.parameters.items() if k in expected_params}
        
        try:
            # Call the MusPy function with the appropriate parameters
            result = self.muspy_func(music, **filtered_params)
            
            # Convert numpy types to Python native types if needed
            if hasattr(result, "item"):
                result = result.item()
                
            return {self.name: result}
        except Exception as e:
            return {"error": str(e)}
    
    def normalize_score(self, score: float, min_val: float = 0, max_val: float = 0, 
                         invert: bool = False, log_scale: bool = False) -> float:
        """
        Normalize a score to the range [0, 1].
        
        Args:
            score: The original metric score
            min_val: The minimum expected value (default: 0)
            max_val: The maximum expected value (default: 0, meaning no upper bound)
            invert: Whether to invert the normalized score (for metrics where lower is better)
            log_scale: Whether to use log scaling for the normalization
            
        Returns:
            Normalized score in the range [0, 1]
        """
        if np.isnan(score):
            return 0.0
            
        if max_val <= min_val and max_val != 0:
            return 0.0
            
        if max_val == 0:  # No upper bound specified
            if log_scale:
                # Log scaling for unbounded metrics
                if score <= 0:
                    return 0.0
                norm_score = np.tanh(np.log1p(score - min_val))
            else:
                # Simple normalization with saturation at high values
                norm_score = 1.0 - np.exp(-0.1 * max(0, score - min_val))
        else:
            # Linear scaling between min and max
            norm_score = max(0, min(1, (score - min_val) / (max_val - min_val)))
            
        # Invert if needed (for metrics where lower is better)
        return 1.0 - norm_score if invert else norm_score


class NPitchesUsed(MusPyMetric):
    """Metric for counting unique pitches used in a MIDI file."""
    
    def __init__(self):
        super().__init__(
            name="n_pitches_used",
            description="Number of unique pitches used",
            category="pitch",
            muspy_func=muspy.metrics.n_pitches_used
        )
    
    def evaluate(self, midi_path: str, reference_midi_path: Optional[str] = None, max_time: float = 0) -> Dict[str, Any]:
        """
        Evaluate and normalize the number of unique pitches.
        
        Higher values indicate more pitch variety, with reasonable music typically using 
        between 10-40 pitches. The normalized score approaches 1.0 as more unique pitches are used,
        with diminishing returns after about 30 pitches.
        """
        result = super().evaluate(midi_path, reference_midi_path, max_time)
        
        if self.name in result:
            original_score = result[self.name]
            # Normalize: 0 pitches = 0, 30+ pitches approaches 1
            normalized_score = self.normalize_score(original_score, min_val=0, log_scale=True)
            result[self.name] = normalized_score
            
        return result


class NPitchClassesUsed(MusPyMetric):
    """Metric for counting unique pitch classes used in a MIDI file."""
    
    def __init__(self):
        super().__init__(
            name="n_pitch_classes_used",
            description="Number of unique pitch classes used",
            category="pitch",
            muspy_func=muspy.metrics.n_pitch_classes_used
        )
    
    def evaluate(self, midi_path: str, reference_midi_path: Optional[str] = None, max_time: float = 0) -> Dict[str, Any]:
        """
        Evaluate and normalize the number of unique pitch classes.
        
        The maximum possible value is 12 (all pitch classes), with typical music using 7-8 (diatonic scale).
        A normalized score of 1.0 means all 12 pitch classes are used, while 0.0 means no pitches.
        Values around 0.6-0.7 typically represent diatonic scales.
        """
        result = super().evaluate(midi_path, reference_midi_path, max_time)
        
        if self.name in result:
            original_score = result[self.name]
            # Linear normalization: 0 = 0, 12 = 1 (max possible pitch classes)
            normalized_score = self.normalize_score(original_score, min_val=0, max_val=12)
            result[self.name] = normalized_score
            
        return result


class PitchRange(MusPyMetric):
    """Metric for calculating the pitch range in a MIDI file."""
    
    def __init__(self):
        super().__init__(
            name="pitch_range",
            description="Range between highest and lowest pitch",
            category="pitch",
            muspy_func=muspy.metrics.pitch_range
        )
    
    def evaluate(self, midi_path: str, reference_midi_path: Optional[str] = None, max_time: float = 0) -> Dict[str, Any]:
        """
        Evaluate and normalize the pitch range.
        
        The original score measures the distance between the highest and lowest pitch.
        Most music has a range of 24-60 semitones (2-5 octaves). The normalized score
        approaches 1.0 as the range increases, with diminishing returns after about 48 semitones (4 octaves).
        """
        result = super().evaluate(midi_path, reference_midi_path, max_time)
        
        if self.name in result:
            original_score = result[self.name]
            # Normalize with saturation: 0 = 0, 48+ (four octaves) approaches 1
            normalized_score = self.normalize_score(original_score, min_val=0, max_val=60)
            result[self.name] = normalized_score
            
        return result


class EmptyBeatRate(MusPyMetric):
    """Metric for calculating the ratio of empty beats in a MIDI file."""
    
    def __init__(self):
        super().__init__(
            name="empty_beat_rate",
            description="Ratio of empty beats",
            category="rhythm",
            muspy_func=muspy.metrics.empty_beat_rate
        )
    
    def evaluate(self, midi_path: str, reference_midi_path: Optional[str] = None, max_time: float = 0) -> Dict[str, Any]:
        """
        Evaluate and normalize the empty beat rate.
        
        The original score is between 0.0 (no empty beats) and 1.0 (all beats empty).
        Since lower is better for this metric, the normalization inverts the score.
        A normalized score of 1.0 means no empty beats (high note density),
        while 0.0 means all beats are empty (silence).
        """
        result = super().evaluate(midi_path, reference_midi_path, max_time)
        
        if self.name in result:
            original_score = result[self.name]
            # Invert because lower is better: 0 empty beats = 1, all empty = 0
            normalized_score = self.normalize_score(original_score, min_val=0, max_val=1, invert=True)
            result[self.name] = normalized_score
            
        return result


class EmptyMeasureRate(MusPyMetric):
    """Metric for calculating the ratio of empty measures in a MIDI file."""
    
    def __init__(self):
        super().__init__(
            name="empty_measure_rate",
            description="Ratio of empty measures",
            category="rhythm",
            parameters={"measure_resolution": 4},
            muspy_func=muspy.metrics.empty_measure_rate
        )
    
    def evaluate(self, midi_path: str, reference_midi_path: Optional[str] = None, max_time: float = 0) -> Dict[str, Any]:
        """
        Evaluate and normalize the empty measure rate.
        
        The original score is between 0.0 (no empty measures) and 1.0 (all measures empty).
        Since lower is better for this metric, the normalization inverts the score.
        A normalized score of 1.0 means no empty measures (good continuity),
        while 0.0 means all measures are empty (complete silence).
        """
        result = super().evaluate(midi_path, reference_midi_path, max_time)
        
        if self.name in result:
            original_score = result[self.name]
            # Invert because lower is better: 0 empty measures = 1, all empty = 0
            normalized_score = self.normalize_score(original_score, min_val=0, max_val=1, invert=True)
            result[self.name] = normalized_score
            
        return result


class Polyphony(MusPyMetric):
    """Metric for calculating the average polyphony in a MIDI file."""
    
    def __init__(self):
        super().__init__(
            name="polyphony",
            description="Average number of pitches played concurrently",
            category="texture",
            muspy_func=muspy.metrics.polyphony
        )
    
    def evaluate(self, midi_path: str, reference_midi_path: Optional[str] = None, max_time: float = 0) -> Dict[str, Any]:
        """
        Evaluate and normalize the average polyphony.
        
        The original score is the average number of concurrent notes, typically between 1-8.
        Higher values indicate richer textures. A normalized score approaches 1.0 as
        polyphony increases, with diminishing returns after about 4 notes.
        """
        result = super().evaluate(midi_path, reference_midi_path, max_time)
        
        if self.name in result:
            original_score = result[self.name]
            # Normalize with saturation: 1 = 0.25, 4+ approaches 1
            normalized_score = self.normalize_score(original_score, min_val=1, max_val=8)
            # Adjust the curve to make sure monophonic music (score=1) doesn't get 0
            normalized_score = max(0.25, normalized_score)
            result[self.name] = normalized_score
            
        return result


class PolyphonyRate(MusPyMetric):
    """Metric for calculating the polyphony rate in a MIDI file."""
    
    def __init__(self):
        super().__init__(
            name="polyphony_rate",
            description="Ratio of time steps where multiple pitches are on",
            category="texture",
            parameters={"threshold": 2},
            muspy_func=muspy.metrics.polyphony_rate
        )
    
    def evaluate(self, midi_path: str, reference_midi_path: Optional[str] = None, max_time: float = 0) -> Dict[str, Any]:
        """
        Evaluate and normalize the polyphony rate.
        
        The original score is between 0.0 (no polyphony) and 1.0 (always polyphonic).
        This directly represents the proportion of time with multiple notes playing.
        Since this is already in 0-1 range, we use it as is, with higher values
        indicating more complex, polyphonic textures.
        """
        result = super().evaluate(midi_path, reference_midi_path, max_time)
        
        if self.name in result:
            # Already in 0-1 range, higher is better
            pass
            
        return result


class ScaleConsistency(MusPyMetric):
    """Metric for calculating the scale consistency in a MIDI file."""
    
    def __init__(self):
        super().__init__(
            name="scale_consistency",
            description="Largest pitch-in-scale rate",
            category="harmony",
            muspy_func=muspy.metrics.scale_consistency
        )
    
    def evaluate(self, midi_path: str, reference_midi_path: Optional[str] = None, max_time: float = 0) -> Dict[str, Any]:
        """
        Evaluate the scale consistency (already normalized).
        
        The original score is between 0.0 (no consistent scale) and 1.0 (perfect scale adherence).
        Higher values indicate more tonal coherence, with typical tonal music around 0.7-0.9.
        This metric is already in 0-1 range, with higher values indicating better scale adherence.
        """
        result = super().evaluate(midi_path, reference_midi_path, max_time)
        
        if self.name in result:
            # Already in 0-1 range, higher is better
            pass
            
        return result


class DrumPatternConsistency(MusPyMetric):
    """Metric for calculating the drum pattern consistency in a MIDI file."""
    
    def __init__(self):
        super().__init__(
            name="drum_pattern_consistency",
            description="Largest drum-in-pattern rate",
            category="rhythm",
            muspy_func=muspy.metrics.drum_pattern_consistency
        )
    
    def evaluate(self, midi_path: str, reference_midi_path: Optional[str] = None, max_time: float = 0) -> Dict[str, Any]:
        """
        Evaluate the drum pattern consistency (already normalized).
        
        The original score is between 0.0 (no consistent drum pattern) and 1.0 (perfectly consistent).
        Higher values indicate more regular and consistent drum patterns.
        This metric is already in 0-1 range, with higher values indicating better pattern consistency.
        """
        result = super().evaluate(midi_path, reference_midi_path, max_time)
        
        if self.name in result:
            # Already in 0-1 range, higher is better
            pass
            
        return result


class PitchEntropy(MusPyMetric):
    """Metric for calculating the pitch entropy in a MIDI file."""
    
    def __init__(self):
        super().__init__(
            name="pitch_entropy",
            description="Entropy of pitch histogram",
            category="pitch",
            muspy_func=muspy.metrics.pitch_entropy
        )
    
    def evaluate(self, midi_path: str, reference_midi_path: Optional[str] = None, max_time: float = 0) -> Dict[str, Any]:
        """
        Evaluate and normalize the pitch entropy.
        
        The original entropy score has no fixed upper bound, with typical values between 1-6.
        Higher values indicate more variety in pitch usage. The normalized score approaches
        1.0 as entropy increases, with values around 0.5-0.7 for typical tonal music.
        """
        result = super().evaluate(midi_path, reference_midi_path, max_time)
        
        if self.name in result:
            original_score = result[self.name]
            # Normalize with log scaling: higher entropy approaches 1
            # Typical music entropy is around 2-5 bits
            normalized_score = self.normalize_score(original_score, min_val=0, log_scale=True)
            # Scale to make typical values around 0.6-0.8
            normalized_score = min(1.0, normalized_score * 1.2)
            result[self.name] = normalized_score
            
        return result


class PitchClassEntropy(MusPyMetric):
    """Metric for calculating the pitch class entropy in a MIDI file."""
    
    def __init__(self):
        super().__init__(
            name="pitch_class_entropy",
            description="Entropy of pitch class histogram",
            category="pitch",
            muspy_func=muspy.metrics.pitch_class_entropy
        )
    
    def evaluate(self, midi_path: str, reference_midi_path: Optional[str] = None, max_time: float = 0) -> Dict[str, Any]:
        """
        Evaluate and normalize the pitch class entropy.
        
        The original entropy score ranges from 0 to about log2(12) ≈ 3.58 for completely uniform distribution.
        Higher values indicate more variety in pitch class usage. The normalized score approaches
        1.0 as entropy increases, with values around 0.6-0.8 for typical tonal music.
        """
        result = super().evaluate(midi_path, reference_midi_path, max_time)
        
        if self.name in result:
            original_score = result[self.name]
            # Max theoretical entropy is log2(12) ≈ 3.58 for pitch classes
            max_entropy = np.log2(12)
            # Normalize linearly against max possible value
            normalized_score = self.normalize_score(original_score, min_val=0, max_val=max_entropy)
            result[self.name] = normalized_score
            
        return result


class GrooveConsistency(MusPyMetric):
    """Metric for calculating the groove consistency in a MIDI file."""
    
    def __init__(self):
        super().__init__(
            name="groove_consistency",
            description="Groove consistency between measures",
            category="rhythm",
            parameters={"measure_resolution": 4},
            muspy_func=muspy.metrics.groove_consistency
        )
    
    def evaluate(self, midi_path: str, reference_midi_path: Optional[str] = None, max_time: float = 0) -> Dict[str, Any]:
        """
        Evaluate the groove consistency (already normalized).
        
        The original score is between 0.0 (no consistency) and 1.0 (perfect consistency).
        Higher values indicate more consistent and recognizable rhythmic patterns across measures.
        This metric is already in 0-1 range, with higher values indicating better groove consistency.
        """
        result = super().evaluate(midi_path, reference_midi_path, max_time)
        
        if self.name in result:
            # Already in 0-1 range, higher is better
            pass
            
        return result


# Dictionary mapping metric names to their classes for easy instantiation
MUSPY_METRICS = {
    "n_pitches_used": NPitchesUsed,
    "n_pitch_classes_used": NPitchClassesUsed,
    "pitch_range": PitchRange,
    "empty_beat_rate": EmptyBeatRate,
    "empty_measure_rate": EmptyMeasureRate,
    "polyphony": Polyphony,
    "polyphony_rate": PolyphonyRate,
    "scale_consistency": ScaleConsistency,
    "drum_pattern_consistency": DrumPatternConsistency,
    "pitch_entropy": PitchEntropy,
    "pitch_class_entropy": PitchClassEntropy,
    "groove_consistency": GrooveConsistency
} 