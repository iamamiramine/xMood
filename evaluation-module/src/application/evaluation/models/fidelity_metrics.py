import numpy as np
import scipy.stats as stats
import muspy
import math
import os
import pickle
from typing import Dict, Any, Optional, List, Tuple
from collections import Counter
import re

from application.evaluation.models.evaluation_base import EvaluationBase
from domain.constants.paths_constants import PROCESSED_PATH


class FidelityMetric(EvaluationBase):
    """Base class for fidelity metrics that compare two MIDI files."""
    
    def __init__(self, name: str, description: str, category: str, 
                parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize a fidelity metric.
        
        Args:
            name: Name of the metric
            description: Description of what the metric measures
            category: Category of the metric
            parameters: Optional parameters required for the metric
        """
        super().__init__(name, description, category, parameters)
        # Define token constants based on the sample format
        self.BAR_KEY = "Bar"
        self.PITCH_KEY = "Pitch"
        self.DURATION_KEY = "Duration"
        self.VELOCITY_KEY = "Velocity"  
        self.TIME_SIGNATURE_KEY = "Time Signature"
        self.KEY_SIGNATURE_KEY = "Key Signature"
        self.INSTRUMENT_KEY = "Instrument"
        self.CHORD_KEY = "Chord"
        self.NOTE_DENSITY_KEY = "Note Density"
        self.MEAN_VELOCITY_KEY = "Mean Velocity"
        self.MEAN_PITCH_KEY = "Mean Pitch"
        self.MEAN_DURATION_KEY = "Mean Duration"
        self.POSITION_KEY = "Position"
    
    def evaluate(self, 
                midi_path: str, 
                reference_midi_path: Optional[str] = None, 
                max_time: float = 0) -> Dict[str, Any]:
        """
        Evaluate a MIDI file against a reference using this fidelity metric.
        
        Args:
            midi_path: Path to the MIDI file to evaluate
            reference_midi_path: Path to the reference MIDI file (required)
            max_time: Maximum time in seconds to consider
            
        Returns:
            Dictionary containing the metric result
        """
        if reference_midi_path is None:
            return {"error": "Reference MIDI file is required for fidelity metrics"}
        
        try:
            # Load processed data for both files
            generated_data = self._load_processed_data(midi_path)
            reference_data = self._load_processed_data(reference_midi_path)
            
            if generated_data is None:
                return {"error": f"Failed to load processed data for generated MIDI file: {midi_path}"}
            
            if reference_data is None:
                return {"error": f"Failed to load processed data for reference MIDI file: {reference_midi_path}"}
                
            # Call the specific implementation in the subclass
            result = self._calculate(reference_data, generated_data)
            return result
        except Exception as e:
            return {"error": str(e)}
    
    def _load_processed_data(self, midi_path: str) -> Optional[Dict[str, Any]]:
        """
        Load processed data for a MIDI file.
        
        Args:
            midi_path: Path to the MIDI file
            
        Returns:
            Dictionary containing processed data or None if loading fails
        """
        try:
            # Get dataset name from parameters or use default
            dataset_name = self.parameters.get("dataset_name", "ReMIDICaps")
            
            # Get filename from path
            midi_filename = os.path.basename(midi_path)
            
            # Construct path to processed data
            processed_file = os.path.join(
                str(PROCESSED_PATH),
                dataset_name,
                f"{midi_filename}_processed.pkl"
            )
            
            # Check if processed file exists
            if not os.path.exists(processed_file):
                print(f"Processed file not found: {processed_file}")
                # Check what files are available in the directory
                dir_path = os.path.join(str(PROCESSED_PATH), dataset_name)
                if os.path.exists(dir_path):
                    print(f"Available files in {dir_path}: {os.listdir(dir_path)}")
                return None
            
            # Load processed data
            processed_data = pickle.load(open(processed_file, "rb"))
            
            return processed_data
            
        except Exception as e:
            print(f"Failed to load processed data for {midi_path}: {e}")
            return None
    
    def _token_matches_prefix(self, token: str, prefix: str) -> bool:
        """
        Check if a token matches a prefix, handling spaces and case-insensitivity.
        
        Args:
            token: The token to check
            prefix: The prefix to match
            
        Returns:
            True if the token starts with the prefix, False otherwise
        """
        # Remove any spaces in prefix for comparison
        clean_prefix = prefix.replace(" ", "")
        # Make case-insensitive comparison
        return token.replace(" ", "").lower().startswith(clean_prefix.lower())
    
    def _extract_value_from_token(self, token: str, prefix: str) -> Any:
        """
        Extract a value from a token, handling different separator formats.
        
        Args:
            token: The token to extract from
            prefix: The prefix to match
            
        Returns:
            The extracted value or None if not found
        """
        try:
            # Try different separator formats
            if "_" in token:
                parts = token.split("_")
                # Handle format like "Mean Pitch_60"
                if len(parts) >= 2:
                    value_part = parts[-1]
                    
                    # Handle formats like "C:min" in "Key Signature_C:min"
                    if ":" in value_part:
                        return value_part  # Return as is for complex values
                    
                    # Try to convert to int/float
                    try:
                        return int(value_part)
                    except ValueError:
                        try:
                            return float(value_part)
                        except ValueError:
                            return value_part
            
            # For more complex parsing cases
            if self._token_matches_prefix(token, prefix):
                # Extract everything after the prefix and a separator
                match = re.search(f"{prefix.replace(' ', '').lower()}_(.+)", token.replace(" ", "").lower())
                if match:
                    value_part = match.group(1)
                    # Try to convert to int/float
                    try:
                        return int(value_part)
                    except ValueError:
                        try:
                            return float(value_part)
                        except ValueError:
                            return value_part
        except Exception as e:
            print(f"Error extracting value from token {token}: {e}")
        
        return None
    
    def _calculate(self, reference_data: Dict[str, Any], generated_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate the fidelity metric between reference and generated data.
        
        This method should be implemented by subclasses to provide specific
        calculation logic for each fidelity metric.
        
        Args:
            reference_data: Processed data for reference MIDI
            generated_data: Processed data for generated MIDI
            
        Returns:
            Dictionary containing the metric result
        """
        raise NotImplementedError("Subclasses must implement _calculate method")
    
    def normalize_score(self, score: float, min_val: float = 0, max_val: float = 1, 
                       invert: bool = False, log_scale: bool = False) -> float:
        """
        Normalize a score to the range [0, 1].
        
        Args:
            score: The original metric score
            min_val: The minimum expected value (default: 0)
            max_val: The maximum expected value (default: 1)
            invert: Whether to invert the normalized score (for metrics where lower is better)
            log_scale: Whether to use log scaling for the normalization
            
        Returns:
            Normalized score in the range [0, 1]
        """
        if np.isnan(score):
            return 0.0
            
        if max_val <= min_val:
            return 0.0
            
        if log_scale:
            # Log scaling for unbounded metrics
            if score <= 0:
                return 0.0
            norm_score = np.tanh(np.log1p(score - min_val))
        else:
            # Linear scaling between min and max
            norm_score = max(0, min(1, (score - min_val) / (max_val - min_val)))
            
        # Invert if needed (for metrics where lower is better)
        return 1.0 - norm_score if invert else norm_score


class MacroOverlappingArea(FidelityMetric):
    """
    Metric for calculating the Macro Overlapping Area (MOA) between two musical sequences.
    MOA measures similarity while preserving temporal order.
    """
    
    def __init__(self):
        super().__init__(
            name="macro_overlapping_area",
            description="Measures similarity between two sequences while preserving temporal order",
            category="similarity",
            parameters={"max_bars": 16, "dataset_name": "ReMIDICaps"}
        )
    
    def _calculate(self, reference_data: Dict[str, Any], generated_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate the Macro Overlapping Area between reference and generated data."""
        # Get the max bars parameter
        max_bars = self.parameters.get("max_bars", 16)
        
        # Extract features from both pieces
        reference_features = self._extract_features_from_processed(reference_data)
        generated_features = self._extract_features_from_processed(generated_data)
        
        # Calculate the MOA
        feature_moas = self._macro_overlapping_area(reference_features, generated_features, max_bars)
        
        # Return the result in a standardized format
        return {
            self.name: {
                "overall": round(feature_moas["overall"], 4),
                "pitch": round(feature_moas["pitch"], 4),
                "duration": round(feature_moas["duration"], 4),
                "velocity": round(feature_moas["velocity"], 4)
            }
        }
    
    def _extract_features_from_processed(self, processed_data: Dict[str, Any]) -> List[Dict[str, List[float]]]:
        """
        Extract musical features from processed data, organized by bars.
        
        Args:
            processed_data: Dictionary of processed MIDI data
            
        Returns:
            List of features by bar
        """
        # Initialize result structure
        features_by_bar = []
        
        # Extract features from symbolic data if available
        if "symbolic_features" in processed_data and "bar_symbolic" in processed_data["symbolic_features"]:
            bar_symbolic = processed_data["symbolic_features"]["bar_symbolic"]
            
            # Find bar boundaries
            bar_indices = [i for i, token in enumerate(bar_symbolic) if self._token_matches_prefix(token, self.BAR_KEY)]
            bar_indices.append(len(bar_symbolic))  # Add end index
            
            # Extract features bar by bar
            for i in range(len(bar_indices) - 1):
                start_idx = bar_indices[i]
                end_idx = bar_indices[i + 1]
                
                # Skip empty bars
                if end_idx - start_idx < 3:
                    continue
                
                # Get tokens for this bar
                bar_tokens = bar_symbolic[start_idx:end_idx]
                
                # Extract pitch, duration, velocity features
                pitches = []
                durations = []
                velocities = []
                
                for token in bar_tokens:
                    # For now, use "Mean Pitch", "Mean Duration", and "Mean Velocity" since the sample doesn't have note-level tokens
                    if self._token_matches_prefix(token, self.MEAN_PITCH_KEY):
                        pitch_value = self._extract_value_from_token(token, self.MEAN_PITCH_KEY)
                        if pitch_value is not None:
                            pitches.append(pitch_value)
                    elif self._token_matches_prefix(token, self.MEAN_DURATION_KEY):
                        duration_value = self._extract_value_from_token(token, self.MEAN_DURATION_KEY)
                        if duration_value is not None:
                            durations.append(duration_value)
                    elif self._token_matches_prefix(token, self.MEAN_VELOCITY_KEY):
                        velocity_value = self._extract_value_from_token(token, self.MEAN_VELOCITY_KEY)
                        if velocity_value is not None:
                            velocities.append(velocity_value)
                
                # Add features for this bar
                features_by_bar.append({
                    'pitch': pitches if pitches else [0],  # Default to [0] if empty
                    'duration': durations if durations else [0],
                    'velocity': velocities if velocities else [0]
                })
        
        # Fallback to event-based features if symbolic not available
        elif "encodings" in processed_data and "events" in processed_data["encodings"]:
            events = processed_data["encodings"]["events"]
            
            # Find bar boundaries
            bar_indices = [i for i, e in enumerate(events) if self._token_matches_prefix(e, self.BAR_KEY)]
            
            if not bar_indices:
                # If no bar tokens, create artificial segments (every 20 events)
                segments = [events[i:i+20] for i in range(0, len(events), 20)]
                
                for segment in segments:
                    # Extract features from this segment
                    pitches = []
                    durations = []
                    velocities = []
                    
                    for event in segment:
                        if self._token_matches_prefix(event, self.MEAN_PITCH_KEY):
                            pitch_value = self._extract_value_from_token(event, self.MEAN_PITCH_KEY)
                            if pitch_value is not None:
                                pitches.append(pitch_value)
                        elif self._token_matches_prefix(event, self.MEAN_DURATION_KEY):
                            duration_value = self._extract_value_from_token(event, self.MEAN_DURATION_KEY)
                            if duration_value is not None:
                                durations.append(duration_value)
                        elif self._token_matches_prefix(event, self.MEAN_VELOCITY_KEY):
                            velocity_value = self._extract_value_from_token(event, self.MEAN_VELOCITY_KEY)
                            if velocity_value is not None:
                                velocities.append(velocity_value)
                    
                    # Add features for this segment
                    features_by_bar.append({
                        'pitch': pitches if pitches else [0],
                        'duration': durations if durations else [0],
                        'velocity': velocities if velocities else [0]
                    })
            else:
                # Create bar segments using bar tokens
                bar_indices = [0] + bar_indices + [len(events)]
                segments = [events[bar_indices[i]:bar_indices[i+1]] for i in range(len(bar_indices)-1)]
                
                for segment in segments:
                    # Extract features from this segment
                    pitches = []
                    durations = []
                    velocities = []
                    
                    for event in segment:
                        if self._token_matches_prefix(event, self.MEAN_PITCH_KEY):
                            pitch_value = self._extract_value_from_token(event, self.MEAN_PITCH_KEY)
                            if pitch_value is not None:
                                pitches.append(pitch_value)
                        elif self._token_matches_prefix(event, self.MEAN_DURATION_KEY):
                            duration_value = self._extract_value_from_token(event, self.MEAN_DURATION_KEY)
                            if duration_value is not None:
                                durations.append(duration_value)
                        elif self._token_matches_prefix(event, self.MEAN_VELOCITY_KEY):
                            velocity_value = self._extract_value_from_token(event, self.MEAN_VELOCITY_KEY)
                            if velocity_value is not None:
                                velocities.append(velocity_value)
                    
                    # Add features for this segment
                    features_by_bar.append({
                        'pitch': pitches if pitches else [0],
                        'duration': durations if durations else [0],
                        'velocity': velocities if velocities else [0]
                    })
        
        # If we have no features, create a placeholder with zeros
        if not features_by_bar:
            features_by_bar = [{'pitch': [0], 'duration': [0], 'velocity': [0]}]
            
        return features_by_bar
    
    def _macro_overlapping_area(self, features1: List[Dict[str, List[float]]], 
                              features2: List[Dict[str, List[float]]], 
                              max_bars: int = 16) -> Dict[str, float]:
        """
        Calculate the Macro Overlapping Area (MOA) between two sequences of feature dictionaries.
        
        Args:
            features1: First sequence of features by bar
            features2: Second sequence of features by bar
            max_bars: Maximum number of bars to compare
            
        Returns:
            Dict mapping feature names to their MOA values
        """
        # Initialize results
        feature_moas = {
            'pitch': 0,
            'duration': 0,
            'velocity': 0
        }
        
        # Limit number of bars to compare
        n_bars = min(len(features1), len(features2), max_bars)
        
        if n_bars == 0:
            return {'pitch': 0, 'duration': 0, 'velocity': 0, 'overall': 0}
        
        bar_count = 0
        
        # Calculate MOA for each bar
        for i in range(n_bars):
            if i >= len(features1) or i >= len(features2):
                break
                
            bar1_features = features1[i]
            bar2_features = features2[i]
            
            # Skip empty bars
            if not bar1_features['pitch'] or not bar2_features['pitch']:
                continue
            
            # Calculate MOA for each feature
            for feature in ['pitch', 'duration', 'velocity']:
                # Extract feature values
                values1 = bar1_features[feature]
                values2 = bar2_features[feature]
                
                # Skip if either list is empty
                if not values1 or not values2:
                    continue
                
                # Calculate distributions
                min_val = min(min(values1), min(values2))
                max_val = max(max(values1), max(values2))
                
                # Use appropriate number of bins
                if feature == 'pitch':
                    bins = 128  # MIDI pitch range
                elif feature == 'duration':
                    # Use logarithmic bins for duration
                    bins = 10
                else:  # velocity
                    bins = 128  # MIDI velocity range
                
                # Create histograms
                hist1, edges = np.histogram(values1, bins=bins, range=(min_val, max_val), density=True)
                hist2, _ = np.histogram(values2, bins=bins, range=(min_val, max_val), density=True)
                
                # Calculate overlapping area
                overlap = np.sum(np.minimum(hist1, hist2)) * (edges[1] - edges[0])
                feature_moas[feature] += overlap
            
            bar_count += 1
        
        # Average MOA over all bars
        if bar_count > 0:
            feature_moas = {feature: moa / bar_count for feature, moa in feature_moas.items()}
        
        # Calculate overall MOA as average of feature MOAs
        feature_moas['overall'] = sum(feature_moas.values()) / len(feature_moas)
        
        return feature_moas


class F1Score(FidelityMetric):
    """
    Metric for calculating the F1 score between two MIDI files.
    F1 score is used for multi-label categorical features like instruments.
    """
    
    def __init__(self):
        super().__init__(
            name="f1_score",
            description="F1 score for multi-label categorical features like instruments",
            category="similarity",
            parameters={"dataset_name": "ReMIDICaps"}
        )
    
    def _calculate(self, reference_data: Dict[str, Any], generated_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate the F1 score between reference and generated data."""
        # Extract features from both pieces
        ref_features = self._extract_features(reference_data)
        gen_features = self._extract_features(generated_data)
        
        results = {}
        
        # Calculate F1 Score for instruments
        if 'instruments' in ref_features and 'instruments' in gen_features:
            instruments_f1 = self._f1_score(ref_features['instruments'], gen_features['instruments'])
            results["instruments"] = round(instruments_f1, 4)
        
        # Calculate F1 Score for pitch classes
        if 'pitch_classes' in ref_features and 'pitch_classes' in gen_features:
            pitch_f1 = self._f1_score(ref_features['pitch_classes'], gen_features['pitch_classes'])
            results["pitch"] = round(pitch_f1, 4)
        
        # Calculate F1 Score for duration groups
        if 'duration_groups' in ref_features and 'duration_groups' in gen_features:
            duration_f1 = self._f1_score(ref_features['duration_groups'], gen_features['duration_groups'])
            results["duration"] = round(duration_f1, 4)
        
        # Calculate F1 Score for velocity groups
        if 'velocity_groups' in ref_features and 'velocity_groups' in gen_features:
            velocity_f1 = self._f1_score(ref_features['velocity_groups'], gen_features['velocity_groups'])
            results["velocity"] = round(velocity_f1, 4)
        
        # Calculate average for overall score
        if results:
            results["overall"] = round(sum(results.values()) / len(results), 4)
        
        return {self.name: results}
    
    def _extract_features(self, processed_data: Dict[str, Any]) -> Dict[str, List[Any]]:
        """Extract categorical features from processed data."""
        features = {}
        
        # Extract symbolic or event data
        all_tokens = []
        if "symbolic_features" in processed_data and "bar_symbolic" in processed_data["symbolic_features"]:
            all_tokens = processed_data["symbolic_features"]["bar_symbolic"]
        elif "encodings" in processed_data and "events" in processed_data["encodings"]:
            all_tokens = processed_data["encodings"]["events"]
        
        # Extract instruments
        instruments = set()
        for token in all_tokens:
            if self._token_matches_prefix(token, self.INSTRUMENT_KEY):
                # For instruments, use the full value (don't split by numbers)
                instrument_parts = token.split('_')
                if len(instrument_parts) > 1:
                    instruments.add(instrument_parts[1])
        features['instruments'] = list(instruments)
        
        # Extract pitches and convert to pitch classes (0-11)
        pitches = []
        durations = []
        velocities = []
        
        for token in all_tokens:
            if self._token_matches_prefix(token, self.MEAN_PITCH_KEY):
                pitch_value = self._extract_value_from_token(token, self.MEAN_PITCH_KEY)
                if pitch_value is not None:
                    pitches.append(pitch_value)
            elif self._token_matches_prefix(token, self.MEAN_DURATION_KEY):
                duration_value = self._extract_value_from_token(token, self.MEAN_DURATION_KEY)
                if duration_value is not None:
                    durations.append(duration_value)
            elif self._token_matches_prefix(token, self.MEAN_VELOCITY_KEY):
                velocity_value = self._extract_value_from_token(token, self.MEAN_VELOCITY_KEY)
                if velocity_value is not None:
                    velocities.append(velocity_value)
        
        # Convert pitches to pitch classes
        pitch_classes = set([pitch % 12 for pitch in pitches])
        features['pitch_classes'] = list(pitch_classes)
        
        # Group durations into categories
        duration_groups = set()
        for duration in durations:
            if duration < 20:  # very short notes
                duration_groups.add('very_short')
            elif duration < 40:  # short notes
                duration_groups.add('short')
            elif duration < 80:  # medium notes
                duration_groups.add('medium')
            else:  # long notes
                duration_groups.add('long')
        features['duration_groups'] = list(duration_groups)
        
        # Group velocities into categories
        velocity_groups = set()
        for velocity in velocities:
            if velocity < 40:  # soft
                velocity_groups.add('soft')
            elif velocity < 80:  # medium
                velocity_groups.add('medium')
            else:  # loud
                velocity_groups.add('loud')
        features['velocity_groups'] = list(velocity_groups)
        
        return features
    
    def _f1_score(self, true_labels: List[Any], pred_labels: List[Any]) -> float:
        """Calculate F1 score for multi-label categorical features."""
        # Convert to sets for intersection calculation
        true_set = set(true_labels)
        pred_set = set(pred_labels)
        
        # Calculate precision and recall
        true_positives = len(true_set.intersection(pred_set))
        
        if len(pred_set) == 0:
            precision = 0
        else:
            precision = true_positives / len(pred_set)
        
        if len(true_set) == 0:
            recall = 0
        else:
            recall = true_positives / len(true_set)
        
        # Calculate F1 score
        if precision + recall == 0:
            return 0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1


class StandardAccuracy(FidelityMetric):
    """
    Metric for calculating standard accuracy between two MIDI files.
    Standard accuracy is used for single-label categorical features like time signatures.
    """
    
    def __init__(self):
        super().__init__(
            name="standard_accuracy",
            description="Standard accuracy for single-label categorical features",
            category="similarity",
            parameters={"dataset_name": "ReMIDICaps"}
        )
    
    def _calculate(self, reference_data: Dict[str, Any], generated_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate the standard accuracy between reference and generated data."""
        # Extract features
        ref_features = self._extract_features(reference_data)
        gen_features = self._extract_features(generated_data)
        
        accuracy_results = {}
        
        # Compare time signature if available
        if ('time_signature' in ref_features and ref_features['time_signature'] and
            'time_signature' in gen_features and gen_features['time_signature']):
            accuracy_results['time_signature'] = round(self._standard_accuracy(
                ref_features['time_signature'], gen_features['time_signature']
            ), 4)
        
        # Compare key signature if available
        if ('key_signature' in ref_features and ref_features['key_signature'] and
            'key_signature' in gen_features and gen_features['key_signature']):
            accuracy_results['key_signature'] = round(self._standard_accuracy(
                ref_features['key_signature'], gen_features['key_signature']
            ), 4)
        
        # Compare most common pitch
        if ('most_common_pitch' in ref_features and 'most_common_pitch' in gen_features):
            accuracy_results['pitch'] = round(self._standard_accuracy(
                ref_features['most_common_pitch'], gen_features['most_common_pitch']
            ), 4)
        
        # Compare most common duration
        if ('most_common_duration' in ref_features and 'most_common_duration' in gen_features):
            accuracy_results['duration'] = round(self._standard_accuracy(
                ref_features['most_common_duration'], gen_features['most_common_duration']
            ), 4)
        
        # Compare most common velocity
        if ('most_common_velocity' in ref_features and 'most_common_velocity' in gen_features):
            accuracy_results['velocity'] = round(self._standard_accuracy(
                ref_features['most_common_velocity'], gen_features['most_common_velocity']
            ), 4)
        
        # Calculate overall accuracy
        if accuracy_results:
            accuracy_results['overall'] = round(sum(accuracy_results.values()) / len(accuracy_results), 4)
        
        return {self.name: accuracy_results}
    
    def _extract_features(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from processed data."""
        features = {}
        
        # Extract symbolic or event data
        all_tokens = []
        if "symbolic_features" in processed_data and "bar_symbolic" in processed_data["symbolic_features"]:
            all_tokens = processed_data["symbolic_features"]["bar_symbolic"]
        elif "encodings" in processed_data and "events" in processed_data["encodings"]:
            all_tokens = processed_data["encodings"]["events"]
        
        # Extract time signature (assuming first one encountered is main time signature)
        for token in all_tokens:
            if self._token_matches_prefix(token, self.TIME_SIGNATURE_KEY):
                # Parse format like "Time Signature_4/4"
                time_sig_value = token.split('_')[-1]
                if '/' in time_sig_value:
                    num, denom = time_sig_value.split('/')
                    try:
                        features['time_signature'] = (int(num), int(denom))
                        break
                    except ValueError:
                        pass
        
        # Extract key signature (assuming first one encountered is main key)
        for token in all_tokens:
            if self._token_matches_prefix(token, self.KEY_SIGNATURE_KEY):
                # Parse format like "Key Signature_C:min"
                key_value = token.split('_')[-1]
                if ':' in key_value:
                    parts = key_value.split(':')
                    if len(parts) >= 2:
                        root, mode = parts[0], parts[1]
                        features['key_signature'] = (root, mode)
                        break
        
        # Extract pitches, durations, and velocities
        pitches = []
        durations = []
        velocities = []
        
        for token in all_tokens:
            if self._token_matches_prefix(token, self.MEAN_PITCH_KEY):
                pitch_value = self._extract_value_from_token(token, self.MEAN_PITCH_KEY)
                if pitch_value is not None:
                    pitches.append(pitch_value)
            elif self._token_matches_prefix(token, self.MEAN_DURATION_KEY):
                duration_value = self._extract_value_from_token(token, self.MEAN_DURATION_KEY)
                if duration_value is not None:
                    durations.append(duration_value)
            elif self._token_matches_prefix(token, self.MEAN_VELOCITY_KEY):
                velocity_value = self._extract_value_from_token(token, self.MEAN_VELOCITY_KEY)
                if velocity_value is not None:
                    velocities.append(velocity_value)
        
        # Find most common pitch, duration, and velocity
        if pitches:
            features['most_common_pitch'] = Counter(pitches).most_common(1)[0][0]
        
        if durations:
            features['most_common_duration'] = Counter(durations).most_common(1)[0][0]
        
        if velocities:
            features['most_common_velocity'] = Counter(velocities).most_common(1)[0][0]
        
        return features
    
    def _standard_accuracy(self, true_label: Any, pred_label: Any) -> float:
        """Calculate standard accuracy for single-label categorical features."""
        return 1.0 if true_label == pred_label else 0.0


class NRMSE(FidelityMetric):
    """
    Metric for calculating Normalized Root Mean Squared Error (NRMSE) between two MIDI files.
    NRMSE is used for continuous features like note density.
    """
    
    def __init__(self):
        super().__init__(
            name="nrmse",
            description="Normalized Root Mean Squared Error for continuous features",
            category="error",
            parameters={"max_bars": 16, "dataset_name": "ReMIDICaps"}
        )
    
    def _calculate(self, reference_data: Dict[str, Any], generated_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate the NRMSE between reference and generated data."""
        # Get the max bars parameter
        max_bars = self.parameters.get("max_bars", 16)
        
        # Extract features per bar
        ref_features = self._extract_bar_features(reference_data, max_bars)
        gen_features = self._extract_bar_features(generated_data, max_bars)
        
        result = {}
        
        # Calculate NRMSE for note density
        ref_note_density = [len(bar['pitch']) for bar in ref_features]
        gen_note_density = [len(bar['pitch']) for bar in gen_features]
        note_density_nrmse = self._normalized_rmse(ref_note_density, gen_note_density)
        result["note_density"] = round(note_density_nrmse, 4) if not math.isnan(note_density_nrmse) else "N/A"
        
        # Calculate NRMSE for pitch
        ref_pitch_means = [np.mean(bar['pitch']) if bar['pitch'] else 0 for bar in ref_features]
        gen_pitch_means = [np.mean(bar['pitch']) if bar['pitch'] else 0 for bar in gen_features]
        pitch_nrmse = self._normalized_rmse(ref_pitch_means, gen_pitch_means)
        result["pitch"] = round(pitch_nrmse, 4) if not math.isnan(pitch_nrmse) else "N/A"
        
        # Calculate NRMSE for duration
        ref_duration_means = [np.mean(bar['duration']) if bar['duration'] else 0 for bar in ref_features]
        gen_duration_means = [np.mean(bar['duration']) if bar['duration'] else 0 for bar in gen_features]
        duration_nrmse = self._normalized_rmse(ref_duration_means, gen_duration_means)
        result["duration"] = round(duration_nrmse, 4) if not math.isnan(duration_nrmse) else "N/A"
        
        # Calculate NRMSE for velocity
        ref_velocity_means = [np.mean(bar['velocity']) if bar['velocity'] else 0 for bar in ref_features]
        gen_velocity_means = [np.mean(bar['velocity']) if bar['velocity'] else 0 for bar in gen_features]
        velocity_nrmse = self._normalized_rmse(ref_velocity_means, gen_velocity_means)
        result["velocity"] = round(velocity_nrmse, 4) if not math.isnan(velocity_nrmse) else "N/A"
        
        # Calculate overall NRMSE as average
        valid_nrmse = [v for v in result.values() if isinstance(v, (int, float)) and not math.isnan(v)]
        if valid_nrmse:
            result["overall"] = round(sum(valid_nrmse) / len(valid_nrmse), 4)
        
        return {self.name: result}
    
    def _extract_bar_features(self, processed_data: Dict[str, Any], max_bars: int) -> List[Dict[str, List[Any]]]:
        """Extract features per bar from processed data."""
        bar_features = []
        
        # Initialize the first bar
        current_bar = {"pitch": [], "duration": [], "velocity": []}
        bar_features.append(current_bar)
        current_bar_index = 0
        
        # Check if processed data has symbolic features
        if ("symbolic_features" in processed_data and 
            "bar_symbolic" in processed_data["symbolic_features"]):
            tokens = processed_data["symbolic_features"]["bar_symbolic"]
            
            # Identify bar tokens
            bar_indices = [i for i, token in enumerate(tokens) 
                          if self._token_matches_prefix(token, self.BAR_KEY)]
            
            # Add end index
            bar_indices.append(len(tokens))
            
            # Process each bar
            for i in range(min(len(bar_indices)-1, max_bars)):
                start_idx = bar_indices[i]
                end_idx = bar_indices[i+1]
                
                # Create a new bar feature set
                bar_data = {"pitch": [], "duration": [], "velocity": []}
                
                # Process tokens in this bar
                for token in tokens[start_idx:end_idx]:
                    # Extract pitches
                    if self._token_matches_prefix(token, self.MEAN_PITCH_KEY):
                        pitch_value = self._extract_value_from_token(token, self.MEAN_PITCH_KEY)
                        if pitch_value is not None:
                            bar_data["pitch"].append(pitch_value)
                    # Extract durations
                    elif self._token_matches_prefix(token, self.MEAN_DURATION_KEY):
                        duration_value = self._extract_value_from_token(token, self.MEAN_DURATION_KEY)
                        if duration_value is not None:
                            bar_data["duration"].append(duration_value)
                    # Extract velocities
                    elif self._token_matches_prefix(token, self.MEAN_VELOCITY_KEY):
                        velocity_value = self._extract_value_from_token(token, self.MEAN_VELOCITY_KEY)
                        if velocity_value is not None:
                            bar_data["velocity"].append(velocity_value)
                
                bar_features.append(bar_data)
            
        elif "encodings" in processed_data and "events" in processed_data["encodings"]:
            events = processed_data["encodings"]["events"]
            
            # Find bar tokens to segment events
            bar_tokens = [i for i, e in enumerate(events) 
                         if self._token_matches_prefix(e, self.BAR_KEY)]
            
            if not bar_tokens:
                # If no bar tokens, create artificial segments (every 20 events)
                segment_indices = list(range(0, len(events), 20))
                if len(events) not in segment_indices:
                    segment_indices.append(len(events))
            else:
                # Add start and end indices
                segment_indices = [0] + bar_tokens
                if segment_indices[-1] != len(events):
                    segment_indices.append(len(events))
            
            # Process each segment (up to max_bars)
            for i in range(min(len(segment_indices)-1, max_bars)):
                start_idx = segment_indices[i]
                end_idx = segment_indices[i+1]
                
                # Create a new bar feature set
                bar_data = {"pitch": [], "duration": [], "velocity": []}
                
                # Process events in this segment
                for event in events[start_idx:end_idx]:
                    # Extract pitches
                    if self._token_matches_prefix(event, self.MEAN_PITCH_KEY):
                        pitch_value = self._extract_value_from_token(event, self.MEAN_PITCH_KEY)
                        if pitch_value is not None:
                            bar_data["pitch"].append(pitch_value)
                    # Extract durations
                    elif self._token_matches_prefix(event, self.MEAN_DURATION_KEY):
                        duration_value = self._extract_value_from_token(event, self.MEAN_DURATION_KEY)
                        if duration_value is not None:
                            bar_data["duration"].append(duration_value)
                    # Extract velocities
                    elif self._token_matches_prefix(event, self.MEAN_VELOCITY_KEY):
                        velocity_value = self._extract_value_from_token(event, self.MEAN_VELOCITY_KEY)
                        if velocity_value is not None:
                            bar_data["velocity"].append(velocity_value)
                
                bar_features.append(bar_data)
        
        # Ensure we have exactly max_bars bars (pad if necessary)
        while len(bar_features) < max_bars:
            bar_features.append({"pitch": [], "duration": [], "velocity": []})
        
        # Trim to max_bars
        bar_features = bar_features[:max_bars]
        
        return bar_features
    
    def _normalized_rmse(self, reference: List[Any], generated: List[Any]) -> float:
        """Calculate normalized Root Mean Square Error."""
        if not reference or not generated:
            return float('nan')
        
        # Make sure both lists are the same length
        min_len = min(len(reference), len(generated))
        ref = reference[:min_len]
        gen = generated[:min_len]
        
        # Calculate mean of reference for normalization
        ref_mean = np.mean(ref)
        if ref_mean == 0:
            ref_mean = 1.0  # Avoid division by zero
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean([(r - g) ** 2 for r, g in zip(ref, gen)]))
        
        # Normalize by mean of reference
        nrmse = rmse / ref_mean
        
        return float(nrmse)


class KLDivergence(FidelityMetric):
    """
    Metric for calculating Kullback-Leibler divergence between two MIDI files.
    KL divergence measures how one probability distribution diverges from another.
    """
    
    def __init__(self):
        super().__init__(
            name="kl_divergence",
            description="Kullback-Leibler divergence between probability distributions",
            category="distribution",
            parameters={"dataset_name": "ReMIDICaps"}
        )
    
    def _calculate(self, reference_data: Dict[str, Any], generated_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate the KL divergence between reference and generated data."""
        # Extract features
        ref_features = self._extract_features(reference_data)
        gen_features = self._extract_features(generated_data)
        
        results = {}
        
        # Calculate KL divergence for pitch
        pitch_kl = self._kl_divergence(ref_features['pitch_hist'], gen_features['pitch_hist'])
        results["pitch"] = round(pitch_kl, 4)
        
        # Calculate KL divergence for duration
        duration_kl = self._kl_divergence(ref_features['duration_hist'], gen_features['duration_hist'])
        results["duration"] = round(duration_kl, 4)
        
        # Calculate KL divergence for velocity
        velocity_kl = self._kl_divergence(ref_features['velocity_hist'], gen_features['velocity_hist'])
        results["velocity"] = round(velocity_kl, 4)
        
        # Calculate overall as average
        results["overall"] = round((pitch_kl + duration_kl + velocity_kl) / 3, 4)
        
        return {self.name: results}
    
    def _extract_features(self, processed_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Extract features and create histograms from processed data."""
        # Extract symbolic or event data
        all_tokens = []
        if "symbolic_features" in processed_data and "bar_symbolic" in processed_data["symbolic_features"]:
            all_tokens = processed_data["symbolic_features"]["bar_symbolic"]
        elif "encodings" in processed_data and "events" in processed_data["encodings"]:
            all_tokens = processed_data["encodings"]["events"]
        
        # Extract pitches, durations, and velocities
        pitches = []
        durations = []
        velocities = []
        
        for token in all_tokens:
            if self._token_matches_prefix(token, self.MEAN_PITCH_KEY):
                pitch_value = self._extract_value_from_token(token, self.MEAN_PITCH_KEY)
                if pitch_value is not None:
                    pitches.append(pitch_value)
            elif self._token_matches_prefix(token, self.MEAN_DURATION_KEY):
                duration_value = self._extract_value_from_token(token, self.MEAN_DURATION_KEY)
                if duration_value is not None:
                    durations.append(duration_value)
            elif self._token_matches_prefix(token, self.MEAN_VELOCITY_KEY):
                velocity_value = self._extract_value_from_token(token, self.MEAN_VELOCITY_KEY)
                if velocity_value is not None:
                    velocities.append(velocity_value)
        
        features = {}
        
        # Create pitch histogram (ensure at least some data)
        if not pitches:
            pitches = [60, 62, 64, 65, 67]  # Default C major scale
        
        # Create pitch histogram
        pitch_bins = 128  # MIDI pitch range is 0-127
        features['pitch_hist'] = self._create_histogram(pitches, bins=pitch_bins)
        
        # Create duration histogram (logarithmic bins)
        if not durations:
            durations = [10, 20, 30, 40, 50]  # Default durations
            
        duration_bins = 10
        max_duration = max(durations) + 1
        duration_bin_edges = np.logspace(0, np.log10(max_duration), duration_bins + 1) - 1
        features['duration_hist'], _ = np.histogram(durations, bins=duration_bin_edges, density=True)
        
        # Create velocity histogram
        if not velocities:
            velocities = [60, 70, 80, 90, 100]  # Default velocities
            
        velocity_bins = 128  # MIDI velocity range is 0-127
        features['velocity_hist'] = self._create_histogram(velocities, bins=velocity_bins)
        
        return features
    
    def _create_histogram(self, values: List[Any], bins: int) -> np.ndarray:
        """Create a histogram from a list of values."""
        if not values:
            # Return a uniform distribution if no values
            return np.ones(bins) / bins
            
        hist, _ = np.histogram(values, bins=bins, density=True)
        
        # Add small constant to avoid zero probabilities
        hist = hist + 1e-10
        
        # Normalize to ensure it sums to 1
        hist = hist / np.sum(hist)
        
        return hist
    
    def _kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Calculate the Kullback-Leibler divergence between two distributions."""
        # Add small constant to avoid zero probabilities
        p = p + 1e-10
        q = q + 1e-10
        
        # Normalize to ensure both sum to 1
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        # Calculate KL divergence
        kl = np.sum(p * np.log(p / q))
        
        # Handle numerical issues
        if np.isnan(kl) or np.isinf(kl):
            return 100.0  # Return a high value instead of infinity
            
        return float(kl)


# Dictionary mapping metric names to their classes for easy instantiation
FIDELITY_METRICS = {
    "macro_overlapping_area": MacroOverlappingArea,
    "f1_score": F1Score,
    "standard_accuracy": StandardAccuracy,
    "nrmse": NRMSE,
    "kl_divergence": KLDivergence
} 