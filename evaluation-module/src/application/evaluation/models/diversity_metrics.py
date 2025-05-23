import os
import numpy as np
import muspy
import scipy
import pickle
import re
from scipy.linalg import sqrtm
from typing import Dict, Any, Optional, List, Tuple, Union
from collections import defaultdict, Counter

from domain.constants.paths_constants import PROCESSED_PATH
from application.evaluation.models.evaluation_base import EvaluationBase
from application.encoder.models.vocab_model import SymbolicFeaturesVocab


class DiversityMetric(EvaluationBase):
    """Base class for diversity metrics that compare distributions of music."""

    def __init__(self, name: str, description: str, category: str, parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize a diversity metric.

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

    def evaluate(self, midi_path: str, reference_midi_path: Optional[str] = None, max_time: float = 0) -> Dict[str, Any]:
        """
        Evaluate a MIDI file against a reference using this diversity metric.

        For single file comparison, this typically isn't meaningful for diversity metrics,
        which are designed to compare datasets. This will return a placeholder message.

        Args:
            midi_path: Path to the MIDI file to evaluate
            reference_midi_path: Path to the reference MIDI file
            max_time: Maximum time in seconds to consider

        Returns:
            Dictionary containing a placeholder message
        """
        return {self.name: {"error": "Diversity metrics are designed to compare datasets, not individual files. Use evaluate_dataset instead."}}

    def extract_features(self, midi_path: str, max_time: float = 0) -> np.ndarray:
        """
        Extract relevant features from a MIDI file for diversity calculation.

        Args:
            midi_path: Path to the MIDI file
            max_time: Maximum time in seconds to consider

        Returns:
            Numpy array of features
        """
        raise NotImplementedError("Subclasses must implement extract_features method")

    def normalize_score(self, score: float, min_val: float = 0, max_val: float = 300, invert: bool = True) -> float:
        """
        Normalize a diversity score to the range [0, 1].

        Args:
            score: The original metric score
            min_val: The minimum expected value (default: 0)
            max_val: The maximum expected value (default: 300, typical for high FMD)
            invert: Whether to invert the normalized score (for metrics where lower is better)

        Returns:
            Normalized score in the range [0, 1]
        """
        if np.isnan(score):
            return 0.0

        # Apply clipping to handle outliers
        score = max(min_val, min(score, max_val))

        # Linear scaling between min and max
        norm_score = (score - min_val) / (max_val - min_val)

        # Invert if needed (e.g., for FMD where lower is better)
        return 1.0 - norm_score if invert else norm_score
    
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


class FrechetMusicDistance(DiversityMetric):
    """
    Metric for calculating Frechet Music Distance (FMD) between two MIDI files.

    FMD is inspired by Frechet Inception Distance (FID) and Frechet Audio Distance (FAD),
    and measures how closely the distribution of generated music aligns with real music.

    The mathematical formula is:
    FMD = ||μr - μt||² + Tr(Σr + Σt - 2 × sqrt(ΣrΣt))

    where μr and μt are the mean vectors of the reference and test distributions,
    Σr and Σt are their covariance matrices, and Tr is the trace operator.

    Lower FMD values indicate better performance, as they suggest a smaller distance
    between the distributions of the generated and reference music.
    """

    def __init__(self):
        super().__init__(
            name="frechet_music_distance",
            description="Measures distributional similarity between reference and generated music",
            category="diversity",
            parameters={
                "dataset_name": "ReMIDICaps",  # Default dataset name
            }
        )
        self.symb_vocab = SymbolicFeaturesVocab()
        self.feature_categories = {
            "pitch": [self.MEAN_PITCH_KEY, self.PITCH_KEY],
            "rhythm": [self.TIME_SIGNATURE_KEY, self.POSITION_KEY, self.NOTE_DENSITY_KEY, self.MEAN_DURATION_KEY],
            "velocity": [self.MEAN_VELOCITY_KEY, self.VELOCITY_KEY],
            "harmony": [self.KEY_SIGNATURE_KEY, self.CHORD_KEY],
            "instrumentation": [self.INSTRUMENT_KEY],
            "structure": [self.BAR_KEY]
        }

    def evaluate(self, midi_path: str, reference_midi_path: Optional[str] = None, max_time: float = 0) -> Dict[str, Any]:
        """
        Calculate Frechet Music Distance between a generated MIDI and a reference.

        Args:
            midi_path: Path to the generated MIDI file
            reference_midi_path: Path to the reference MIDI file
            max_time: Maximum time in seconds to consider

        Returns:
            Dictionary with raw FMD scores (not normalized)
        """
        if reference_midi_path is None:
            return {self.name: {"error": "Reference MIDI file is required for FMD calculation"}}

        try:
            # Extract features from processed data
            gen_distributions = self._extract_symbolic_distributions(midi_path)
            ref_distributions = self._extract_symbolic_distributions(reference_midi_path)
            
            # Calculate FMD for all feature categories
            fmd_scores = {}
            
            # Calculate FMD for each feature category
            for category_name in gen_distributions.keys():
                fmd_scores[category_name] = self._calculate_fmd(
                    ref_distributions[category_name], 
                    gen_distributions[category_name]
                )
            
            # Calculate overall FMD as the mean of all category FMDs
            overall_fmd = sum(fmd_scores.values()) / len(fmd_scores)
            fmd_scores["overall"] = overall_fmd
            
            # Round all scores for cleaner output
            result = {k: round(v, 4) for k, v in fmd_scores.items()}
            
            return {self.name: result}

        except Exception as e:
            return {"error": f"Error calculating FMD: {str(e)}"}
    
    def evaluate_dataset(self, generated_dir: str, reference_dir: str, 
                        max_files: int = 0, midi_extension: str = ".mid") -> Dict[str, Any]:
        """
        Calculate Frechet Music Distance between two directories of MIDI files.
        
        This method properly implements FMD as a distribution-level metric by:
        1. Aggregating features across all MIDI files in each directory
        2. Calculating statistical distributions (mean and covariance) on these aggregated features
        3. Computing the FMD between these distributions
        
        Args:
            generated_dir: Directory containing generated MIDI files
            reference_dir: Directory containing reference MIDI files
            max_files: Maximum number of files to process from each directory (0 = all files)
            midi_extension: File extension to filter for MIDI files
            
        Returns:
            Dictionary with FMD scores for each feature category
        """
        try:
            # Collect MIDI files from both directories
            gen_files = [os.path.join(generated_dir, f) for f in os.listdir(generated_dir) 
                        if f.endswith(midi_extension)]
            ref_files = [os.path.join(reference_dir, f) for f in os.listdir(reference_dir) 
                        if f.endswith(midi_extension)]
            
            # Limit number of files if specified
            if max_files > 0:
                gen_files = gen_files[:max_files]
                ref_files = ref_files[:max_files]
            
            # Print number of files being processed
            print(f"Processing {len(gen_files)} generated files and {len(ref_files)} reference files")
            
            if not gen_files or not ref_files:
                return {self.name: {"error": "No MIDI files found in one or both directories"}}
            
            # Initialize feature vectors for all categories
            all_gen_features = {category: [] for category in self.feature_categories.keys()}
            all_ref_features = {category: [] for category in self.feature_categories.keys()}
            
            # Process generated files
            print("Extracting features from generated files...")
            for midi_path in gen_files:
                try:
                    # Extract distributions from this file
                    file_distributions = self._extract_symbolic_distributions(midi_path)
                    
                    # For each feature category, collect the feature vectors
                    for category, feature_vectors in all_gen_features.items():
                        # The raw feature vectors come from the aggregation of all bar segments
                        # before the mean and covariance calculation
                        if category in file_distributions:
                            # Get raw feature vectors that were used to compute distribution
                            raw_vectors = self._get_raw_feature_vectors(midi_path, category)
                            if raw_vectors:
                                feature_vectors.extend(raw_vectors)
                except Exception as e:
                    print(f"Error processing generated file {os.path.basename(midi_path)}: {e}")
                    continue
            
            # Process reference files
            print("Extracting features from reference files...")
            for midi_path in ref_files:
                try:
                    # Extract distributions from this file
                    file_distributions = self._extract_symbolic_distributions(midi_path)
                    
                    # For each feature category, collect the feature vectors
                    for category, feature_vectors in all_ref_features.items():
                        # The raw feature vectors come from the aggregation of all bar segments
                        # before the mean and covariance calculation
                        if category in file_distributions:
                            # Get raw feature vectors that were used to compute distribution
                            raw_vectors = self._get_raw_feature_vectors(midi_path, category)
                            if raw_vectors:
                                feature_vectors.extend(raw_vectors)
                except Exception as e:
                    print(f"Error processing reference file {os.path.basename(midi_path)}: {e}")
                    continue
            
            # Calculate distributions for each category from the aggregated feature vectors
            gen_distributions = {}
            ref_distributions = {}
            
            for category in self.feature_categories.keys():
                gen_vectors = all_gen_features[category]
                ref_vectors = all_ref_features[category]
                
                # Need at least 2 vectors for meaningful distribution
                if len(gen_vectors) < 2:
                    print(f"Warning: Not enough generated feature vectors for category '{category}'. Using synthetic data.")
                    gen_vectors = [np.random.rand(16) for _ in range(10)]
                
                if len(ref_vectors) < 2:
                    print(f"Warning: Not enough reference feature vectors for category '{category}'. Using synthetic data.")
                    ref_vectors = [np.random.rand(16) for _ in range(10)]
                
                # Calculate mean and covariance
                gen_array = np.array(gen_vectors)
                ref_array = np.array(ref_vectors)
                
                gen_distributions[category] = {
                    "mean": np.mean(gen_array, axis=0),
                    "cov": np.cov(gen_array, rowvar=False)
                }
                
                ref_distributions[category] = {
                    "mean": np.mean(ref_array, axis=0),
                    "cov": np.cov(ref_array, rowvar=False)
                }
            
            # Calculate FMD for all feature categories
            fmd_scores = {}
            
            # Calculate FMD for each feature category
            for category_name in gen_distributions.keys():
                fmd_scores[category_name] = self._calculate_fmd(
                    ref_distributions[category_name], 
                    gen_distributions[category_name]
                )
            
            # Calculate overall FMD as the mean of all category FMDs
            # Skip any infinity or NaN values
            valid_scores = [score for score in fmd_scores.values() if not np.isnan(score) and not np.isinf(score)]
            if valid_scores:
                overall_fmd = sum(valid_scores) / len(valid_scores)
            else:
                overall_fmd = 1000.0  # Default to a high value if all scores are invalid
            fmd_scores["overall"] = overall_fmd
            
            # Add vector counts for reference
            fmd_scores["feature_vector_counts"] = {
                "generated": {category: len(vectors) for category, vectors in all_gen_features.items()},
                "reference": {category: len(vectors) for category, vectors in all_ref_features.items()}
            }
            
            # [FIX 4] - Add sanity check comparison against random data
            # Generate random data with similar structure to the generated data
            print("Running sanity check with random feature vectors...")
            random_vectors = {category: [] for category in self.feature_categories.keys()}
            
            for category in self.feature_categories.keys():
                feature_dim = 16  # Standard dimension for our feature vectors
                num_vectors = len(all_gen_features[category])
                if num_vectors > 0:
                    # Create random vectors similar in size to generated
                    for _ in range(num_vectors):
                        random_vectors[category].append(np.random.rand(feature_dim))
            
            # Calculate distributions for random data
            random_distributions = {}
            for category in self.feature_categories.keys():
                if len(random_vectors[category]) > 1:
                    rand_array = np.array(random_vectors[category])
                    random_distributions[category] = {
                        "mean": np.mean(rand_array, axis=0),
                        "cov": np.cov(rand_array, rowvar=False)
                    }
            
            # Calculate FMD between reference and random
            random_fmd = {}
            for category_name in random_distributions.keys():
                if category_name in ref_distributions:
                    random_fmd[category_name] = self._calculate_fmd(
                        ref_distributions[category_name],
                        random_distributions[category_name]
                    )
            
            # Calculate overall random FMD
            valid_random_scores = [score for score in random_fmd.values() if not np.isnan(score) and not np.isinf(score)]
            if valid_random_scores:
                random_fmd["overall"] = sum(valid_random_scores) / len(valid_random_scores)
            else:
                random_fmd["overall"] = 1000.0
            
            # Add random FMD results to output
            fmd_scores["random_baseline"] = {k: round(v, 4) if isinstance(v, (int, float)) else v 
                                             for k, v in random_fmd.items()}
            
            # Round all scores for cleaner output (except feature_vector_counts and random_baseline)
            result = {}
            for k, v in fmd_scores.items():
                if k not in ["feature_vector_counts", "random_baseline"]:
                    result[k] = round(v, 4) if not np.isnan(v) and not np.isinf(v) else v
                else:
                    result[k] = v
            
            return {self.name: result}
            
        except Exception as e:
            import traceback
            traceback_str = traceback.format_exc()
            return {self.name: {"error": f"Error calculating dataset FMD: {str(e)}\n{traceback_str}"}}
    
    def _get_raw_feature_vectors(self, midi_path: str, category: str) -> List[np.ndarray]:
        """
        Extract raw feature vectors for a specific category from a MIDI file.
        
        This method extracts feature vectors from each bar segment without 
        calculating the mean and covariance.
        
        Args:
            midi_path: Path to the MIDI file
            category: Feature category to extract
            
        Returns:
            List of feature vectors
        """
        try:
            # Get dataset name
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
                return []
            
            # Load processed data
            processed_data = pickle.load(open(processed_file, "rb"))
            
            # Initialize feature vectors for this category
            feature_vectors = []
            
            # Extract symbolic features
            if "symbolic_features" in processed_data and "bar_symbolic" in processed_data["symbolic_features"]:
                # Extract bar_symbolic tokens
                bar_symbolic = processed_data["symbolic_features"]["bar_symbolic"]
                
                # Find bar tokens to segment the symbolic features
                bar_indices = [i for i, token in enumerate(bar_symbolic) 
                              if self._token_matches_prefix(token, self.BAR_KEY)]
                
                # Add end index
                bar_indices.append(len(bar_symbolic))
                
                # Process each bar segment
                for i in range(len(bar_indices) - 1):
                    start_idx = bar_indices[i]
                    end_idx = bar_indices[i+1]
                    
                    # Skip empty or too small segments
                    if end_idx - start_idx < 3:
                        continue
                    
                    # Get tokens for this segment
                    segment_tokens = bar_symbolic[start_idx:end_idx]
                    
                    # Extract features for this category from this segment
                    prefixes = self.feature_categories[category]
                    feature_vec = self._extract_category_features_from_segment(segment_tokens, prefixes)
                    if feature_vec is not None:
                        feature_vectors.append(feature_vec)
            elif "encodings" in processed_data and "events" in processed_data["encodings"]:
                # Fallback to event-based feature extraction
                events = processed_data["encodings"]["events"]
                
                # Segment events by bars
                bar_tokens = [i for i, e in enumerate(events) if self._token_matches_prefix(e, self.BAR_KEY)]
                if not bar_tokens:
                    # If no bar tokens, create artificial segments (every 20 events)
                    segments = [events[i:i+20] for i in range(0, len(events), 20)]
                else:
                    # Add start and end indices
                    bar_tokens = [0] + bar_tokens + [len(events)]
                    # Create segments using bar tokens
                    segments = [events[bar_tokens[i]:bar_tokens[i+1]] for i in range(len(bar_tokens)-1)]
                
                # Process each segment
                for segment in segments:
                    if len(segment) < 3:  # Skip very short segments
                        continue
                        
                    # Extract features for this category
                    prefixes = self.feature_categories[category]
                    category_events = [e for e in segment if any(self._token_matches_prefix(e, prefix) for prefix in prefixes)]
                    if category_events:
                        feature_vec = self._extract_features_from_event_segment(category_events)
                        feature_vectors.append(feature_vec)
                        
            return feature_vectors
            
        except Exception as e:
            print(f"Error extracting raw feature vectors for {midi_path}: {e}")
            return []

    def _extract_symbolic_distributions(self, midi_path: str) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Extract distributions from pre-processed symbolic MIDI data.
        
        Args:
            midi_path: Path to the MIDI file
            
        Returns:
            Dictionary of distributions for each feature category
        """
        dataset_name = self.parameters.get("dataset_name", "ReMIDICaps")
        
        # Get filename from path
        midi_filename = os.path.basename(midi_path)
        
        # Construct path to processed data
        processed_file = os.path.join(
            str(PROCESSED_PATH),
            dataset_name,
            f"{midi_filename}_processed.pkl"
        )
        
        # Load processed data
        if not os.path.exists(processed_file):
            print(f"Processed file not found: {processed_file}")
            # Check what files are available in the directory
            dir_path = os.path.join(str(PROCESSED_PATH), dataset_name)
            if os.path.exists(dir_path):
                print(f"Available files in {dir_path}: {os.listdir(dir_path)}")
            raise FileNotFoundError(f"Processed file not found: {processed_file}")
        
        processed_data = pickle.load(open(processed_file, "rb"))
        
        # Initialize feature vectors for each category
        feature_vectors = {category: [] for category in self.feature_categories.keys()}
        
        # Extract symbolic features
        if "symbolic_features" in processed_data and "bar_symbolic" in processed_data["symbolic_features"]:
            # Extract bar_symbolic tokens
            bar_symbolic = processed_data["symbolic_features"]["bar_symbolic"]
            
            # Find bar tokens to segment the symbolic features
            bar_indices = [i for i, token in enumerate(bar_symbolic) 
                          if self._token_matches_prefix(token, self.BAR_KEY)]
            
            # Add end index
            bar_indices.append(len(bar_symbolic))
            
            # Process each bar segment (or other meaningful segments)
            for i in range(len(bar_indices) - 1):
                start_idx = bar_indices[i]
                end_idx = bar_indices[i+1]
                
                # Skip empty or too small segments
                if end_idx - start_idx < 3:
                    continue
                
                # Get tokens for this segment
                segment_tokens = bar_symbolic[start_idx:end_idx]
                
                # Extract features for each category from this segment
                for category, prefixes in self.feature_categories.items():
                    feature_vec = self._extract_category_features_from_segment(segment_tokens, prefixes)
                    if feature_vec is not None:
                        feature_vectors[category].append(feature_vec)
        else:
            # Fallback to event-based feature extraction if bar_symbolic is not available
            if "encodings" in processed_data and "events" in processed_data["encodings"]:
                events = processed_data["encodings"]["events"]
                
                # Segment events by bars
                bar_tokens = [i for i, e in enumerate(events) if self._token_matches_prefix(e, self.BAR_KEY)]
                if not bar_tokens:
                    # If no bar tokens, create artificial segments (every 20 events)
                    segments = [events[i:i+20] for i in range(0, len(events), 20)]
                else:
                    # Add start and end indices
                    bar_tokens = [0] + bar_tokens + [len(events)]
                    # Create segments using bar tokens
                    segments = [events[bar_tokens[i]:bar_tokens[i+1]] for i in range(len(bar_tokens)-1)]
                
                # Process each segment
                for segment in segments:
                    if len(segment) < 3:  # Skip very short segments
                        continue
                        
                    # Extract features for each category from this segment
                    for category, prefixes in self.feature_categories.items():
                        category_events = [e for e in segment if any(self._token_matches_prefix(e, prefix) for prefix in prefixes)]
                        if category_events:
                            feature_vec = self._extract_features_from_event_segment(category_events)
                            feature_vectors[category].append(feature_vec)
        
        # Calculate distributions for each category from the feature vectors
        distributions = {}
        for category, vectors in feature_vectors.items():
            # Need at least 2 vectors for meaningful distribution
            if len(vectors) < 2:
                # If we don't have enough vectors, create some with small variations
                if len(vectors) == 1:
                    base_vector = vectors[0]
                    # Create additional vectors with small noise
                    additional_vectors = [
                        base_vector + np.random.normal(0, 0.01, base_vector.shape) 
                        for _ in range(9)
                    ]
                    vectors = vectors + additional_vectors
                else:
                    # Create completely random vectors if we have none
                    vectors = [np.random.rand(16) for _ in range(10)]
            
            # Calculate mean and covariance
            vectors_array = np.array(vectors)
            distributions[category] = {
                "mean": np.mean(vectors_array, axis=0),
                "cov": np.cov(vectors_array, rowvar=False)
            }
        
        return distributions
    
    def _extract_category_features_from_segment(self, segment_tokens, prefixes):
        """Extract features for a specific category from a segment of tokens."""
        # Filter tokens for this category
        category_tokens = [token for token in segment_tokens if 
                         any(self._token_matches_prefix(token, prefix) for prefix in prefixes)]
        
        if not category_tokens:
            return None  # No relevant tokens in this segment
        
        # Create feature vector (16 dimensions)
        feature_vec = np.zeros(16)
        
        # Count token occurrences
        token_counts = Counter(category_tokens)
        
        # Fill in the first part with token frequencies (normalized)
        for i, (token, count) in enumerate(token_counts.most_common(8)):
            if i < 8:
                feature_vec[i] = count / len(category_tokens)
                # Also encode token information using hash
                feature_vec[i+8] = hash(token) % 100 / 100.0
        
        return feature_vec
    
    def _extract_features_from_event_segment(self, events):
        """Extract features from a segment of events."""
        if not events:
            return np.zeros(16)
        
        # Count event occurrences
        event_counts = Counter(events)
        
        # Create feature vector (16 dimensions)
        feature_vec = np.zeros(16)
        
        # Fill in with event frequencies and hashed values
        for i, (event, count) in enumerate(event_counts.most_common(8)):
            if i < 8:
                feature_vec[i] = count / len(events)
                feature_vec[i+8] = hash(event) % 100 / 100.0
        
        return feature_vec

    def _calculate_fmd(self, dist1: Dict[str, np.ndarray], dist2: Dict[str, np.ndarray]) -> float:
        """
        Calculate the Frechet Music Distance between two distributions.

        FMD = ||μr - μt||² + Tr(Σr + Σt - 2 × sqrt(ΣrΣt))

        Args:
            dist1: First distribution with mean and covariance
            dist2: Second distribution with mean and covariance

        Returns:
            Frechet Music Distance score
        """
        # Extract mean vectors and covariance matrices
        mu1, sigma1 = dist1["mean"], dist1["cov"]
        mu2, sigma2 = dist2["mean"], dist2["cov"]

        # Add small regularization to prevent singular matrices
        # This ensures numerical stability in the matrix square root calculation
        eps = 1e-6
        sigma1 = sigma1 + np.eye(sigma1.shape[0]) * eps
        sigma2 = sigma2 + np.eye(sigma2.shape[0]) * eps

        # Calculate squared distance between means
        mean_diff_squared = np.sum((mu1 - mu2) ** 2)

        # Calculate matrix square root term
        try:
            sqrt_term = sqrtm(sigma1.dot(sigma2))
            # Ensure the output is real (sqrt of a complex number might have small imaginary part due to numerical error)
            if np.iscomplexobj(sqrt_term):
                sqrt_term = sqrt_term.real

            # Calculate trace term
            trace_term = np.trace(sigma1 + sigma2 - 2 * sqrt_term)

            # Calculate FMD
            fmd = mean_diff_squared + trace_term

            # Handle numerical instability
            if np.isnan(fmd) or np.isinf(fmd):
                print(f"Warning: FMD calculation produced {fmd}. Using high value instead.")
                return 1000.0  # Return a high finite value instead of infinity

            return float(fmd)

        except Exception as e:
            print(f"Error in FMD calculation: {e}")
            return 1000.0  # Return a high value instead of infinity


# Dictionary mapping metric names to their classes for easy instantiation
DIVERSITY_METRICS = {"frechet_music_distance": FrechetMusicDistance}
