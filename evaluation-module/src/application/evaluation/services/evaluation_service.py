import os
import json
import asyncio
import pandas as pd
import numpy as np
import random
from typing import Dict, List, Any, Optional
from collections import defaultdict
from fastapi import HTTPException

from application.evaluation.models.muspy_metrics import MUSPY_METRICS
from application.evaluation.models.fidelity_metrics import FIDELITY_METRICS
from application.evaluation.models.diversity_metrics import DIVERSITY_METRICS

# Combine all metrics for easy access
METRICS = {**MUSPY_METRICS, **FIDELITY_METRICS, **DIVERSITY_METRICS}


def _load_metrics_config():
    config_path = "./shared/conf/metrics_config.json"

    with open(config_path, "r") as f:
        return json.load(f)


def _build_metrics_registry(metrics_config):
    registry = {}

    # Create instances of all metrics from all categories
    for category, metrics in metrics_config["available_metrics"].items():
        for metric in metrics:
            metric_id = metric["id"]
            if metric_id in METRICS:
                try:
                    # Create an instance of the metric class
                    registry[metric_id] = METRICS[metric_id]()
                except Exception as e:
                    print(f"Failed to create metric instance for {metric_id}: {e}")

    return registry


def _metric_requires_reference(metrics_config, metric_id):
    # Check in all categories
    for category, metrics in metrics_config["available_metrics"].items():
        for metric in metrics:
            if metric["id"] == metric_id:
                # If source is "fidelity" or the metric has reference_required=True
                return category == "fidelity" or category == "diversity" or metric.get("reference_required", False)

    return False


def get_available_metrics() -> Dict[str, List[Dict[str, Any]]]:
    """Get all available metrics.

    Returns:
        Dict[str, List[Dict[str, Any]]]: Dictionary of available metrics
    """
    metrics_config = _load_metrics_config()
    return metrics_config["available_metrics"]


def evaluate(
    midi_path: str,
    metrics: Optional[List[str]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    reference_midi_path: Optional[str] = None,
    max_time: float = 30.0,
) -> Dict[str, Any]:
    """
    Unified evaluation function for all metrics.

    Args:
        midi_path (str): Path to the MIDI file to evaluate
        metrics (Optional[List[str]]): List of metrics to evaluate. If None, uses metrics from config
        parameters (Optional[Dict[str, Any]]): Parameters for metrics that require them
        reference_midi_path (Optional[str]): Path to a reference MIDI file for metrics that require it
        max_time (float): Maximum time in seconds to keep from the MIDI file (default: 30 seconds)

    Returns:
        Dict[str, Any]: Dictionary with evaluation results
    """
    # Load the configuration
    metrics_config = _load_metrics_config()
    metrics_registry = _build_metrics_registry(metrics_config)

    # If metrics not specified, use all metrics from metrics_to_use
    if metrics is None:
        metrics = []
        for category, metric_list in metrics_config.get("metrics_to_use", {}).items():
            metrics.extend(metric_list)

    # Check if MIDI file exists
    if not os.path.exists(midi_path):
        return {"error": f"MIDI file not found: {midi_path}"}

    # Use default parameters if none specified
    if parameters is None:
        parameters = metrics_config["default_parameters"]

    results = {}

    # Apply parameters to metric instances
    for metric_id, metric_instance in metrics_registry.items():
        if metric_id in metrics:
            # Extract parameters for this specific metric from the config
            metric_params = {}
            for category, category_metrics in metrics_config["available_metrics"].items():
                for metric_config in category_metrics:
                    if metric_config["id"] == metric_id and "parameters" in metric_config:
                        required_params = metric_config["parameters"]
                        for param in required_params:
                            if param in parameters:
                                metric_params[param] = parameters[param]

            # Update the metric instance parameters
            if metric_params:
                metric_instance.parameters.update(metric_params)

    # Evaluate each requested metric
    for metric_id in metrics:
        if metric_id in metrics_registry:
            metric_instance = metrics_registry[metric_id]

            # Check if this metric requires a reference file
            requires_reference = _metric_requires_reference(metrics_config, metric_id)

            # Skip metrics that require a reference if none is provided
            if requires_reference and not reference_midi_path:
                results[metric_id] = {"error": "Reference MIDI file required but not provided"}
                continue

            # Skip metrics that don't need a reference if one is provided but not used
            if reference_midi_path and not requires_reference:
                # This is not an error, just skip adding the reference
                pass

            try:
                # Call the evaluate method of the metric instance
                metric_result = metric_instance.evaluate(
                    midi_path=midi_path, reference_midi_path=reference_midi_path if requires_reference else None, max_time=max_time
                )

                # Check for errors
                if "error" in metric_result:
                    results[metric_id] = {"error": metric_result["error"]}
                else:
                    # Add the result to the overall results
                    results.update(metric_result)
            except Exception as e:
                results[metric_id] = {"error": str(e)}
        else:
            results[metric_id] = {"error": f"Metric not found: {metric_id}"}

    return results


async def evaluate_dataset(config_path: str) -> dict:
    """Evaluate a dataset of MIDI files using parameters from the config file.

    Args:
        config_path: Path to the configuration file containing evaluation settings

    Returns:
        dict: Summary of the evaluation process including statistics and errors
    """
    # Load configuration
    with open(config_path, "r") as f:
        evaluation_config = json.load(f)

    dataset_dir = "output/demos/demo_2/generated/ReMIDICaps_test_set"
    reference_dir = "datasets/ReMIDICaps/midi"
    reference_available = reference_dir is not None and os.path.exists(reference_dir)
    csv_path = "output/demos/demo_2/generated/ReMIDICaps_test_set_out.csv"
    output_dir = "output/demos/demo_2/evaluation/Diversity"
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    else:
        return {"Message": "Output directory not specified"}

    # Load metrics configuration
    metrics_config = _load_metrics_config()
    metrics_to_use = metrics_config["metrics_to_use"]
    parameters = evaluation_config["evaluation"]["parameters"]
    max_time = parameters["max_time"]

    # Get list of MIDI files
    file_extensions = (".mid", ".midi", ".MID", ".MIDI")
    all_midi_files = [f for f in os.listdir(dataset_dir) if f.endswith(file_extensions)]

    # Filter MIDI files based on CSV if provided
    if csv_path and os.path.exists(csv_path):
        # Load CSV file
        df = pd.read_csv(csv_path)

        # Extract file names from CSV (assuming 'file' is the column name)
        if "file" in df.columns:
            csv_midi_files = df["file"].tolist()
            # Filter MIDI files to only include those in the CSV
            midi_files = [f for f in all_midi_files if f in csv_midi_files]
            print(f"Filtered {len(all_midi_files)} files to {len(midi_files)} based on CSV")
        else:
            print(f"Warning: 'file' column not found in CSV, using all files")
            midi_files = all_midi_files
    else:
        # If no CSV path provided or file doesn't exist, use all MIDI files
        midi_files = all_midi_files

    if not midi_files:
        return {"Message": "No MIDI files found in the dataset directory"}

    total_files = len(midi_files)
    processed = 0
    successful = 0
    errors = []
    all_results = {}

    # Collect all metrics to evaluate
    all_metrics = []
    for category, metric_list in metrics_to_use.items():
        all_metrics.extend(metric_list)

    async def process_file(file_path: str) -> tuple:
        try:
            # Get the filename
            filename = os.path.basename(file_path)

            # Get reference path if reference directory is available
            reference_path = None
            if reference_available:
                # Extract the original reference filename from the generated filename
                # Format: Mood_X_score_Mood_Y_score_..._originalID_generated.mid
                if "_generated.mid" in filename:
                    # Extract the original ID before "_generated.mid"
                    parts = filename.split("_")
                    if len(parts) > 2:  # Ensure we have enough parts
                        # Find the part before "generated"
                        for i in range(len(parts) - 1):
                            if parts[i + 1] == "generated.mid":
                                original_filename = f"{parts[i]}.mid"
                                reference_path = os.path.join(reference_dir, original_filename)
                                break
                else:
                    # If not in the expected format, try using the filename directly
                    reference_path = os.path.join(reference_dir, filename)

                if not os.path.exists(reference_path):
                    print(f"Warning: Reference file not found for {filename}")

            # Evaluate all metrics at once
            eval_results = evaluate(midi_path=file_path, reference_midi_path=reference_path, metrics=all_metrics, parameters=parameters, max_time=max_time)

            # Save evaluation results to output directory
            if output_dir:
                output_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_eval.json")
                with open(output_file, "w") as f:
                    json.dump(eval_results, f, indent=2)

            # Add to overall results
            all_results[filename] = eval_results

            return (True, "")

        except Exception as e:
            filename = os.path.basename(file_path)
            return (False, f"Error processing {filename}: {str(e)}")

    # Process files in batches
    batch_size = 10
    while processed < total_files:
        batch = midi_files[processed : processed + batch_size]
        batch_tasks = [process_file(os.path.join(dataset_dir, file)) for file in batch]

        # Process batch
        results = await asyncio.gather(*batch_tasks, return_exceptions=False)

        # Update counters
        for eval_success, error_msg in results:
            if eval_success:
                successful += 1
            elif error_msg:
                errors.append(error_msg)

        processed += len(batch)
        print(f"Evaluated {processed}/{total_files} files", flush=True)

    # Save overall results
    if output_dir:
        # Save evaluation results
        summary_file = os.path.join(output_dir, f"RemidiCaps_generated_summary.json")
        with open(summary_file, "w") as f:
            json.dump(all_results, f, indent=2)

    # Calculate aggregate statistics if requested
    aggregate_stats = {}
    metric_values = {}
    for file_results in all_results.values():
        for metric, value in file_results.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if metric not in metric_values:
                    metric_values[metric] = []
                metric_values[metric].append(value)

    # Calculate statistics
    for metric, values in metric_values.items():
        # Only calculate stats if we have values
        if values:
            aggregate_stats[metric] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "std": float(np.std(values)),
            }
    # Save aggregate statistics
    if output_dir:
        stats_file = os.path.join(output_dir, f"RemidiCaps_generated_aggregate_stats.json")
        with open(stats_file, "w") as f:
            json.dump(aggregate_stats, f, indent=2)

    # Prepare summary message
    summary = f"Dataset Evaluation Complete\n" f"Total files: {total_files}\n" f"Successfully evaluated: {successful}\n" f"Failed: {len(errors)}\n"

    if errors:
        # Limit the number of errors shown to avoid overwhelming output
        max_errors_to_show = 10
        summary += "\nErrors:\n" + "\n".join(errors[:max_errors_to_show])
        if len(errors) > max_errors_to_show:
            summary += f"\n... and {len(errors) - max_errors_to_show} more errors"

    return {"Message": summary}


async def evaluate_dataset_fmd() -> dict:
    """
    Evaluate a dataset of MIDI files using the batch FMD implementation.
    
    This function computes the Frechet Music Distance between two directories
    of MIDI files at the dataset level, properly capturing distributional
    characteristics across all files.
    
    Args:
        generated_dir: Directory containing generated MIDI files
        reference_dir: Directory containing reference MIDI files
        output_dir: Directory to save results (if None, results are only returned)
        max_files: Maximum number of files to process from each directory (0 = all files)
        parameters: Optional parameters for the FMD calculation
    
    Returns:
        Dictionary with FMD results and summary information
    """


    generated_dir = "output/demos/demo_2/generated/ReMIDICaps_test_set"
    reference_dir = "datasets/ReMIDICaps/midi"
    output_dir = "output/demos/demo_2/evaluation/Diversity"

    # Verify directories exist
    if not os.path.exists(generated_dir):
        return {"error": f"Generated directory not found: {generated_dir}"}
    
    if not os.path.exists(reference_dir):
        return {"error": f"Reference directory not found: {reference_dir}"}
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Create an instance of FrechetMusicDistance
    fmd_metric = DIVERSITY_METRICS["frechet_music_distance"]()
    
    
    print(f"Computing batch FMD between '{generated_dir}' and '{reference_dir}'...")
    print(f"Using parameters: {fmd_metric.parameters}")
    
    # Call the batch evaluation method
    try:
        fmd_results = fmd_metric.evaluate_dataset(
            generated_dir=generated_dir,
            reference_dir=reference_dir,
            max_files=900,
            midi_extension=".mid"  # Can be parameterized if needed
        )
        
        # Save results to output directory if specified
        if output_dir:
            results_file = os.path.join(output_dir, "fmd_dataset_results.json")
            with open(results_file, "w") as f:
                json.dump(fmd_results, f, indent=2)
            print(f"Saved results to {results_file}")
        
        return fmd_results
        
    except Exception as e:
        import traceback
        error_msg = f"Error computing batch FMD: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return {"error": error_msg}


def create_quality_filtered_test_set(
    csv_path: str,
    midi_dir: str,
    output_dir: str,
    num_samples: int = 20,
    quality_thresholds: Optional[Dict[str, float]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    max_time: float = 30.0,
) -> dict:
    """
    Creates a test set of high-quality pieces by randomly shuffling and filtering based on quality metrics.

    Args:
        csv_path: Path to the CSV file containing piece metadata
        midi_dir: Directory containing MIDI files
        output_dir: Directory to save outputs
        num_samples: Number of samples to select (default: 20)
        quality_thresholds: Dictionary of metric names and minimum thresholds (on 0-1 scale)
        parameters: Parameters for the evaluation metrics
        max_time: Maximum time in seconds to keep from the MIDI file (default: 30 seconds)

    Returns:
        dict: Summary of the test set creation
    """
    # Verify file and directories exist
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail=f"CSV file not found: {csv_path}")

    if not os.path.exists(midi_dir):
        raise HTTPException(status_code=404, detail=f"MIDI directory not found: {midi_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load metrics configuration
    metrics_config = _load_metrics_config()

    # Define quality metrics to evaluate
    quality_metrics = ["n_pitches_used", "pitch_range", "polyphony", "empty_beat_rate", "scale_consistency", "pitch_class_entropy"]

    # Define default quality thresholds if not provided
    # These thresholds are now on a normalized 0-1 scale
    if quality_thresholds is None:
        quality_thresholds = {
            "n_pitches_used": 0.5,  # At least moderate pitch variety
            "pitch_range": 0.4,  # At least moderate pitch range
            "polyphony": 0.4,  # At least some polyphony
            "empty_beat_rate": 0.6,  # Not too many empty beats
            "scale_consistency": 0.7,  # Good scale consistency
            "pitch_class_entropy": 0.5,  # Good pitch class variety
        }

    # Use default parameters if none specified
    if parameters is None:
        parameters = metrics_config["default_parameters"]

    print(f"Loading dataset from {csv_path}...")

    # Load the CSV file
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return {"error": f"Failed to load CSV file: {str(e)}"}

    print(f"Loaded dataset with {len(df)} samples")

    # Get all valid MIDI files
    all_files = df["file"].tolist()
    valid_files = []

    # Check if files exist
    for midi_file in all_files:
        if os.path.exists(os.path.join(midi_dir, midi_file)):
            valid_files.append(midi_file)
        else:
            print(f"Warning: File not found, skipping: {midi_file}")

    if not valid_files:
        return {"error": "No valid MIDI files found"}

    print(f"Found {len(valid_files)} valid MIDI files")

    # Randomly shuffle the list of valid files
    random.shuffle(valid_files)

    # Create a set to store selected files
    selected_files = []
    rejected_files = []
    midi_evaluation_cache = {}  # Cache for MIDI evaluation results

    # Process files until we have enough that meet the quality thresholds
    for midi_file in valid_files:
        # Skip once we have enough samples
        if len(selected_files) >= num_samples:
            break

        midi_path = os.path.join(midi_dir, midi_file)

        # Check if we've already evaluated this MIDI file
        if midi_file in midi_evaluation_cache:
            eval_results = midi_evaluation_cache[midi_file]
        else:
            # Evaluate the MIDI file
            try:
                eval_results = evaluate(midi_path=midi_path, metrics=quality_metrics, parameters=parameters, max_time=max_time)
                midi_evaluation_cache[midi_file] = eval_results
            except Exception as e:
                print(f"Error evaluating {midi_file}: {str(e)}")
                rejected_files.append({"file": midi_file, "reason": f"Evaluation error: {str(e)}"})
                continue

        # Check for errors in evaluation results
        if any(isinstance(v, dict) and "error" in v for v in eval_results.values()):
            error_metrics = [k for k, v in eval_results.items() if isinstance(v, dict) and "error" in v]
            print(f"Warning: Errors in metrics for {midi_file}: {', '.join(error_metrics)}")
            rejected_files.append({"file": midi_file, "reason": f"Metric errors: {', '.join(error_metrics)}"})
            continue

        # Check if the file meets all quality thresholds
        passes_thresholds = True
        failed_metrics = []

        for metric, threshold in quality_thresholds.items():
            if metric in eval_results:
                value = eval_results[metric]

                # All metrics are now on 0-1 scale where higher is better
                if value < threshold:
                    passes_thresholds = False
                    failed_metrics.append(f"{metric}={value:.2f}<{threshold}")
            else:
                # If metric is missing, consider it a fail
                passes_thresholds = False
                failed_metrics.append(f"{metric}=missing")

        # Add the file to selected or rejected list
        if passes_thresholds:
            print(f"Selected {midi_file} (meets all quality thresholds)")
            selected_files.append({"file": midi_file, "metrics": eval_results})
        else:
            print(f"Rejected {midi_file} (failed metrics: {', '.join(failed_metrics)})")
            rejected_files.append({"file": midi_file, "reason": f"Failed metrics: {', '.join(failed_metrics)}", "metrics": eval_results})

    # Check if we found enough samples
    if len(selected_files) < num_samples:
        print(f"Warning: Only found {len(selected_files)} samples out of requested {num_samples}")

    # Extract just the filenames for the final CSV
    selected_filenames = [item["file"] for item in selected_files]

    # Create a filtered dataframe with selected files
    selected_df = df[df["file"].isin(selected_filenames)].copy()

    # Save the selected files to CSV
    output_csv_path = os.path.join(output_dir, "quality_filtered_test_set.csv")
    selected_df.to_csv(output_csv_path, index=False)

    # Save detailed metrics for selected files
    metrics_output_path = os.path.join(output_dir, "quality_filtered_metrics.json")
    with open(metrics_output_path, "w") as f:
        json.dump(selected_files, f, indent=2)

    # Save rejected files info
    rejected_output_path = os.path.join(output_dir, "rejected_files.json")
    with open(rejected_output_path, "w") as f:
        json.dump(rejected_files, f, indent=2)

    # Calculate average metrics for the selected files
    avg_metrics = {}
    for metric in quality_metrics:
        values = [item["metrics"].get(metric) for item in selected_files if metric in item["metrics"]]
        values = [v for v in values if isinstance(v, (int, float)) and not np.isnan(v)]

        if values:
            avg_metrics[metric] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "std": float(np.std(values)),
            }

    # Save average metrics
    avg_metrics_path = os.path.join(output_dir, "quality_filtered_avg_metrics.json")
    with open(avg_metrics_path, "w") as f:
        json.dump(avg_metrics, f, indent=2)

    print(f"\nCreated quality-filtered test set with {len(selected_files)} samples")
    print(f"Saved to {output_csv_path}")

    # Return summary
    return {"Message": f"Created quality-filtered test set"}


def create_balanced_test_set(
    csv_path: str,
    midi_dir: str,
    output_dir: str,
    num_samples_per_mood: int = 100,
    parameters: Optional[Dict[str, Any]] = None,
    max_time: float = 30.0,
) -> dict:
    """
    Creates a balanced test set based on mood labels and quality evaluation.

    Args:
        csv_path: Path to the CSV file containing mood labels
        midi_dir: Directory containing MIDI files
        output_dir: Directory to save outputs
        num_samples_per_mood: Number of samples to select per mood
        parameters: Parameters for the evaluation metrics
        max_time: Maximum time in seconds to keep from the MIDI file (default: 30 seconds)

    Returns:
        dict: Summary of the test set creation
    """
    # Verify file and directories exist
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail=f"CSV file not found: {csv_path}")

    if not os.path.exists(midi_dir):
        raise HTTPException(status_code=404, detail=f"MIDI directory not found: {midi_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load metrics configuration
    metrics_config = _load_metrics_config()

    # Define quality metrics to evaluate - all now normalized to 0-1 range
    quality_metrics = ["n_pitches_used", "pitch_range", "polyphony", "empty_beat_rate", "scale_consistency", "pitch_class_entropy"]

    # Use default parameters if none specified
    if parameters is None:
        parameters = metrics_config["default_parameters"]

    print(f"Loading dataset from {csv_path}...")

    # Load the CSV file
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return {"error": f"Failed to load CSV file: {str(e)}"}

    print(f"Loaded dataset with {len(df)} samples")

    # Extract all mood categories from the dataset
    moods = set()
    for mood_tokens in df["mood_tokens"]:
        mood_list = mood_tokens.split()
        # Extract the mood name from each token (format: Mood_Name_Value)
        for token in mood_list:
            parts = token.split("_")
            if len(parts) >= 2 and parts[0] == "Mood":
                moods.add(parts[1])

    print(f"Found {len(moods)} unique mood categories: {', '.join(moods)}")

    # Process each mood category
    mood_groups = {}
    midi_evaluation_cache = {}  # Cache for MIDI evaluation results

    for mood in moods:
        print(f"\nProcessing mood: {mood}")

        # Create a column for the target mood probability
        mood_col = f"Mood_{mood}_prob"
        df[mood_col] = 0.0

        # Extract the probability values for the target mood
        for i, mood_tokens in enumerate(df["mood_tokens"]):
            mood_list = mood_tokens.split()
            for token in mood_list:
                parts = token.split("_")
                if len(parts) >= 3 and parts[0] == "Mood" and parts[1] == mood:
                    try:
                        df.at[i, mood_col] = float(parts[2])
                    except ValueError:
                        pass

        # Calculate co-occurrence scores
        # We'll use a weighted sum of the target mood probability and co-occurrence patterns
        print(f"Calculating ranking scores for {mood}...")

        # First, analyze co-occurrence patterns for this mood
        co_occurrence = defaultdict(float)
        mood_count = 0

        for mood_tokens in df["mood_tokens"]:
            tokens = mood_tokens.split()
            has_target_mood = False
            other_moods = []

            for token in tokens:
                parts = token.split("_")
                if len(parts) >= 3 and parts[0] == "Mood":
                    if parts[1] == mood:
                        has_target_mood = True
                    else:
                        other_moods.append(parts[1])

            if has_target_mood:
                mood_count += 1
                for other_mood in other_moods:
                    co_occurrence[other_mood] += 1

        # Normalize co-occurrence counts
        if mood_count > 0:
            for other_mood in co_occurrence:
                co_occurrence[other_mood] /= mood_count

        # Calculate ranking score: combination of target mood probability and co-occurrence pattern
        df[f"{mood}_ranking_score"] = df[mood_col]

        # Sort by ranking score in descending order
        df_mood = df.sort_values(by=f"{mood}_ranking_score", ascending=False)

        # Take the top candidates (e.g., top 500) for further evaluation
        top_candidates = min(500, len(df_mood))
        df_mood_top = df_mood.head(top_candidates)

        print(f"Selected {top_candidates} top candidates based on {mood} probability")

        # Evaluate the quality of MIDI files
        quality_scores = []

        for idx, row in df_mood_top.iterrows():
            midi_file = row["file"]
            midi_path = os.path.join(midi_dir, midi_file)

            # Skip if file doesn't exist
            if not os.path.exists(midi_path):
                print(f"Warning: File not found: {midi_path}")
                quality_scores.append({"file": midi_file, "quality_score": 0, "error": "File not found"})
                continue

            # Check if we've already evaluated this MIDI file
            if midi_file in midi_evaluation_cache:
                eval_results = midi_evaluation_cache[midi_file]
            else:
                # Evaluate the MIDI file
                try:
                    eval_results = evaluate(midi_path=midi_path, metrics=quality_metrics, parameters=parameters, max_time=max_time)
                    midi_evaluation_cache[midi_file] = eval_results
                except Exception as e:
                    print(f"Error evaluating {midi_file}: {str(e)}")
                    quality_scores.append({"file": midi_file, "quality_score": 0, "error": str(e)})
                    continue

            # Calculate a quality score based on the evaluation metrics
            # Since all metrics are now normalized to 0-1 range, calculation is simpler
            quality_score = 0
            error_found = False

            for metric in eval_results:
                if isinstance(eval_results[metric], dict) and "error" in eval_results[metric]:
                    error_found = True
                    break

            if error_found:
                quality_scores.append({"file": midi_file, "quality_score": 0, "error": "Evaluation error"})
                continue

            # Normalize and combine metrics to create a quality score
            try:
                # Define relative importance weights for each metric (must sum to 1.0)
                weights = {
                    "n_pitches_used": 0.2,  # Pitch variety is important
                    "pitch_range": 0.15,  # Good range adds interest
                    "polyphony": 0.15,  # Texture richness
                    "empty_beat_rate": 0.15,  # Rhythm density
                    "scale_consistency": 0.2,  # Tonal coherence
                    "pitch_class_entropy": 0.15,  # Pitch class variety
                }

                # Calculate weighted score - all metrics are already in 0-1 range where higher is better
                for metric, weight in weights.items():
                    if metric in eval_results and isinstance(eval_results[metric], (int, float)) and not np.isnan(eval_results[metric]):
                        # Simple weighted sum
                        quality_score += eval_results[metric] * weight * 10  # Scale to 0-10 range

                quality_scores.append({"file": midi_file, "quality_score": quality_score, "eval_results": eval_results})

            except Exception as e:
                print(f"Error calculating quality score for {midi_file}: {str(e)}")
                quality_scores.append({"file": midi_file, "quality_score": 0, "error": str(e)})

        # Sort by quality score in descending order
        quality_scores.sort(key=lambda x: x.get("quality_score", 0), reverse=True)

        # Select the top N MIDI files
        top_quality_files = [item["file"] for item in quality_scores[:num_samples_per_mood]]

        # Filter the dataframe to get only the selected files
        mood_df = df[df["file"].isin(top_quality_files)].copy()

        # Store the mood group
        mood_groups[mood] = mood_df

        # Save to CSV
        mood_csv_path = os.path.join(output_dir, f"{mood}_test_set.csv")
        mood_df.to_csv(mood_csv_path, index=False)

        print(f"Created test set for {mood} with {len(mood_df)} samples")
        print(f"Saved to {mood_csv_path}")

    # Create a combined CSV with all mood groups
    combined_df = pd.concat(mood_groups.values())
    combined_df = combined_df.drop_duplicates(subset=["file"])
    combined_csv_path = os.path.join(output_dir, "ReMIDICaps_test_set.csv")
    combined_df.to_csv(combined_csv_path, index=False)

    print(f"\nCreated combined test set with {len(combined_df)} unique samples")
    print(f"Saved to {combined_csv_path}")

    # Return summary
    return {
        "Message": f"Created balanced test set with {len(combined_df)} unique samples across {len(moods)} mood categories",
        "moods": list(moods),
        "samples_per_mood": {mood: len(df) for mood, df in mood_groups.items()},
        "total_unique_samples": len(combined_df),
        "output_dir": output_dir,
    }
