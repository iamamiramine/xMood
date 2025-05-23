from fastapi import APIRouter, HTTPException, BackgroundTasks
import os
from application.evaluation.services import evaluation_service
from typing import Dict, List, Any, Optional

router = APIRouter()


@router.get("/metrics")
def get_metrics() -> Dict[str, List[Dict[str, Any]]]:
    """
    Description:
    ------------
        Get all available metrics.

    Returns:
    --------
    dict
        Dictionary of available metrics
    """
    return evaluation_service.get_available_metrics()



@router.post("/evaluate-dataset")
async def evaluate_midi_dataset(config_path: str, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """
    Description:
    ------------
        Evaluate a dataset of MIDI files according to configuration.
        This operation is performed asynchronously in the background.

    Parameters:
    -----------
        config_path: str
            Path to the configuration file with evaluation settings

    Returns:
    --------
    dict
        A message indicating that the evaluation has started
    """
    # Verify config file exists
    if not os.path.exists(config_path):
        raise HTTPException(status_code=404, detail=f"Configuration file not found: {config_path}")

    # Start background task to process the dataset
    background_tasks.add_task(evaluation_service.evaluate_dataset, config_path)

    return {"message": "Dataset evaluation started in the background. Results will be saved to the specified output directory."}


@router.post("/create-balanced-test-set")
def create_balanced_test(
    csv_path: str,
    midi_dir: str,
    output_dir: str,
    num_samples_per_mood: int = 100,
    parameters: Optional[Dict[str, Any]] = None,
    max_time: float = 30.0,
) -> Dict[str, Any]:
    """
    Description:
    ------------
        Creates a balanced test set based on mood labels and quality evaluation.
        This operation is performed asynchronously in the background.

    Parameters:
    -----------
        csv_path: str
            Path to the CSV file containing mood labels
        midi_dir: str
            Directory containing MIDI files
        output_dir: str
            Directory to save outputs
        num_samples_per_mood: int
            Number of samples to select per mood (default: 100)
        parameters: Optional[Dict[str, Any]]
            Parameters for the evaluation metrics
        max_time: float
            Maximum time in seconds to keep from the MIDI file (default: 30 seconds)

    Returns:
    --------
    dict
        A message indicating that the test set creation has started
    """

    evaluation_service.create_balanced_test_set(
        csv_path=csv_path,
        midi_dir=midi_dir,
        output_dir=output_dir,
        num_samples_per_mood=num_samples_per_mood,
        parameters=parameters,
        max_time=max_time,
    )

    return {
        "message": (
            f"Balanced test set creation started in the background. "
            f"Selecting {num_samples_per_mood} samples per mood category. "
            f"Results will be saved to {output_dir}."
        )
    }



@router.post("/create-quality-filtered-test-set")
def create_quality_filtered_test(
    csv_path: str,
    midi_dir: str,
    output_dir: str,
    num_samples: int = 20,
    parameters: Optional[Dict[str, Any]] = None,
    max_time: float = 30.0,
) -> Dict[str, Any]:
    """
    Description:
    ------------
        Creates a balanced test set based on mood labels and quality evaluation.
        This operation is performed asynchronously in the background.

    Parameters:
    -----------
        csv_path: str
            Path to the CSV file containing mood labels
        midi_dir: str
            Directory containing MIDI files
        output_dir: str
            Directory to save outputs
        num_samples: int
            Number of samples to select (default: 20)
        parameters: Optional[Dict[str, Any]]
            Parameters for the evaluation metrics
        max_time: float
            Maximum time in seconds to keep from the MIDI file (default: 30 seconds)

    Returns:
    --------
    dict
        A message indicating that the test set creation has started
    """

    evaluation_service.create_quality_filtered_test_set(
        csv_path=csv_path,
        midi_dir=midi_dir,
        output_dir=output_dir,
        num_samples=num_samples,
        parameters=parameters,
        max_time=max_time,
    )

    return {"message": "Quality filtered test set creation started in the background."}


@router.post("/evaluate-dataset-fmd")
async def evaluate_dataset_fmd(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """
    Description:
    ------------
        Evaluate a dataset of MIDI files using the batch FMD implementation.
        This operation is performed asynchronously in the background.

    Returns:
    --------
    dict
        A message indicating that the evaluation has started
    """

    # Start background task to process the dataset
    background_tasks.add_task(evaluation_service.evaluate_dataset_fmd)

    return {"message": "FMD Dataset evaluation started in the background. Results will be saved to the specified output directory."}
