# Standard library imports
import asyncio
import os

# MIDI processing
import pretty_midi as pm

# Local application imports - REMI encoding
from src.application.encoder.helpers.remi_helper import get_remi_events
from src.application.encoder.models.item_model import Item
from src.application.encoder.helpers.encoder_helper import (
    read_note_tempo,
    quantize_midi,
    extract_beats,
    extract_downbeats,
    group_items,
    extract_dominant_keys,
)

# Local application imports - Music analysis
from src.application.music_base.services.music_base_service import (
    extract_chords,
    estimate_tonal_plan,
)

# Domain models and parameters
from src.domain.models.encoder.encoder_model import (
    EncodeParameters,
    EncodeDatasetParameters,
)
from src.domain.models.music_base.music_base_model import (
    TonalPlanParameters,
    MusicBaseParameters,
)

# Constants - File paths
from src.domain.constants.paths_constants import (
    ENCODINGS_PATH,
    MIDI_PATH,
    CHORDS_PATH,
    KEYS_PATH,
)

# Data persistence
from src.persistence.dataloader.repositories.dataloader_repository import (
    save_async,
    async_load,
)


def encode_midi(parameters: EncodeParameters) -> dict:
    """
    Encode a MIDI file into a symbolic representation with harmonic analysis.

    This function processes a MIDI file through several stages:
    1. MIDI Loading and Basic Processing:
       - Loads and quantizes MIDI data
       - Extracts timing information (beats and downbeats)
    2. Harmonic Analysis:
       - Extracts chord progressions
       - Analyzes tonal plan (key changes)
    3. REMI (REvamped MIDI) Encoding:
       - Combines notes, chords, and keys into a unified representation
       - Groups events by musical bars
       - Creates a sequence of musical events

    Args:
        parameters (EncodeParameters): Configuration parameters including:
            - midi (str or PrettyMIDI): MIDI file path or loaded MIDI object
            - alpha, beta, gamma (float): Key detection parameters
            - c, w (float): Tonal plan estimation parameters
            - save (bool): Whether to save the encoded output
            - encodings_out_dir (str): Directory for saving encodings
            - chords_out_dir (str): Directory for saving chord analysis
            - keys_out_dir (str): Directory for saving key analysis

    Returns:
        dict: Status message indicating successful encoding

    Note:
        The function caches intermediate results (chords and keys) to avoid
        redundant computation when processing the same MIDI file multiple times.
    """
    # Step 1: Load MIDI file
    # Convert file path to PrettyMIDI object if needed
    if isinstance(parameters.midi, str):
        midi = pm.PrettyMIDI(parameters.midi)
    else:
        midi = parameters.midi
    sample = {}  # Will store the final encoded representation

    # Step 2: Extract basic MIDI features
    # Get note and tempo events from the MIDI file
    note_items, tempo_items = read_note_tempo(midi)
    # Quantize note timings to align with musical grid
    quantize_midi(midi, note_items, midi.resolution)
    # Extract beat and bar information
    beats = extract_beats(midi)  # Get all beat positions
    downbeats = extract_downbeats(midi)  # Get bar start positions

    # Step 3: Harmonic Analysis - Chord Extraction
    # Check if chord analysis already exists
    chords_path = os.path.join(parameters.chords_out_dir, f"{os.path.basename(parameters.midi)}_chords.pkl")
    if os.path.isfile(chords_path):
        # Load existing chord analysis
        chords = async_load(parameters.chords_out_dir, parameters.midi, "chords")
    else:
        # Perform new chord analysis
        chords = extract_chords(
            MusicBaseParameters(
                midi=parameters.midi,
                save=parameters.save,
                out_dir=parameters.chords_out_dir,
            ),
        )

    # Step 4: Harmonic Analysis - Key Detection
    # Check if key analysis already exists
    keys_path = os.path.join(parameters.keys_out_dir, f"{os.path.basename(parameters.midi)}_keys.pkl")
    if os.path.isfile(keys_path):
        # Load existing key analysis
        keys = async_load(parameters.keys_out_dir, parameters.midi, "keys")
    else:
        # Perform new key analysis with tonal plan estimation
        keys = estimate_tonal_plan(
            TonalPlanParameters(
                midi=parameters.midi,
                alpha=parameters.alpha,  # Weight for chord-key relationships
                beta=parameters.beta,    # Weight for key transition smoothness
                gamma=parameters.gamma,   # Weight for local key stability
                c=parameters.c,          # Confidence threshold
                w=parameters.w,          # Window size for key detection
                save=parameters.save,
                out_dir=parameters.keys_out_dir,
                chords=chords,
            ),
        )

    # Step 5: Convert chord analysis to REMI format
    if "remi_chords" in chords:
        # Use existing REMI chord representation
        remi_chords = chords["remi_chords"]
    else:
        # Convert chord analysis to REMI format
        remi_chords = []
        for chord in chords["chords"]:
            remi_chords.append(
                Item(
                    name="Chord",
                    start=midi.time_to_tick(chord[0]),  # Convert time to ticks
                    end=midi.time_to_tick(chord[1]),    # Convert time to ticks
                    velocity=None,                      # Chords don't have velocity
                    pitch=chord[2].split("/")[0],       # Extract chord root/quality
                )
            )

        # Handle case where piece starts without a chord
        if len(remi_chords) == 0 or remi_chords[0].start > 0:
            # Add a "no chord" marker from start to first chord (or end if no chords)
            end = midi.time_to_tick(midi.get_end_time()) if len(remi_chords) == 0 else remi_chords[0].start
            remi_chords.append(Item(name="Chord", start=0, end=end, velocity=None, pitch="N:N"))

        # Save the REMI chord representation for future use
        chords_out = {"chords": chords["chords"], "remi_chords": remi_chords}
        save_async(parameters.chords_out_dir, parameters.midi, chords_out, "chords")

    # Step 6: Convert key analysis to REMI format
    if "remi_keys" in keys:
        # Use existing REMI key representation
        remi_keys = keys["remi_keys"]
    else:
        # Convert key analysis to REMI format
        remi_keys = []
        # Create key events for each beat interval
        for i in range(1, len(beats), 1):
            remi_keys.append(
                Item(
                    name="Key",
                    start=midi.time_to_tick(beats[i - 1]),  # Start at previous beat
                    end=midi.time_to_tick(beats[i]),        # End at current beat
                    velocity=None,                          # Keys don't have velocity
                    pitch=f"{keys['keys'][i-1][0]}:{keys['keys'][i-1][1]}",  # Format: root:mode
                )
            )

        # Save the REMI key representation for future use
        keys_out = {"keys": keys["keys"], "remi_keys": remi_keys}
        save_async(parameters.keys_out_dir, parameters.midi, keys_out, "keys")

    # Step 7: Combine all musical events
    # Attach key analysis to MIDI object for REMI encoding
    midi.tonal_plan = remi_keys

    # Combine all musical events in order: keys, chords, tempo, notes
    items = remi_keys + remi_chords + tempo_items + note_items
    # Group items by musical bars using downbeat positions
    groups = group_items(midi, downbeats, items=items)
    # Extract dominant keys for each group
    groups = extract_dominant_keys(groups)
    # Generate final REMI event sequence
    sample["events"] = get_remi_events(midi, groups)[1]

    # Step 8: Save encoded representation
    if parameters.save:
        save_async(parameters.encodings_out_dir, parameters.midi, sample, "encoding")

    return {"Message": "Midi Encoded Successfully"}


async def encode_dataset(parameters: EncodeDatasetParameters) -> dict:
    """
    Asynchronously encode an entire dataset of MIDI files with error handling and batch processing.

    This function processes a collection of MIDI files through several stages:
    1. Directory Setup:
       - Creates output directories for encodings, chords, and keys
       - Sets up a dump directory for failed files
    2. Batch Processing:
       - Processes files in small batches to manage memory
       - Handles errors gracefully by moving failed files to dump directory
    3. Progress Tracking:
       - Maintains counts of processed and successful files
       - Collects error messages for failed files

    Args:
        parameters (EncodeDatasetParameters): Configuration parameters including:
            - dataset_name (str): Name of the dataset to process
            - alpha, beta, gamma (float): Key detection parameters
            - c, w (float): Tonal plan estimation parameters
            - batch_size (int): Number of files to process in parallel

    Returns:
        dict: Processing summary including:
            - Total number of files processed
            - Number of successful encodings
            - Number of failed files
            - List of error messages

    Note:
        Files that fail to process (due to corruption, invalid format, etc.)
        are automatically moved to a 'dump' directory for later inspection.
    """
    # Step 1: Set up directory paths
    # Base paths for input and output
    dataset_path = os.path.join(MIDI_PATH, parameters.dataset_name)  # Source MIDI files
    encodings_out_dir = os.path.join(ENCODINGS_PATH, parameters.dataset_name)  # Encoded outputs
    chords_out_dir = os.path.join(CHORDS_PATH, parameters.dataset_name)  # Chord analysis
    keys_out_dir = os.path.join(KEYS_PATH, parameters.dataset_name)  # Key analysis

    # Create dump directory for failed files
    dump_dir = os.path.join(dataset_path, "dump")
    os.makedirs(dump_dir, exist_ok=True)

    async def process_file(file_path: str) -> tuple[bool, str]:
        """
        Process a single MIDI file with error handling.

        Args:
            file_path (str): Path to the MIDI file to process

        Returns:
            tuple[bool, str]: Success status and error message (if any)
                - First element: True if successful, False if failed
                - Second element: Error message if failed, empty string if successful
        """
        try:
            # Validate MIDI file by attempting to load it
            pm.PrettyMIDI(file_path)

            # Encode the MIDI file with specified parameters
            await encode_midi(
                EncodeParameters(
                    midi=file_path,
                    alpha=parameters.alpha,  # Weight for chord-key relationships
                    beta=parameters.beta,    # Weight for key transition smoothness
                    gamma=parameters.gamma,   # Weight for local key stability
                    c=parameters.c,          # Confidence threshold
                    w=parameters.w,          # Window size for key detection
                    save=True,               # Always save results for dataset processing
                    encodings_out_dir=encodings_out_dir,
                    chords_out_dir=chords_out_dir,
                    keys_out_dir=keys_out_dir,
                ),
            )
            return True, ""  # Indicate successful processing
        except Exception as e:
            # Handle any errors during processing
            filename = os.path.basename(file_path)
            dump_path = os.path.join(dump_dir, filename)
            # Move failed file to dump directory for later inspection
            os.rename(file_path, dump_path)
            return False, f"Error processing {filename}: {str(e)}"

    # Step 2: Prepare file list and initialize counters
    # Get all MIDI files in the dataset directory
    midi_files = [f for f in os.listdir(dataset_path) if f.endswith((".mid", ".midi"))]
    total_files = len(midi_files)  # Total number of files to process
    processed = 0  # Counter for processed files
    successful = 0  # Counter for successfully processed files
    errors = []  # List to collect error messages

    # Step 3: Process files in batches
    batch_size = 10  # Number of files to process in parallel
    while processed < total_files:
        # Extract next batch of files
        batch = midi_files[processed : processed + batch_size]
        # Create processing tasks for the batch
        batch_tasks = [process_file(os.path.join(dataset_path, file)) for file in batch]

        # Process batch asynchronously
        results = await asyncio.gather(*batch_tasks, return_exceptions=False)

        # Update processing statistics
        for success, error_msg in results:
            if success:
                successful += 1  # Increment successful count
            else:
                errors.append(error_msg)  # Collect error message

        processed += len(batch)  # Update total processed count

    return {"Message": "Dataset Encoded Successfully"}
