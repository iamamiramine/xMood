import asyncio
import os
import json
import time

import pretty_midi as pm
import pandas as pd
import music21
import torch

from application.encoder.helpers.remi_helper import get_remi_events
from domain.models.encoder.encoder_model import (
    EncodeParameters,
    EncodeDatasetParameters,
    TokenizeRemiDatasetParameters,
)
from application.music_base.services.music_base_service import (
    extract_chords,
)
from application.encoder.models.item_model import Item
from domain.models.music_base.music_base_model import (
    MusicBaseParameters,
)
from application.encoder.helpers.encoder_helper import (
    read_note_tempo,
    quantize_midi,
    extract_beats,
    extract_downbeats,
    group_items,
    extract_dominant_keys,
)
from application.encoder.models.vocab_model import RemiVocab

from domain.constants.paths_constants import (
    MIDI_PATH,
    PROCESSED_PATH,
    LABELS_PATH,
)
from persistence.dataloader.repositories.dataloader_repository import (
    save_async,
    async_load,
)


def convert_dataset_key(key: str) -> str:
    """Convert dataset key format (e.g. 'F# minor' or 'Eb major') to REMI format (e.g. 'F#:min' or 'D#:maj')

    Handles both sharp and flat notations, converting flats to their enharmonic sharp equivalents.
    """
    # Mapping of flat notes to their enharmonic sharp equivalents
    flat_to_sharp = {
        "Db": "C#",
        "Eb": "D#",
        "Gb": "F#",
        "Ab": "G#",
        "Bb": "A#",
    }

    # Split into root and mode
    parts = key.split(" ")
    root = parts[0]
    mode = parts[1]

    # Convert flat notation to sharp if needed
    if "b" in root:
        root = flat_to_sharp.get(root, root)

    # Convert mode to short form
    mode_map = {"major": "maj", "minor": "min"}
    short_mode = mode_map[mode]

    return f"{root}:{short_mode}"


def encode_midi(parameters: EncodeParameters) -> dict:
    # Load MIDI file asynchronously
    if isinstance(parameters.midi, str):
        midi = pm.PrettyMIDI(parameters.midi)
    else:
        midi = parameters.midi
    sample = {}

    # Determine the processed directory
    if parameters.encodings_out_dir:
        processed_dir = parameters.encodings_out_dir
    else:
        raise ValueError("processed_dir must be provided")

    # Try to load processed data if it exists
    try:
        processed_data = async_load(processed_dir, parameters.midi, "processed")
        if "encodings" in processed_data:
            return {"Message": "Midi Already Encoded"}
    except:
        processed_data = {}

    note_items, tempo_items = read_note_tempo(midi)
    quantize_midi(midi, note_items, midi.resolution)
    beats = extract_beats(midi)
    downbeats = extract_downbeats(midi)

    # Check if we need to extract chords and keys
    if "chords" not in processed_data:
        chords = extract_chords(
            MusicBaseParameters(
                midi=parameters.midi,
                save=False,
            ),
        )

        # Generate REMI chords
        remi_chords = []
        for chord in chords["chords"]:
            remi_chords.append(
                Item(
                    name="Chord",
                    start=midi.time_to_tick(chord[0]),
                    end=midi.time_to_tick(chord[1]),
                    velocity=None,
                    pitch=chord[2].split("/")[0],
                )
            )

        if len(remi_chords) == 0 or remi_chords[0].start > 0:
            end = midi.time_to_tick(midi.get_end_time()) if len(remi_chords) == 0 else remi_chords[0].start
            remi_chords.append(Item(name="Chord", start=0, end=end, velocity=None, pitch="N:N"))

        processed_data["chords"] = {"chords": chords["chords"], "remi_chords": remi_chords, "expanded_chords": chords["expanded_chords"]}

    # Use music21's key detection
    if isinstance(midi, str):
        midi_file = music21.converter.parse(midi)
    else:
        # Extract filename from the MIDI path or generate a unique name
        if hasattr(midi, "filename"):
            filename = os.path.basename(midi.filename)
            base_filename = os.path.splitext(filename)[0]
        else:
            # If no filename available, use a timestamp
            base_filename = str(int(time.time() * 1000))

        # Create temporary file with unique name
        temp_path = f"temp_midi_{base_filename}.mid"
        midi.write(temp_path)
        midi_file = music21.converter.parse(temp_path)
        os.remove(temp_path)

    # Analyze the key using music21
    key_analysis = midi_file.analyze("key")
    mode = "maj" if key_analysis.mode == "major" else "min"
    remi_key = f"{key_analysis.tonic.name}:{mode}"

    # Create key items for each beat
    remi_keys = []
    for i in range(1, len(beats), 1):
        remi_keys.append(
            Item(
                name="Key",
                start=midi.time_to_tick(beats[i - 1]),
                end=midi.time_to_tick(beats[i]),
                velocity=None,
                pitch=remi_key,
            )
        )

        processed_data["keys"] = {"keys": [], "remi_keys": remi_keys}

    # Get REMI events from processed data
    remi_chords = processed_data["chords"]["remi_chords"]
    remi_keys = processed_data["keys"]["remi_keys"]

    midi.tonal_plan = remi_keys

    items = remi_keys + remi_chords + tempo_items + note_items
    groups = group_items(midi, downbeats, items=items)
    groups = extract_dominant_keys(groups)
    sample["events"] = get_remi_events(midi, groups)[1]

    # Add encoding to processed data
    if "encodings" not in processed_data:
        processed_data["encodings"] = {}
    processed_data["encodings"] = sample

    # Save processed data
    if parameters.save:
        save_async(processed_dir, parameters.midi, processed_data, "processed")

    return {"Message": "Midi Encoded Successfully"}


async def encode_dataset(parameters: EncodeDatasetParameters) -> dict:
    """
    Encode a dataset of MIDI files using BaseModel parameters.

    Args:
        parameters: EncodeDatasetParameters containing all configuration
    """
    
    # Set up paths
    dataset_path = MIDI_PATH
    processed_dir = os.path.join(PROCESSED_PATH, parameters.dataset_name)
    if parameters.encodings_out_dir:
        processed_dir = parameters.encodings_out_dir

    # Create output directory if it doesn't exist
    os.makedirs(processed_dir, exist_ok=True)

    # Load dataset CSV file if available
    dataset_csv = os.path.join(LABELS_PATH, f"{parameters.dataset_name}.csv")
    key_map = {}
    if os.path.exists(dataset_csv):
        df = pd.read_csv(dataset_csv)
        # Create a mapping of filename to key if key column exists
        if "key" in df.columns:
            key_map = dict(zip(df["file"].apply(lambda x: os.path.basename(x)), df.get("key")))

    async def process_file(file_path: str) -> tuple[bool, str]:
        try:
            # Test if MIDI file can be loaded
            pm.PrettyMIDI(file_path)

            # Check if already processed and not overwriting
            filename = os.path.basename(file_path)
            processed_file = os.path.join(processed_dir, f"{filename}_processed.pkl")
            if os.path.exists(processed_file) and not parameters.overwrite_existing:
                return True, f"Skipped {filename} (already exists)"

            # Get the key from the dataset if available
            dataset_key = key_map.get(filename) if key_map else None

            encode_midi(
                EncodeParameters(
                    midi=file_path,
                    alpha=parameters.alpha,
                    beta=parameters.beta,
                    gamma=parameters.gamma,
                    c=parameters.c,
                    w=parameters.w,
                    save=parameters.save,
                    encodings_out_dir=processed_dir,
                    dataset_key=dataset_key,
                ),
            )
            return True, ""
        except Exception as e:
            filename = os.path.basename(file_path)
            if parameters.skip_invalid:
                print(f"Skipping invalid file {filename}: {str(e)}", flush=True)
                return False, f"Skipped invalid file {filename}: {str(e)}"
            else:
                return False, f"Error processing {filename}: {str(e)}"

    # Get MIDI files from the dataset path
    midi_files = [f for f in os.listdir(dataset_path) if f.endswith((".mid", ".midi"))]
    
    # Limit files if specified
    if parameters.max_files:
        midi_files = midi_files[:parameters.max_files]
    
    # Resume from specific file if specified
    if parameters.resume_from:
        try:
            start_index = midi_files.index(parameters.resume_from)
            midi_files = midi_files[start_index:]
        except ValueError:
            print(f"Warning: Resume file {parameters.resume_from} not found, starting from beginning")

    total_files = len(midi_files)
    processed = 0
    successful = 0
    errors = []

    # Process files in batches
    while processed < total_files:
        batch = midi_files[processed : processed + parameters.batch_size]
        batch_tasks = [process_file(os.path.join(dataset_path, file)) for file in batch]

        # Process batch
        results = await asyncio.gather(*batch_tasks, return_exceptions=False)

        # Update counters
        for success, error_msg in results:
            if success:
                successful += 1
            else:
                errors.append(error_msg)

        processed += len(batch)

    return {"Message": "Successfully encoded dataset"}


async def tokenize_remi_dataset(parameters: TokenizeRemiDatasetParameters) -> dict:
    """
    Tokenize the encoded REMI sequences and save them as PyTorch tensors.

    Args:
        parameters: TokenizeRemiDatasetParameters containing all configuration

    Returns:
        dict: Summary of the tokenization process
    """
    
    # Set up paths
    dataset_path = MIDI_PATH
    processed_dir = os.path.join(PROCESSED_PATH, parameters.dataset_name)
    remi_path = os.path.join(PROCESSED_PATH, f"{parameters.dataset_name}_tokenized")
    if parameters.tokens_out_dir:
        remi_path = parameters.tokens_out_dir

    # Create the output directory if it doesn't exist
    os.makedirs(remi_path, exist_ok=True)

    # Load dataset CSV file
    dataset_csv = os.path.join(LABELS_PATH, f"{parameters.dataset_name}.csv")
    if not os.path.exists(dataset_csv):
        print(f"Warning: Dataset CSV file not found: {dataset_csv}. Processing all MIDI files in directory.")
        midi_files = [f for f in os.listdir(dataset_path) if f.endswith((".mid", ".midi"))]
    else:
        df = pd.read_csv(dataset_csv)
        # ONLY get the MIDI files listed in the CSV file
        csv_midi_files = [os.path.basename(file) for file in df["file"]]
        # Filter to only include files that exist in the dataset directory
        existing_midi_files = set(os.listdir(dataset_path))
        midi_files = [file for file in csv_midi_files if file in existing_midi_files]

    if not midi_files:
        return {"Message": "No MIDI files found for tokenization"}

    # Initialize vocabulary for tokenization
    vocab = RemiVocab()
    
    # Load custom vocabulary if specified
    if parameters.vocab_path and os.path.exists(parameters.vocab_path):
        import pickle
        with open(parameters.vocab_path, 'rb') as f:
            vocab = pickle.load(f)

    # Limit files if specified
    if parameters.max_files:
        midi_files = midi_files[:parameters.max_files]
    
    # Resume from specific file if specified
    if parameters.resume_from:
        try:
            start_index = midi_files.index(parameters.resume_from)
            midi_files = midi_files[start_index:]
        except ValueError:
            print(f"Warning: Resume file {parameters.resume_from} not found, starting from beginning")

    total_files = len(midi_files)
    processed = 0
    successful = 0
    errors = []

    async def process_file(file_path: str) -> tuple[bool, str]:
        try:
            # Get the filename
            midi_file = os.path.basename(file_path)
            
            # Check if already processed and not overwriting
            remi_fn = os.path.join(remi_path, midi_file).replace(".mid", ".pt")
            if os.path.exists(remi_fn) and not parameters.overwrite_existing:
                return True, f"Skipped {midi_file} (already exists)"

            # Load processed data
            try:
                processed_data = async_load(processed_dir, midi_file, "processed")
            except Exception as e:
                return False, f"Could not load processed data for {midi_file}: {str(e)}"

            # Check if the file has been encoded
            if "encodings" not in processed_data or "events" not in processed_data["encodings"]:
                return False, f"No encodings found for {midi_file}"

            # Get the REMI events
            events = processed_data["encodings"]["events"]

            # Apply context size if specified
            if parameters.context_size > 0:
                events = events[:parameters.context_size]

            # Convert string tokens to integers using the vocabulary
            token_ids = torch.tensor(vocab.encode(events), dtype=torch.long)

            # Save as PyTorch tensor
            torch.save(token_ids, remi_fn)

            return True, ""
        except Exception as e:
            filename = os.path.basename(file_path)
            if parameters.skip_invalid:
                return False, f"Skipped invalid file {filename}: {str(e)}"
            else:
                return False, f"Error processing {filename}: {str(e)}"

    # Process files in batches
    while processed < total_files:
        batch = midi_files[processed : processed + parameters.batch_size]
        batch_tasks = [process_file(os.path.join(dataset_path, file)) for file in batch]

        # Process batch
        results = await asyncio.gather(*batch_tasks, return_exceptions=False)

        # Update counters
        for success, error_msg in results:
            if success:
                successful += 1
            else:
                errors.append(error_msg)

        processed += len(batch)

    return {"Message": "Successfully tokenized REMI dataset"}
