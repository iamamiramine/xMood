import asyncio
import os
import json
import time

import pretty_midi as pm
import pandas as pd
import music21

from application.encoder.helpers.remi_helper import get_remi_events
from domain.models.encoder.encoder_model import (
    EncodeParameters,
)
from application.music_base.services.music_base_service import (
    extract_chords,
    estimate_tonal_plan,
)
from application.encoder.models.item_model import Item
from domain.models.music_base.music_base_model import (
    TonalPlanParameters,
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

    if "keys" not in processed_data:
        if parameters.use_algorithm:
            # Use the key detection algorithm
            keys = estimate_tonal_plan(
                TonalPlanParameters(
                    midi=parameters.midi,
                    alpha=parameters.alpha,
                    beta=parameters.beta,
                    gamma=parameters.gamma,
                    c=parameters.c,
                    w=parameters.w,
                    save=False,
                    chords=processed_data["chords"],
                ),
            )

            # Generate REMI keys
            remi_keys = []
            for i in range(1, len(beats), 1):
                remi_keys.append(
                    Item(
                        name="Key",
                        start=midi.time_to_tick(beats[i - 1]),
                        end=midi.time_to_tick(beats[i]),
                        velocity=None,
                        pitch=f"{keys['keys'][i-1][0]}:{keys['keys'][i-1][1]}",
                    )
                )
        elif parameters.use_music21:
            # Use music21's key detection
            if isinstance(midi, str):
                midi_file = music21.converter.parse(midi)
            else:
                # Extract filename from the MIDI path or generate a unique name
                if hasattr(midi, 'filename'):
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
            key_analysis = midi_file.analyze('key')
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
        else:
            # Use the key from the dataset
            if not parameters.dataset_key:
                raise ValueError("dataset_key must be provided when use_algorithm and use_music21 are False")

            # Convert dataset key format to REMI format
            remi_key = convert_dataset_key(parameters.dataset_key)

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

            # # Create a single key that spans the entire piece
            # end_tick = midi.time_to_tick(midi.get_end_time())
            # remi_keys = [Item(name="Key", start=0, end=end_tick, velocity=None, pitch=remi_key)]

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


async def encode_dataset(config_path: str) -> dict:
    """
    Encode a dataset of MIDI files using parameters from the config file.

    Args:
        config_path: Path to the configuration file
    """
    # Load configuration
    with open(config_path, "r") as f:
        config = json.load(f)

    encoder_config = config.get("encoder", {})
    dataset_name = config.get("dataloader", {}).get("dataset_name")

    if not dataset_name:
        raise ValueError("dataset_name must be provided in the dataloader config")

    dataset_path = MIDI_PATH
    processed_dir = os.path.join(PROCESSED_PATH, dataset_name)

    # Load dataset CSV file
    dataset_csv = os.path.join(LABELS_PATH, "MIDICaps.csv")
    if not os.path.exists(dataset_csv):
        raise ValueError(f"Dataset CSV file not found: {dataset_csv}")

    df = pd.read_csv(dataset_csv)
    # Create a mapping of filename to key
    key_map = dict(zip(df["file"].apply(lambda x: os.path.basename(x)), df["key"]))

    async def process_file(file_path: str) -> tuple[bool, str]:
        try:
            # Test if MIDI file can be loaded
            pm.PrettyMIDI(file_path)

            # Get the key from the dataset
            filename = os.path.basename(file_path)
            dataset_key = key_map.get(filename)
            if not dataset_key:
                raise ValueError(f"Key not found in dataset for file: {filename}")

            encode_midi(
                EncodeParameters(
                    midi=file_path,
                    alpha=encoder_config.get("alpha", 0.016),
                    beta=encoder_config.get("beta", 0.3),
                    gamma=encoder_config.get("gamma", 0.4),
                    c=encoder_config.get("c", 12),
                    w=encoder_config.get("w", 4),
                    save=True,
                    encodings_out_dir=processed_dir,
                    use_algorithm=False,
                    use_music21=True,
                    dataset_key=dataset_key,
                ),
            )
            return True, ""
        except Exception as e:
            # Move file to dump directory
            filename = os.path.basename(file_path)
            print("Error processing", filename, str(e), flush=True)
            return False, f"Error processing {filename}: {str(e)}"

    # Create tasks for each file
    midi_files = [f for f in os.listdir(dataset_path) if f.endswith((".mid", ".midi"))]
    total_files = len(midi_files)
    processed = 0
    successful = 0
    errors = []

    # Process files in batches using batch size from config
    batch_size = encoder_config.get("batch_size", 10)
    while processed < total_files:
        batch = midi_files[processed : processed + batch_size]
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

    # Prepare summary message
    summary = f"Dataset Encoding Complete\n" f"Total files: {total_files}\n" f"Successfully processed: {successful}\n" f"Failed: {len(errors)}\n"
    if errors:
        summary += "\nErrors:\n" + "\n".join(errors)

    return {"Message": summary}
