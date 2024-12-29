import asyncio
import os
from collections import Counter

import glob

import numpy as np

import pretty_midi as pm

from src.application.encoder.helpers.remi_helper import get_remi_events
from src.domain.models.encoder.encoder_model import (
    EncodeParameters,
    EncodeDatasetParameters,
)
from src.application.music_base.services.music_base_service import (
    extract_chords,
    estimate_tonal_plan,
)
from src.application.encoder.models.item_model import Item
from src.domain.models.music_base.music_base_model import (
    TonalPlanParameters,
    MusicBaseParameters,
)
from src.application.encoder.helpers.encoder_helper import (
    read_note_tempo,
    quantize_midi,
    extract_beats,
    extract_downbeats,
    group_items,
    extract_dominant_keys,
)

from src.domain.constants.paths_constants import (
    ENCODINGS_PATH,
    MIDI_PATH,
    CHORDS_PATH,
    KEYS_PATH,
)
from src.persistence.dataloader.repositories.dataloader_repository import (
    save_async,
    async_load,
)


def encode_midi(parameters: EncodeParameters) -> dict:
    # Load MIDI file asynchronously
    if isinstance(parameters.midi, str):
        midi = pm.PrettyMIDI(parameters.midi)
    else:
        midi = parameters.midi
    sample = {}

    note_items, tempo_items = read_note_tempo(midi)
    quantize_midi(midi, note_items, midi.resolution)
    beats = extract_beats(midi)
    downbeats = extract_downbeats(midi)

    # Harmony - Check if chords file exists asynchronously
    chords_path = os.path.join(parameters.chords_out_dir, f"{os.path.basename(parameters.midi)}_chords.pkl")
    if os.path.isfile(chords_path):
        chords = async_load(parameters.chords_out_dir, parameters.midi, "chords")
    else:
        chords = extract_chords(
            MusicBaseParameters(
                midi=parameters.midi,
                save=parameters.save,
                out_dir=parameters.chords_out_dir,
            ),
        )

    # Tonal Plan - Check if keys file exists asynchronously
    keys_path = os.path.join(parameters.keys_out_dir, f"{os.path.basename(parameters.midi)}_keys.pkl")
    if os.path.isfile(keys_path):
        keys = async_load(parameters.keys_out_dir, parameters.midi, "keys")
    else:
        keys = estimate_tonal_plan(
            TonalPlanParameters(
                midi=parameters.midi,
                alpha=parameters.alpha,
                beta=parameters.beta,
                gamma=parameters.gamma,
                c=parameters.c,
                w=parameters.w,
                save=parameters.save,
                out_dir=parameters.keys_out_dir,
                chords=chords,
            ),
        )

    # Get REMI Chords
    if "remi_chords" in chords:
        remi_chords = chords["remi_chords"]
    else:
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

        chords_out = {"chords": chords["chords"], "remi_chords": remi_chords}

        save_async(parameters.chords_out_dir, parameters.midi, chords_out, "chords")

    # Get REMI Keys
    if "remi_keys" in keys:
        remi_keys = keys["remi_keys"]
    else:
        # Generate REMI keys
        remi_keys = []
        # for b1, b2 in zip(beats[:-1], beats[1:]):
        # for i, key in enumerate(keys["keys"]):
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

        keys_out = {"keys": keys["keys"], "remi_keys": remi_keys}
        save_async(parameters.keys_out_dir, parameters.midi, keys_out, "keys")

    midi.tonal_plan = remi_keys

    items = remi_keys + remi_chords + tempo_items + note_items
    groups = group_items(midi, downbeats, items=items)
    groups = extract_dominant_keys(groups)
    sample["events"] = get_remi_events(midi, groups)[1]

    # Save sample asynchronously
    if parameters.save:
        save_async(parameters.encodings_out_dir, parameters.midi, sample, "encoding")

    return {"Message": "Midi Encoded Successfully"}


async def encode_dataset(parameters: EncodeDatasetParameters) -> dict:
    dataset_path = os.path.join(MIDI_PATH, parameters.dataset_name)
    encodings_out_dir = os.path.join(ENCODINGS_PATH, parameters.dataset_name)
    chords_out_dir = os.path.join(CHORDS_PATH, parameters.dataset_name)
    keys_out_dir = os.path.join(KEYS_PATH, parameters.dataset_name)
    
    # Create dump directory
    dump_dir = os.path.join(dataset_path, "dump")
    os.makedirs(dump_dir, exist_ok=True)

    async def process_file(file_path: str) -> tuple[bool, str]:
        try:
            # Test if MIDI file can be loaded
            pm.PrettyMIDI(file_path)
            
            await encode_midi(
                EncodeParameters(
                    midi=file_path,
                    alpha=parameters.alpha,
                    beta=parameters.beta,
                    gamma=parameters.gamma,
                    c=parameters.c,
                    w=parameters.w,
                    save=True,
                    encodings_out_dir=encodings_out_dir,
                    chords_out_dir=chords_out_dir,
                    keys_out_dir=keys_out_dir,
                ),
            )
            return True, ""
        except Exception as e:
            # Move file to dump directory
            filename = os.path.basename(file_path)
            dump_path = os.path.join(dump_dir, filename)
            os.rename(file_path, dump_path)
            return False, f"Error processing {filename}: {str(e)}"

    # Create tasks for each file
    midi_files = [f for f in os.listdir(dataset_path) if f.endswith(('.mid', '.midi'))]
    total_files = len(midi_files)
    processed = 0
    successful = 0
    errors = []

    # Process files in batches
    batch_size = 10
    while processed < total_files:
        batch = midi_files[processed:processed + batch_size]
        batch_tasks = [
            process_file(os.path.join(dataset_path, file)) 
            for file in batch
        ]
        
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
    summary = (
        f"Dataset Encoding Complete\n"
        f"Total files: {total_files}\n"
        f"Successfully processed: {successful}\n"
        f"Failed: {len(errors)}\n"
    )
    if errors:
        summary += "\nErrors:\n" + "\n".join(errors)

    return {"Message": summary}
