import os
import pickle
from typing import Any

import numpy as np

import pretty_midi as pm
import soundfile
import fluidsynth

from persistence.dataloader.repositories.dataloader_repository import (
    save_async,
)
from application.music_base.helpers.tonal_plan_helper import (
    evaluate_diatonic_pitch_set_distance,
    update_diatonic_pitch_set,
    compute_tonality_anchoring_cost,
    find_dominant_instrument,
    reconstruct_optimal_path,
    compute_proximity_costs,
    detect_modulation_points,
)
from application.music_base.helpers.chord_extraction_helper import (
    get_candidate_chords,
    dynamic_chords,
    dedupe_chords,
)
from domain.constants.encoder.harmony_constants import (
    get_all_major_minor_keys,
    get_all_major_minor_keys_chords,
    generate_webers_table,
    generate_chord_notes,
)
from domain.models.music_base.music_base_model import (
    MusicBaseParameters,
    TonalPlanParameters,
)

pm.instrument._HAS_FLUIDSYNTH = True
pm.instrument.fluidsynth = fluidsynth


def synthesize_midi(file: str, out_dir: str) -> dict:
    # # synthesize the generated MIDI and display it
    midi = pm.PrettyMIDI(file)
    audio = midi.fluidsynth()
    soundfile.write(os.path.join(out_dir, f"{os.path.basename(file)}.wav"), audio, 44100)

    return {"Message": "MIDI Synthesized Successfully!"}


def extract_chords(parameters: MusicBaseParameters) -> Any:
    """
    Chord Extraction algorithm of Pop Music Transformer: Beat-based Modeling and Generation of Expressive Pop Piano Compositions
    """
    if isinstance(parameters.midi, str):
        midi = pm.PrettyMIDI(parameters.midi)
    else:
        midi = parameters.midi
    resolution = midi.resolution
    beats = midi.get_beats()
    chroma = midi.get_chroma(times=beats)

    end_tick = midi.time_to_tick(midi.get_end_time())
    if end_tick < resolution:  # If sequence is shorter than 1/4th note, it's probably empty
        return []

    # get lots of candidates
    candidates = get_candidate_chords(chroma, max_tick=len(beats))

    # greedy
    chords = dynamic_chords(candidates=candidates, max_tick=len(beats), min_length=1)
    chords = dedupe_chords(chords)
    for chord in chords:
        chord[0] = beats[chord[0]]
        if chord[1] >= len(beats):
            chord[1] = midi.get_end_time()
        else:
            chord[1] = beats[chord[1]]

    def expand_chords(chords, beats):
        expanded_chords = []
        for beat in beats:
            # Find the chord whose time range includes this beat
            for chord in chords:
                start_time, end_time, chord_name = chord
                if start_time <= beat < end_time:
                    expanded_chords.append((beat, chord_name))
                    break
        return expanded_chords

    # Get the expanded chords list
    beats = midi.get_beats()
    expanded_chords = expand_chords(chords, beats)

    out = {"chords": chords, "expanded_chords": expanded_chords}

    if parameters.save:
        save_async(parameters.out_dir, parameters.midi, out, "chords")  # Async save

    if not parameters.api_call:
        return out
    else:
        return {"Message": "Chords extracted successfully!"}


def estimate_tonal_plan(parameters: TonalPlanParameters) -> Any:
    """
    Estimate the tonal plan of a musical piece using dynamic programming.
    Based on the paper: "Estimating keys and modulations in musical pieces"

    Args:
        parameters: TonalPlanParameters containing MIDI file, chords, and algorithm parameters

    Returns:
        dict: Contains the optimal sequence of keys or API success message
    """
    # Load and prepare input data
    midi = parameters.midi if isinstance(parameters.midi, pm.PrettyMIDI) else pm.PrettyMIDI(parameters.midi)

    # Find dominant instrument
    try:
        dominant_instrument_idx = find_dominant_instrument(midi)
        dominant_instrument = midi.instruments[dominant_instrument_idx]
    except ValueError as e:
        print(f"Warning: {e}. Falling back to first instrument.")
        dominant_instrument = midi.instruments[0]

    if isinstance(parameters.chords, str):
        with open(parameters.chords, "rb") as f:
            chords = pickle.load(f)
    else:
        chords = parameters.chords
    expanded_chords = chords["expanded_chords"]

    # Initialize constants and lookup tables
    candidate_keys = get_all_major_minor_keys()
    key_chords = get_all_major_minor_keys_chords()
    webers_table = generate_webers_table()
    chord_notes = generate_chord_notes()
    cs_b = ["C", "D", "E", "F", "G", "A", "B"]  # Initial diatonic pitch set

    # Initialize dynamic programming matrices
    beats = midi.get_beats()
    num_beats = len(beats)
    num_keys = len(candidate_keys)

    D = np.full((num_beats, num_keys), float("inf"))
    proximity_costs = np.full((num_beats, num_keys, num_keys), float("inf"))
    tonality_anchoring_costs = np.zeros((num_beats, num_keys))

    # Initialize first beat
    D[0, :] = 0
    # Initialize first beat with actual costs instead of zeros
    for key_idx, (key_pitch, key_mode) in enumerate(candidate_keys):
        # Get key-specific information
        candidate_key = f"{key_pitch}:{key_mode}"
        key_specific_chords = key_chords[key_mode][key_pitch]

        # Compute initial costs
        anchoring_cost = compute_tonality_anchoring_cost(
            expanded_chords[0][1],
            expanded_chords[1][1] if len(expanded_chords) > 1 else None,
            key_specific_chords,
            chord_notes,
            0,  # No previous cost for first beat
        )

        compatibility_cost = evaluate_diatonic_pitch_set_distance(key_pitch, key_mode, cs_b)

        D[0, key_idx] = (parameters.alpha * anchoring_cost / parameters.c) + (parameters.beta * compatibility_cost / 7)

    # Forward pass: Compute optimal costs
    for beat in range(1, num_beats):
        # Update current diatonic pitch set based on notes at this beat
        cs_b = update_diatonic_pitch_set(dominant_instrument, beats[beat], cs_b)

        # Get chord information
        current_chord = expanded_chords[beat][1]
        next_chord = expanded_chords[beat + 1][1] if beat < num_beats - 1 else None

        # Compute costs for each candidate key
        for key_idx, (key_pitch, key_mode) in enumerate(candidate_keys):
            # Get key-specific information
            candidate_key = f"{key_pitch}:{key_mode}"
            key_specific_chords = key_chords[key_mode][key_pitch]

            # Compute individual cost components
            anchoring_cost = compute_tonality_anchoring_cost(
                current_chord, next_chord, key_specific_chords, chord_notes, tonality_anchoring_costs[beat - 1, key_idx]
            )
            tonality_anchoring_costs[beat, key_idx] = anchoring_cost

            compatibility_cost = evaluate_diatonic_pitch_set_distance(key_pitch, key_mode, cs_b)

            # Compute proximity costs for all possible previous keys
            compute_proximity_costs(beat, key_idx, candidate_key, candidate_keys, webers_table, D, proximity_costs, parameters)

            # Find minimum proximity cost from previous keys
            min_proximity_idx = np.argmin(proximity_costs[beat, :, key_idx])
            min_proximity_cost = proximity_costs[beat, min_proximity_idx, key_idx]

            # Compute total cost
            D[beat, key_idx] = (parameters.alpha * anchoring_cost / parameters.c) + (parameters.beta * compatibility_cost / 7) + min_proximity_cost

    # Backward pass: Reconstruct optimal path
    optimal_plan = reconstruct_optimal_path(D, proximity_costs, num_beats, num_keys)

    # Detect modulation points
    modulation_points = detect_modulation_points(D, optimal_plan, max_cost=20, modulation_threshold=0.3, min_key_duration=4)

    # Convert indices to key names
    optimal_plan_keys = [(candidate_keys[idx][0], candidate_keys[idx][1]) for idx in optimal_plan]

    # Add modulation information to output
    out = {
        "keys": optimal_plan_keys,
        "modulations": [
            {
                "beat": beat,
                "confidence": conf,
                "from_key": f"{optimal_plan_keys[beat-1][0]}:{optimal_plan_keys[beat-1][1]}",
                "to_key": f"{optimal_plan_keys[beat][0]}:{optimal_plan_keys[beat][1]}",
            }
            for beat, conf in modulation_points
        ],
    }

    # Handle output
    if parameters.save:
        save_async(parameters.out_dir, parameters.midi, out, "keys")

    return out if not parameters.api_call else {"Message": "Tonal Plan Estimated Successfully!"}
