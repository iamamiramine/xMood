import os
from typing import Any

import numpy as np

import pretty_midi as pm
import soundfile
import fluidsynth

from persistence.dataloader.repositories.dataloader_repository import (
    save_async,
)
from application.music_base.helpers.chord_extraction_helper import (
    get_candidate_chords,
    dynamic_chords,
    dedupe_chords,
)
from domain.models.music_base.music_base_model import (
    MusicBaseParameters,
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
