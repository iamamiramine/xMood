# Standard library imports
import numpy as np
from typing import List, Union

# Third-party music processing
from pretty_midi import utilities as pm_utils

# Local application imports - REMI encoding helpers
from application.encoder.helpers.remi_helper import get_time_signature, get_key_signature, get_positions_per_bar
from application.encoder.models.event_model import Event

# Constants - Feature binning defaults
from domain.constants.encoder.midi_constants import (
    DEFAULT_NOTE_DENSITY_BINS,
    DEFAULT_MEAN_VELOCITY_BINS,
    DEFAULT_MEAN_PITCH_BINS,
    DEFAULT_MEAN_DURATION_BINS,
)

# Constants - Token types
from domain.constants.encoder.token_constants import (
    # Musical context tokens
    TIME_SIGNATURE_KEY,
    KEY_SIGNATURE_KEY,
    BAR_KEY,
    INSTRUMENT_KEY,
    CHORD_KEY,
    # Statistical feature tokens
    NOTE_DENSITY_KEY,
    MEAN_VELOCITY_KEY,
    MEAN_PITCH_KEY,
    MEAN_DURATION_KEY,
    # Position token
    POSITION_KEY,
)


def get_symbolic_features(
    midi,
    groups,
    omit_time_sig=False,
    omit_instruments=False,
    omit_chords=False,
    omit_meta=False,
    add_position_tokens=False,
):
    """
    Extract musical symbolic features from MIDI data by analyzing bar-level features and events.

    This function processes MIDI data bar by bar, extracting various musical features:
    1. Structural elements: Bar numbers and positions
    2. Musical context: Time signatures, key signatures, instruments, chords
    3. Statistical features: Note density, mean velocity, mean pitch, mean duration

    Each feature token can be optionally preceded by a position token to maintain temporal ordering.

    Args:
        midi (pretty_midi.PrettyMIDI): The MIDI file data to analyze
        groups (list): List of bar-level event groups, where each group contains:
            - First element: Bar start time
            - Last element: Bar end time
            - Middle elements: Note and Chord events within the bar
        omit_time_sig (bool, optional): Skip time signature extraction. Defaults to False.
        omit_instruments (bool, optional): Skip instrument extraction. Defaults to False.
        omit_chords (bool, optional): Skip chord extraction. Defaults to False.
        omit_meta (bool, optional): Skip statistical feature extraction. Defaults to False.
        add_position_tokens (bool, optional): Whether to add position tokens before features. Defaults to False.

    Returns:
        list[str]: List of tokens describing the musical content, where each token is
                  formatted as "{feature_type}_{value}". For example:
                  - Bar token: "bar_1"
                  - Position token: "position_0" (if add_position_tokens is True)
                  - Time signature: "time_4/4"
                  - Note density: "note_density_0.5"

    Raises:
        ValueError: If a bar has invalid timing (positions_per_bar <= 0)
    """
    # Initialize list to store symbolic features events and tracking variables
    events = []  # Will store all musical events as they're extracted
    n_downbeat = 0  # Tracks the current bar number
    current_chord = None  # Maintains chord context between bars
    current_position = 0  # Tracks position within each bar

    # Process each bar group in the piece
    for i in range(len(groups)):
        # Extract bar timing information
        bar_st, bar_et = groups[i][0], groups[i][-1]  # Start and end times of current bar
        n_downbeat += 1  # Increment bar counter

        # Get musical context for current bar
        time_sig = get_time_signature(midi, bar_st)  # Time signature at bar start
        key_sig = get_key_signature(midi, bar_st)  # Key signature at bar start
        positions_per_bar = get_positions_per_bar(midi, time_sig=time_sig)  # Number of positions in bar

        # Reset position counter for new bar
        current_position = 0

        # Validate bar timing
        if positions_per_bar <= 0:
            raise ValueError("Invalid REMI file: There must be at least 1 position in each bar.")

        # Add bar marker event
        events.append(Event(name=BAR_KEY, time=None, value="{}".format(n_downbeat), text="{}".format(n_downbeat)))

        # Add position and time signature events if not omitted
        if not omit_time_sig:
            if add_position_tokens:
                events.append(Event(name=POSITION_KEY, time=None, value=str(current_position), text=f"{current_position + 1}/{positions_per_bar}"))
            events.append(
                Event(
                    name=TIME_SIGNATURE_KEY,
                    time=None,
                    value="{}/{}".format(time_sig.numerator, time_sig.denominator),
                    text="{}/{}".format(time_sig.numerator, time_sig.denominator),
                )
            )
            if add_position_tokens:
                current_position = (current_position + 1) % positions_per_bar

        # Add position and key signature events
        if add_position_tokens:
            events.append(Event(name=POSITION_KEY, time=None, value=str(current_position), text=f"{current_position + 1}/{positions_per_bar}"))
        events.append(
            Event(
                name=KEY_SIGNATURE_KEY,
                time=None,
                value="{}".format(key_sig),
                text="{}".format(key_sig),
            )
        )
        if add_position_tokens:
            current_position = (current_position + 1) % positions_per_bar

        # Process statistical features if not omitted
        if not omit_meta:
            # Extract all note events from current bar (excluding bar start/end markers)
            notes = [item for item in groups[i][1:-1] if item.name == "Note"]
            n_notes = len(notes)  # Total number of notes in bar

            # Calculate note properties for statistical analysis
            velocities = np.array([item.velocity for item in notes])  # Note velocities (loudness)
            pitches = np.array([item.pitch for item in notes])  # Note pitches (height)
            durations = np.array([item.end - item.start for item in notes])  # Note durations (length)

            # Calculate and add note density event (notes per position)
            if add_position_tokens:
                events.append(Event(name=POSITION_KEY, time=None, value=str(current_position), text=f"{current_position + 1}/{positions_per_bar}"))
            note_density = n_notes / positions_per_bar
            index = np.argmin(abs(DEFAULT_NOTE_DENSITY_BINS - note_density))  # Find closest predefined bin
            events.append(Event(name=NOTE_DENSITY_KEY, time=None, value=index, text="{:.2f}/{:.2f}".format(note_density, DEFAULT_NOTE_DENSITY_BINS[index])))
            if add_position_tokens:
                current_position = (current_position + 1) % positions_per_bar

            # Calculate and add mean velocity event
            if add_position_tokens:
                events.append(Event(name=POSITION_KEY, time=None, value=str(current_position), text=f"{current_position + 1}/{positions_per_bar}"))
            mean_velocity = velocities.mean() if len(velocities) > 0 else np.nan
            index = np.argmin(abs(DEFAULT_MEAN_VELOCITY_BINS - mean_velocity))
            events.append(
                Event(
                    name=MEAN_VELOCITY_KEY,
                    time=None,
                    value=index if mean_velocity != np.nan else "NaN",
                    text="{:.2f}/{:.2f}".format(mean_velocity, DEFAULT_MEAN_VELOCITY_BINS[index]),
                )
            )
            if add_position_tokens:
                current_position = (current_position + 1) % positions_per_bar

            # Calculate and add mean pitch event
            if add_position_tokens:
                events.append(Event(name=POSITION_KEY, time=None, value=str(current_position), text=f"{current_position + 1}/{positions_per_bar}"))
            mean_pitch = pitches.mean() if len(pitches) > 0 else np.nan
            index = np.argmin(abs(DEFAULT_MEAN_PITCH_BINS - mean_pitch))
            events.append(
                Event(
                    name=MEAN_PITCH_KEY,
                    time=None,
                    value=index if mean_pitch != np.nan else "NaN",
                    text="{:.2f}/{:.2f}".format(mean_pitch, DEFAULT_MEAN_PITCH_BINS[index]),
                )
            )
            if add_position_tokens:
                current_position = (current_position + 1) % positions_per_bar

            # Calculate and add mean duration event
            if add_position_tokens:
                events.append(Event(name=POSITION_KEY, time=None, value=str(current_position), text=f"{current_position + 1}/{positions_per_bar}"))
            mean_duration = durations.mean() if len(durations) > 0 else np.nan
            index = np.argmin(abs(DEFAULT_MEAN_DURATION_BINS - mean_duration))
            events.append(
                Event(
                    name=MEAN_DURATION_KEY,
                    time=None,
                    value=index if mean_duration != np.nan else "NaN",
                    text="{:.2f}/{:.2f}".format(mean_duration, DEFAULT_MEAN_DURATION_BINS[index]),
                )
            )
            if add_position_tokens:
                current_position = (current_position + 1) % positions_per_bar

        # Process instruments if not omitted
        if not omit_instruments:
            # Extract unique instruments from notes in current bar
            instruments = set([item.instrument for item in notes])
            if instruments:  # Only add position and instruments if we have any
                # Add single position token for all instruments
                if add_position_tokens:
                    events.append(Event(name=POSITION_KEY, time=None, value=str(current_position), text=f"{current_position + 1}/{positions_per_bar}"))
                # Add all instrument events
                for instrument in instruments:
                    instrument = pm_utils.program_to_instrument_name(instrument) if instrument != "drum" else "drum"
                    events.append(Event(name=INSTRUMENT_KEY, time=None, value=instrument, text=instrument))
                if add_position_tokens:
                    current_position = (current_position + 1) % positions_per_bar

        # Process chords if not omitted
        if not omit_chords:
            # Extract chord events from current bar
            chords = [item for item in groups[i][1:-1] if item.name == "Chord"]

            # Handle chord context between bars
            if len(chords) == 0 and current_chord is not None:
                # If no chords in current bar but we have a previous chord, use it
                chords = [current_chord]
            elif len(chords) > 0:
                # If we have chords and the first chord starts after bar start,
                # insert previous chord at beginning if available
                if chords[0].start > bar_st and current_chord is not None:
                    chords.insert(0, current_chord)
                # Update chord context for next bar
                current_chord = chords[-1]

            # Add events for all chords in bar if we have any
            if chords:
                # Add single position token for all chords
                if add_position_tokens:
                    events.append(Event(name=POSITION_KEY, time=None, value=str(current_position), text=f"{current_position + 1}/{positions_per_bar}"))
                # Add all chord events
                for chord in chords:
                    events.append(Event(name=CHORD_KEY, time=None, value=chord.pitch, text="{}".format(chord.pitch)))
                if add_position_tokens:
                    current_position = (current_position + 1) % positions_per_bar

    # Convert events to token strings in format "feature_value"
    return [f"{e.name}_{e.value}" for e in events]


def get_piece_level_symbolic_features(
    midi,
    groups,
    omit_time_sig=False,
    omit_instruments=False,
    omit_chords=False,
    omit_meta=False,
) -> List[str]:
    """
    Extract musical symbolic features at the piece level by analyzing global features.

    This function processes the entire MIDI piece as a single unit, extracting:
    1. Musical context: Predominant time signature, key signature, instruments
    2. Statistical features: Overall note density, mean velocity, mean pitch, mean duration

    Args:
        midi (pretty_midi.PrettyMIDI): The MIDI file data to analyze
        groups (list): List of bar-level event groups
        omit_time_sig (bool): Skip time signature extraction
        omit_instruments (bool): Skip instrument extraction
        omit_chords (bool): Skip chord extraction
        omit_meta (bool): Skip statistical feature extraction

    Returns:
        list[str]: List of tokens describing the piece-level musical content
    """
    events = []
    all_notes = []

    # Extract all notes from all bars
    for group in groups:
        notes = [item for item in group[1:-1] if item.name == "Note"]
        all_notes.extend(notes)

    # Get total number of positions across all bars
    total_positions = 0
    for i in range(len(groups)):
        bar_st = groups[i][0]
        time_sig = get_time_signature(midi, bar_st)
        total_positions += get_positions_per_bar(midi, time_sig=time_sig)

    # Get predominant time signature if not omitted
    if not omit_time_sig:
        time_sigs = {}
        for i in range(len(groups)):
            bar_st = groups[i][0]
            time_sig = get_time_signature(midi, bar_st)
            sig_str = f"{time_sig.numerator}/{time_sig.denominator}"
            time_sigs[sig_str] = time_sigs.get(sig_str, 0) + 1

        predominant_time_sig = max(time_sigs.items(), key=lambda x: x[1])[0]
        events.append(Event(name=TIME_SIGNATURE_KEY, time=None, value=predominant_time_sig, text=predominant_time_sig))

    # Get predominant key signature
    key_sigs = {}
    for i in range(len(groups)):
        bar_st = groups[i][0]
        key_sig = get_key_signature(midi, bar_st)
        key_sigs[key_sig] = key_sigs.get(key_sig, 0) + 1

    predominant_key = max(key_sigs.items(), key=lambda x: x[1])[0]
    events.append(Event(name=KEY_SIGNATURE_KEY, time=None, value=str(predominant_key), text=str(predominant_key)))

    # Process statistical features if not omitted
    if not omit_meta and all_notes:
        # Calculate piece-level statistics
        n_notes = len(all_notes)
        velocities = np.array([note.velocity for note in all_notes])
        pitches = np.array([note.pitch for note in all_notes])
        durations = np.array([note.end - note.start for note in all_notes])

        # Note density (notes per position across entire piece)
        note_density = n_notes / total_positions
        index = np.argmin(abs(DEFAULT_NOTE_DENSITY_BINS - note_density))
        events.append(Event(name=NOTE_DENSITY_KEY, time=None, value=index, text=f"{note_density:.2f}/{DEFAULT_NOTE_DENSITY_BINS[index]:.2f}"))

        # Mean velocity
        mean_velocity = velocities.mean()
        index = np.argmin(abs(DEFAULT_MEAN_VELOCITY_BINS - mean_velocity))
        events.append(Event(name=MEAN_VELOCITY_KEY, time=None, value=index, text=f"{mean_velocity:.2f}/{DEFAULT_MEAN_VELOCITY_BINS[index]:.2f}"))

        # Mean pitch
        mean_pitch = pitches.mean()
        index = np.argmin(abs(DEFAULT_MEAN_PITCH_BINS - mean_pitch))
        events.append(Event(name=MEAN_PITCH_KEY, time=None, value=index, text=f"{mean_pitch:.2f}/{DEFAULT_MEAN_PITCH_BINS[index]:.2f}"))

        # Mean duration
        mean_duration = durations.mean()
        index = np.argmin(abs(DEFAULT_MEAN_DURATION_BINS - mean_duration))
        events.append(Event(name=MEAN_DURATION_KEY, time=None, value=index, text=f"{mean_duration:.2f}/{DEFAULT_MEAN_DURATION_BINS[index]:.2f}"))

    # Process instruments if not omitted
    if not omit_instruments:
        instruments = set(note.instrument for note in all_notes)
        for instrument in instruments:
            instrument = pm_utils.program_to_instrument_name(instrument) if instrument != "drum" else "drum"
            events.append(Event(name=INSTRUMENT_KEY, time=None, value=instrument, text=instrument))

    # Process chords if not omitted
    if not omit_chords:
        all_chords = []
        for group in groups:
            chords = [item for item in group[1:-1] if item.name == "Chord"]
            all_chords.extend(chords)

        if all_chords:
            # Get unique chords while preserving order of first appearance
            unique_chords = []
            seen_chords = set()
            for chord in all_chords:
                if chord.pitch not in seen_chords:
                    unique_chords.append(chord)
                    seen_chords.add(chord.pitch)
            
            # Add all unique chords
            for chord in unique_chords:
                events.append(Event(name=CHORD_KEY, time=None, value=chord.pitch, text=str(chord.pitch)))

    return [f"{e.name}_{e.value}" for e in events]
