# Standard library imports
import numpy as np

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
)


def get_symbolic_features(
    midi,
    groups,
    omit_time_sig=False,
    omit_instruments=False,
    omit_chords=False,
    omit_meta=False,
):
    """
    Extract musical symbolic features from MIDI data by analyzing bar-level features and events.

    This function processes MIDI data bar by bar, extracting various musical features:
    1. Structural elements: Bar numbers and positions
    2. Musical context: Time signatures, key signatures, instruments, chords
    3. Statistical features: Note density, mean velocity, mean pitch, mean duration

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

    Returns:
        list[str]: List of tokens describing the musical content, where each token is
                  formatted as "{feature_type}_{value}". For example:
                  - Bar token: "bar_1"
                  - Time signature: "time_4/4"
                  - Note density: "note_density_0.5"

    Raises:
        ValueError: If a bar has invalid timing (positions_per_bar <= 0)
    """
    # Initialize list to store symbolic features events and tracking variables
    events = []  # Will store all musical events as they're extracted
    n_downbeat = 0  # Tracks the current bar number
    current_chord = None  # Maintains chord context between bars

    # Process each bar group in the piece
    for i in range(len(groups)):
        # Extract bar timing information
        bar_st, bar_et = groups[i][0], groups[i][-1]  # Start and end times of current bar
        n_downbeat += 1  # Increment bar counter

        # Get musical context for current bar
        time_sig = get_time_signature(midi, bar_st)  # Time signature at bar start
        key_sig = get_key_signature(midi, bar_st)  # Key signature at bar start
        positions_per_bar = get_positions_per_bar(midi, time_sig=time_sig)  # Number of positions in bar

        # Validate bar timing
        if positions_per_bar <= 0:
            raise ValueError("Invalid REMI file: There must be at least 1 position in each bar.")

        # Add bar marker event
        events.append(Event(name=BAR_KEY, time=None, value="{}".format(n_downbeat), text="{}".format(n_downbeat)))

        # Add time signature event if not omitted
        if not omit_time_sig:
            events.append(
                Event(
                    name=TIME_SIGNATURE_KEY,
                    time=None,
                    value="{}/{}".format(time_sig.numerator, time_sig.denominator),
                    text="{}/{}".format(time_sig.numerator, time_sig.denominator),
                )
            )

        # Add key signature event
        events.append(
            Event(
                name=KEY_SIGNATURE_KEY,
                time=None,
                value="{}".format(key_sig),
                text="{}".format(key_sig),
            )
        )

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
            note_density = n_notes / positions_per_bar
            index = np.argmin(abs(DEFAULT_NOTE_DENSITY_BINS - note_density))  # Find closest predefined bin
            events.append(Event(name=NOTE_DENSITY_KEY, time=None, value=index, text="{:.2f}/{:.2f}".format(note_density, DEFAULT_NOTE_DENSITY_BINS[index])))

            # Calculate and add mean velocity event
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

            # Calculate and add mean pitch event
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

            # Calculate and add mean duration event
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

        # Process instruments if not omitted
        if not omit_instruments:
            # Extract unique instruments from notes in current bar
            instruments = set([item.instrument for item in notes])
            # Add event for each instrument, converting program numbers to names
            for instrument in instruments:
                instrument = pm_utils.program_to_instrument_name(instrument) if instrument != "drum" else "drum"
                events.append(Event(name=INSTRUMENT_KEY, time=None, value=instrument, text=instrument))

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

            # Add events for all chords in bar
            for chord in chords:
                events.append(Event(name=CHORD_KEY, time=None, value=chord.pitch, text="{}".format(chord.pitch)))

    # Convert events to token strings in format "feature_value"
    return [f"{e.name}_{e.value}" for e in events]
