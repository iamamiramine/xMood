# Standard library imports
import numpy as np

# MIDI processing and utilities
import pretty_midi
from pretty_midi.containers import TimeSignature
from pretty_midi import utilities as pm_utils

# Local application models
from core.symbolic.models.event_model import Event

# Constants - Harmony
from domain.constants.harmony_constants import key_index

# Constants - Token types
from domain.constants.token_constants import (
    # Special tokens
    EOS_TOKEN,
    # Musical feature tokens
    TIME_SIGNATURE_KEY,
    KEY_SIGNATURE_KEY,
    BAR_KEY,
    POSITION_KEY,
    INSTRUMENT_KEY,
    PITCH_KEY,
    VELOCITY_KEY,
    DURATION_KEY,
    TEMPO_KEY,
    CHORD_KEY,
)

# Constants - MIDI parameters
from domain.constants.midi_constants import (
    # Discretization parameters
    DEFAULT_POS_PER_QUARTER,
    DEFAULT_VELOCITY_BINS,
    DEFAULT_DURATION_BINS,
    DEFAULT_TEMPO_BINS,
)


def get_time_signature(midi: pretty_midi.PrettyMIDI, start: int) -> TimeSignature:
    """
    Get the time signature that is active at a specific time point in the MIDI file.

    This function determines which time signature is in effect at the given start time
    by examining the time signature changes in the MIDI file. It follows standard MIDI
    conventions where time signatures remain in effect until the next change.

    Args:
        midi (pretty_midi.PrettyMIDI): The MIDI file object to analyze
        start (int): The time point (in ticks) to check for active time signature

    Returns:
        TimeSignature: A TimeSignature object containing:
            - numerator: Top number in time signature (beats per measure)
            - denominator: Bottom number in time signature (beat unit)
            - time: Time of the signature change in seconds

    Note:
        This method assumes that time signature changes don't happen within a bar,
        which is a convention that commonly holds in musical notation. If no time
        signature is found, it defaults to 4/4 time (the standard time signature).

    Example:
        >>> midi = pretty_midi.PrettyMIDI('example.mid')
        >>> start_tick = 480  # Start of second bar in standard MIDI
        >>> time_sig = get_time_signature(midi, start_tick)
        >>> print(f"{time_sig.numerator}/{time_sig.denominator}")
        4/4
    """
    # Initialize time signature as None
    time_sig = None

    # Iterate through pairs of consecutive time signature changes
    # Exclude the last change by using [:-1] to always have a next signature to compare with
    for curr_sig, next_sig in zip(midi.time_signature_changes[:-1], midi.time_signature_changes[1:]):
        # Convert time points to ticks for comparison with start parameter
        curr_time = midi.time_to_tick(curr_sig.time)  # Start time of current signature
        next_time = midi.time_to_tick(next_sig.time)  # Start time of next signature

        # Check if start falls within the current time signature's range
        if curr_time <= start and next_time > start:
            time_sig = curr_sig  # Found the active time signature
            break

    # Handle cases where no matching time signature was found
    if time_sig is None:
        if len(midi.time_signature_changes) > 0:
            # If there are time signatures but none matched,
            # use the last time signature (still in effect)
            time_sig = midi.time_signature_changes[-1]
        else:
            # If no time signatures defined, create default 4/4 signature
            time_sig = TimeSignature(numerator=4, denominator=4, time=0.0)  # 4 beats per measure  # Quarter note gets one beat  # Start at beginning of piece

    return time_sig


def get_key_signature(midi: pretty_midi.PrettyMIDI, start: int) -> str:
    """
    Get the key signature that is active at a specific time point in the MIDI file.

    This function determines which key signature is in effect at the given start time
    by examining the tonal plan (key changes) in the MIDI file. The tonal plan is a
    sequence of key signature changes that define the harmonic context of the piece.

    Args:
        midi (pretty_midi.PrettyMIDI): The MIDI file object to analyze, must have a
            'tonal_plan' attribute containing key signature changes
        start (int): The time point (in ticks) to check for active key signature

    Returns:
        str: A string representing the key signature in format "root:mode"
            (e.g., "C:major", "A:minor"). Returns "C:major" as default if no
            key signature is found.

    Note:
        This method assumes that key signature changes don't happen within a bar,
        which is a convention that commonly holds in musical notation. The tonal
        plan should be pre-computed and stored in the MIDI object's tonal_plan
        attribute.

    Example:
        >>> midi = pretty_midi.PrettyMIDI('example.mid')
        >>> midi.tonal_plan = [key_changes]  # Pre-computed key changes
        >>> start_tick = 480  # Start of second bar in standard MIDI
        >>> key_sig = get_key_signature(midi, start_tick)
        >>> print(key_sig)
        C:major
    """
    # Initialize key signature as None
    key_sig = None

    # Iterate through pairs of consecutive key signature changes
    # Exclude the last change by using [:-1] to always have a next signature to compare with
    for curr_sig, next_sig in zip(midi.tonal_plan[:-1], midi.tonal_plan[1:]):
        # Convert time points to ticks for comparison with start parameter
        # Use end time of current signature and start time of next signature
        # to define the range where current signature is active
        curr_end = midi.time_to_tick(curr_sig.end)  # End time of current key
        next_start = midi.time_to_tick(next_sig.start)  # Start time of next key

        # Check if start falls within the current key signature's range
        if curr_end <= start and next_start > start:
            key_sig = curr_sig.pitch  # Found the active key signature
            break

    # Handle cases where no matching key signature was found
    if key_sig is None:
        if len(midi.tonal_plan) > 0:
            # If there are key signatures but none matched,
            # use the last key signature (still in effect)
            key_sig = midi.tonal_plan[-1].pitch
        else:
            # If no key signatures defined, default to C major
            key_sig = "C:major"  # Standard default key signature

    return key_sig


def get_ticks_per_bar(midi: pretty_midi.PrettyMIDI, start: int) -> int:
    """
    Calculate the number of MIDI ticks in a bar at a specific time point.

    This function determines the number of ticks in a bar based on:
    1. The time signature at the given start time
    2. The MIDI file's resolution (ticks per quarter note)
    3. The standard convention of 4 quarter notes per whole note

    The calculation follows the formula:
    ticks_per_bar = midi_resolution * (4 * numerator / denominator)
    where:
    - midi_resolution is ticks per quarter note
    - numerator is beats per bar
    - denominator defines the beat unit (e.g., 4 = quarter note)

    Args:
        midi (pretty_midi.PrettyMIDI): The MIDI file object containing resolution info
        start (int): The time point (in ticks) to check for active time signature

    Returns:
        int: Number of MIDI ticks in one bar at the given time point

    Example:
        >>> midi = pretty_midi.PrettyMIDI('example.mid')
        >>> start_tick = 480  # Start of second bar
        >>> ticks = get_ticks_per_bar(midi, start_tick)
        >>> print(ticks)  # For 4/4 time with standard resolution (480)
        1920  # (480 * 4 beats per bar)
    """
    # Get the time signature active at the given start time
    time_sig = get_time_signature(midi, start)

    # Calculate quarters per bar:
    # - Multiply numerator by 4 to convert to quarter notes
    # - Divide by denominator to account for beat unit
    quarters_per_bar = 4 * time_sig.numerator / time_sig.denominator

    # Convert quarters to ticks using MIDI resolution
    # midi.resolution is ticks per quarter note
    return midi.resolution * quarters_per_bar


def get_positions_per_bar(midi: pretty_midi.PrettyMIDI, start: int | None = None, time_sig: TimeSignature | None = None) -> int:
    """
    Calculate the number of discrete positions within a bar based on time signature.

    This function determines how many discrete time positions exist within a bar,
    taking into account:
    1. The time signature (either provided or determined from the start time)
    2. The default number of positions per quarter note (DEFAULT_POS_PER_QUARTER)
    3. The standard convention of 4 quarter notes per whole note

    The calculation follows the formula:
    positions = DEFAULT_POS_PER_QUARTER * (4 * numerator / denominator)
    where:
    - DEFAULT_POS_PER_QUARTER is the resolution of position quantization
    - numerator is beats per bar
    - denominator defines the beat unit (e.g., 4 = quarter note)

    Args:
        midi (pretty_midi.PrettyMIDI): The MIDI file object to analyze
        start (int | None, optional): The time point (in ticks) to check for active
            time signature. Only used if time_sig is None. Defaults to None.
        time_sig (TimeSignature | None, optional): Pre-determined time signature to use.
            If None, will be determined using start time. Defaults to None.

    Returns:
        int: Number of discrete positions in one bar

    Example:
        >>> midi = pretty_midi.PrettyMIDI('example.mid')
        >>> # For 4/4 time with DEFAULT_POS_PER_QUARTER = 12
        >>> positions = get_positions_per_bar(midi, start=480)
        >>> print(positions)
        48  # (12 positions/quarter * 4 quarters/bar)
    """
    # If no time signature provided, get it from the MIDI file at start time
    if time_sig is None:
        time_sig = get_time_signature(midi, start)

    # Calculate quarters per bar:
    # - Multiply numerator by 4 to convert to quarter notes
    # - Divide by denominator to account for beat unit
    quarters_per_bar = 4 * time_sig.numerator / time_sig.denominator

    # Calculate total positions per bar:
    # - Multiply quarters by positions per quarter
    # - Convert to integer as we need discrete positions
    positions_per_bar = int(DEFAULT_POS_PER_QUARTER * quarters_per_bar)

    return positions_per_bar


def tick_to_position(midi: pretty_midi.PrettyMIDI, tick: int) -> int:
    """
    Convert MIDI ticks to quantized position values based on the MIDI resolution.

    This function translates raw MIDI tick values into quantized position values
    that align with the DEFAULT_POS_PER_QUARTER grid. This quantization is essential
    for creating a discrete representation of time that can be used in the REMI
    (REvamped MIDI) format.

    The calculation follows the formula:
    position = round(tick / midi_resolution * DEFAULT_POS_PER_QUARTER)
    where:
    - tick is the input MIDI tick value
    - midi_resolution is ticks per quarter note (e.g., 480)
    - DEFAULT_POS_PER_QUARTER is the number of positions per quarter note (e.g., 12)

    Args:
        midi (pretty_midi.PrettyMIDI): The MIDI file object containing resolution info
        tick (int): The MIDI tick value to convert

    Returns:
        int: Quantized position value aligned to the DEFAULT_POS_PER_QUARTER grid

    Example:
        >>> midi = pretty_midi.PrettyMIDI('example.mid')  # resolution = 480
        >>> tick = 240  # Half a quarter note in standard MIDI
        >>> position = tick_to_position(midi, tick)
        >>> print(position)  # With DEFAULT_POS_PER_QUARTER = 12
        6  # (12 positions/quarter * 0.5 quarters)
    """
    # Convert ticks to quarter notes (tick / resolution)
    # Then convert quarter notes to positions (result * DEFAULT_POS_PER_QUARTER)
    # Round to nearest integer to ensure discrete positions
    return round(tick / midi.resolution * DEFAULT_POS_PER_QUARTER)


def get_remi_events(midi: pretty_midi.PrettyMIDI, groups: list) -> tuple[list[Event], list[str]]:
    """
    Convert MIDI data into a sequence of REMI (REvamped MIDI) events.

    This function transforms MIDI data into a sequence of symbolic events following
    the REMI format. It processes musical content bar by bar, creating events for:
    - Bar markers and positions
    - Time signatures and key signatures
    - Notes with their properties (instrument, pitch, velocity, duration)
    - Chords and their progressions
    - Tempo changes

    The events are ordered to maintain musical structure and temporal relationships,
    with bar-level events followed by their contained musical elements.

    Args:
        midi (pretty_midi.PrettyMIDI): The MIDI file object to process
        groups (list): List of bar-level event groups, where each group contains
            the musical events that occur within that bar

    Returns:
        tuple[list[Event], list[str]]:
            - list[Event]: List of Event objects representing the REMI encoding
            - list[str]: Human-readable string representations of the events

    Raises:
        ValueError: If a bar contains invalid position information (positions_per_bar <= 0)

    Example:
        >>> midi = pretty_midi.PrettyMIDI('example.mid')
        >>> groups = group_items(midi, downbeats)
        >>> events, readable = get_remi_events(midi, groups)
        >>> print(readable[:3])  # First three events
        ['Bar_1', 'Time_4/4', 'Key_C:major']
    """
    # Initialize event list and tracking variables
    events = []  # Will store all REMI events
    n_downbeat = 0  # Counter for bar numbers
    current_chord = None  # Track current chord for chord changes
    current_tempo = None  # Track current tempo for tempo changes

    # Process each bar group
    for i in range(len(groups)):
        # Extract bar boundaries (start and end times)
        bar_st, bar_et = groups[i][0], groups[i][-1]
        n_downbeat += 1  # Increment bar counter

        # Calculate number of discrete positions in this bar
        positions_per_bar = get_positions_per_bar(midi, bar_st)
        if positions_per_bar <= 0:
            raise ValueError("Invalid REMI file: There must be at least 1 position per bar.")

        # Add bar marker event
        events.append(Event(name=BAR_KEY, time=None, value=str(n_downbeat), text=str(n_downbeat)))  # Bar markers don't need specific time

        # Add time signature event
        time_sig = get_time_signature(midi, bar_st)
        events.append(
            Event(name=TIME_SIGNATURE_KEY, time=None, value=f"{time_sig.numerator}/{time_sig.denominator}", text=f"{time_sig.numerator}/{time_sig.denominator}")
        )

        # Add key signature event
        key_sig = get_key_signature(midi, bar_st)
        events.append(Event(name=KEY_SIGNATURE_KEY, time=None, value=str(key_sig), text=str(key_sig)))

        # If there's a current chord, add it at the start of the bar
        if current_chord is not None:
            # Add position event (start of bar)
            events.append(Event(name=POSITION_KEY, time=0, value="0", text=f"1/{positions_per_bar}"))  # Human-readable position
            # Add chord event
            events.append(Event(name=CHORD_KEY, time=current_chord.start, value=current_chord.pitch, text=str(current_chord.pitch)))

        # If there's a current tempo, add it at the start of the bar
        if current_tempo is not None:
            # Add position event (start of bar)
            events.append(Event(name=POSITION_KEY, time=0, value="0", text=f"1/{positions_per_bar}"))
            # Add tempo event
            tempo = current_tempo.pitch
            index = np.argmin(abs(DEFAULT_TEMPO_BINS - tempo))  # Find closest quantized tempo
            events.append(
                Event(name=TEMPO_KEY, time=current_tempo.start, value=index, text=f"{tempo}/{DEFAULT_TEMPO_BINS[index]}")  # Original and quantized tempo
            )

        # Calculate position grid for this bar
        quarters_per_bar = 4 * time_sig.numerator / time_sig.denominator
        ticks_per_bar = midi.resolution * quarters_per_bar
        # Create evenly spaced position markers
        flags = np.linspace(bar_st, bar_st + ticks_per_bar, positions_per_bar, endpoint=False)

        # Process each event in the bar (excluding bar boundaries)
        for item in groups[i][1:-1]:
            # Calculate quantized position for this event
            index = np.argmin(abs(flags - item.start))  # Find closest position
            pos_event = Event(name=POSITION_KEY, time=item.start, value=str(index), text=f"{index + 1}/{positions_per_bar}")

            if item.name == "Note":
                # Process note event
                events.append(pos_event)  # Add position first

                # Add instrument event
                if item.instrument == "drum":
                    name = "drum"
                else:
                    name = pm_utils.program_to_instrument_name(item.instrument)
                events.append(Event(name=INSTRUMENT_KEY, time=item.start, value=name, text=str(name)))

                # Add pitch event
                events.append(
                    Event(
                        name=PITCH_KEY,
                        time=item.start,
                        value=("drum_" + str(item.pitch)) if name == "drum" else item.pitch,
                        text=pm_utils.note_number_to_name(item.pitch),
                    )
                )

                # Add velocity event (quantized)
                velocity_index = np.argmin(abs(DEFAULT_VELOCITY_BINS - item.velocity))
                events.append(Event(name=VELOCITY_KEY, time=item.start, value=velocity_index, text=f"{item.velocity}/{DEFAULT_VELOCITY_BINS[velocity_index]}"))

                # Add duration event (quantized)
                duration = tick_to_position(midi, item.end - item.start)
                index = np.argmin(abs(DEFAULT_DURATION_BINS - duration))
                events.append(Event(name=DURATION_KEY, time=item.start, value=index, text=f"{duration}/{DEFAULT_DURATION_BINS[index]}"))

            elif item.name == "Chord":
                # Process chord event (only if it's different from current)
                if current_chord is None or item.pitch != current_chord.pitch:
                    events.append(pos_event)
                    events.append(Event(name=CHORD_KEY, time=item.start, value=item.pitch, text=str(item.pitch)))
                    current_chord = item

            elif item.name == "Tempo":
                # Process tempo event (only if it's different from current)
                if current_tempo is None or item.pitch != current_tempo.pitch:
                    events.append(pos_event)
                    tempo = item.pitch
                    index = np.argmin(abs(DEFAULT_TEMPO_BINS - tempo))
                    events.append(Event(name=TEMPO_KEY, time=item.start, value=index, text=f"{tempo}/{DEFAULT_TEMPO_BINS[index]}"))
                    current_tempo = item

    # Create human-readable event strings
    readable_events = [f"{e.name}_{e.value}" for e in events]

    return events, readable_events


def remi2midi(events: list[str], bpm: int = 120, time_signature: tuple[int, int] = (4, 4), polyphony_limit: int = 16) -> pretty_midi.PrettyMIDI:
    """
    Convert a sequence of REMI events back into a MIDI file.

    This function transforms a sequence of REMI (REvamped MIDI) symbolic events
    back into a playable MIDI file. It handles:
    - Time signature and tempo changes
    - Note events with their properties (pitch, velocity, duration)
    - Instrument assignments and program changes
    - Polyphony control to prevent excessive simultaneous notes
    - Bar and position-based timing calculations

    Args:
        events (list[str]): List of REMI event strings to convert
        bpm (int, optional): Initial tempo in beats per minute. Defaults to 120.
        time_signature (tuple[int, int], optional): Initial time signature as (numerator, denominator).
            Defaults to (4, 4).
        polyphony_limit (int, optional): Maximum number of simultaneous notes per instrument
            per position. Defaults to 16.

    Returns:
        pretty_midi.PrettyMIDI: A PrettyMIDI object containing the converted music

    Example:
        >>> events = ['Bar_1', 'Time_4/4', 'Position_0', 'Note_C4', ...]
        >>> midi = remi2midi(events)
        >>> midi.write('output.mid')

    Note:
        The function uses an implicit timeline system that tracks the last tempo/time
        signature change event and calculates time differences relative to that reference
        point. This helps maintain accurate timing across tempo and meter changes.
    """

    def _get_time(reference: dict, bar: int, pos: int) -> float:
        """
        Calculate absolute time in seconds for a given bar and position.

        This helpers function converts bar and position information into actual
        time in seconds, taking into account:
        - Time signature (affects beats per bar)
        - Tempo (affects seconds per beat)
        - Position within the bar
        - Distance from the last reference point

        Args:
            reference (dict): Reference point containing:
                - time (float): Time in seconds
                - pos (tuple): (bar, position) of reference
                - time_sig (TimeSignature): Current time signature
                - tempo (float): Current tempo in BPM
            bar (int): Target bar number
            pos (int): Target position within the bar

        Returns:
            float: Absolute time in seconds for the target bar and position
        """
        time_sig = reference["time_sig"]
        num, denom = time_sig.numerator, time_sig.denominator

        # Calculate quarters per bar based on time signature
        # For example, 4/4 = 4 quarters, 6/8 = 3 quarters
        qpb = 4 * num / denom

        # Get reference position
        ref_pos = reference["pos"]

        # Calculate distance in bars from reference
        d_bars = bar - ref_pos[0]

        # Calculate total position difference:
        # 1. Position difference within the bar
        # 2. Add complete bars' worth of positions
        d_pos = (pos - ref_pos[1]) + d_bars * qpb * DEFAULT_POS_PER_QUARTER

        # Convert position difference to quarter notes
        d_quarters = d_pos / DEFAULT_POS_PER_QUARTER

        # Convert quarters to seconds based on tempo
        dt = d_quarters / reference["tempo"] * 60

        # Return absolute time by adding to reference time
        return reference["time"] + dt

    # Check for tempo changes in the event sequence
    tempo_changes = [event for event in events if f"{TEMPO_KEY}_" in event]
    if len(tempo_changes) > 0:
        # If tempo changes exist, use the first one as initial tempo
        bpm = DEFAULT_TEMPO_BINS[int(tempo_changes[0].split("_")[-1])]

    # Create MIDI file with initial tempo
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)

    # Set up initial time signature
    num, denom = time_signature
    pm.time_signature_changes.append(pretty_midi.TimeSignature(num, denom, 0))
    current_time_sig = pm.time_signature_changes[0]

    # Dictionary to store instrument objects
    instruments = {}

    # Initialize timeline reference point
    # This tracks the last tempo/time signature change for timing calculations
    last_tl_event = {
        "time": 0,  # Current time in seconds
        "pos": (0, 0),  # Current (bar, position)
        "time_sig": current_time_sig,  # Current time signature
        "tempo": bpm,  # Current tempo
    }

    # Initialize bar counter and note tracking
    bar = -1  # Start before first bar
    n_notes = 0  # Count total notes added
    polyphony_control = {}  # Track simultaneous notes per position

    # Process each event in sequence
    for i, event in enumerate(events):
        # Stop at end-of-sequence token
        if event == EOS_TOKEN:
            break

        # Initialize polyphony tracking for new bars
        if not bar in polyphony_control:
            polyphony_control[bar] = {}

        # Handle bar markers
        if f"{BAR_KEY}_" in events[i]:
            # Next bar is starting
            bar += 1
            polyphony_control[bar] = {}

            # Check for time signature change in next event
            if i + 1 < len(events) and f"{TIME_SIGNATURE_KEY}_" in events[i + 1]:
                # Parse new time signature
                num, denom = events[i + 1].split("_")[-1].split("/")
                num, denom = int(num), int(denom)

                # Only create new time signature if it's different
                current_time_sig = last_tl_event["time_sig"]
                if num != current_time_sig.numerator or denom != current_time_sig.denominator:
                    # Calculate time for the change
                    time = _get_time(last_tl_event, bar, 0)
                    # Create and add new time signature
                    time_sig = pretty_midi.TimeSignature(num, denom, time)
                    pm.time_signature_changes.append(time_sig)
                    # Update reference point
                    last_tl_event["time"] = time
                    last_tl_event["pos"] = (bar, 0)
                    last_tl_event["time_sig"] = time_sig

            # Check for key signature change
            if i + 2 < len(events) and f"{KEY_SIGNATURE_KEY}_" in events[i + 2]:
                # Parse and add key signature
                key = events[i + 2].split("_")[-1]
                key = key_index().index(key)
                time = _get_time(last_tl_event, bar, 0)
                key_sig = pretty_midi.KeySignature(key, time)
                pm.key_signature_changes.append(key_sig)

        # Handle tempo changes
        elif i + 1 < len(events) and f"{POSITION_KEY}_" in events[i] and f"{TEMPO_KEY}_" in events[i + 1]:
            # Get position and new tempo
            position = int(events[i].split("_")[-1])
            tempo_idx = int(events[i + 1].split("_")[-1])
            tempo = DEFAULT_TEMPO_BINS[tempo_idx]

            # Update tempo if it changed
            if tempo != last_tl_event["tempo"]:
                time = _get_time(last_tl_event, bar, position)
                last_tl_event["time"] = time
                last_tl_event["pos"] = (bar, position)
                last_tl_event["tempo"] = tempo

        # Handle note events
        elif (
            i + 4 < len(events)
            and f"{POSITION_KEY}_" in events[i]
            and f"{INSTRUMENT_KEY}_" in events[i + 1]
            and f"{PITCH_KEY}_" in events[i + 2]
            and f"{VELOCITY_KEY}_" in events[i + 3]
            and f"{DURATION_KEY}_" in events[i + 4]
        ):
            # Get position
            position = int(events[i].split("_")[-1])

            # Initialize polyphony tracking for new positions
            if not position in polyphony_control[bar]:
                polyphony_control[bar][position] = {}

            # Get instrument
            instrument_name = events[i + 1].split("_")[-1]

            # Initialize polyphony tracking for new instruments
            if instrument_name not in polyphony_control[bar][position]:
                polyphony_control[bar][position][instrument_name] = 0
            # Skip if polyphony limit reached
            elif polyphony_control[bar][position][instrument_name] >= polyphony_limit:
                continue

            # Create or get instrument
            if instrument_name not in instruments:
                if instrument_name == "drum":
                    instrument = pretty_midi.Instrument(0, is_drum=True)
                else:
                    program = pretty_midi.instrument_name_to_program(instrument_name)
                    instrument = pretty_midi.Instrument(program)
                instruments[instrument_name] = instrument
            else:
                instrument = instruments[instrument_name]

            # Parse note properties
            pitch = int(events[i + 2].split("_")[-1])
            velocity_index = int(events[i + 3].split("_")[-1])
            velocity = min(127, DEFAULT_VELOCITY_BINS[velocity_index])
            duration_index = int(events[i + 4].split("_")[-1])
            duration = DEFAULT_DURATION_BINS[duration_index]

            # Calculate note timing
            start = _get_time(last_tl_event, bar, position)
            end = _get_time(last_tl_event, bar, position + duration)

            # Create and add note
            note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end)
            instrument.notes.append(note)
            n_notes += 1
            polyphony_control[bar][position][instrument_name] += 1

    # Add all instruments to the MIDI file
    for instrument in instruments.values():
        pm.instruments.append(instrument)

    return pm
