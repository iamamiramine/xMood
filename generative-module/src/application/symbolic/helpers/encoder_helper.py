# Standard library imports
from collections import Counter

# Scientific computing
import numpy as np

# Local application models
from core.symbolic.models.item_model import Item

# Constants - MIDI parameters
from domain.constants.midi_constants import (
    # Discretization parameters
    DEFAULT_POS_PER_QUARTER,
    DEFAULT_RESOLUTION,
)


def extract_beats(midi):
    """
    Extract beat positions from a MIDI file.

    This function retrieves the temporal positions of all beats in the MIDI file,
    using the file's time signature and tempo information to determine beat locations.
    Beats are the fundamental rhythmic units in music (e.g., quarter notes in 4/4 time).

    The function uses PrettyMIDI's beat tracking algorithm which:
    1. Analyzes the time signature to determine beat structure
    2. Uses tempo information to calculate beat timing
    3. Returns beat positions in seconds from the start of the piece

    Args:
        midi (pretty_midi.PrettyMIDI): A PrettyMIDI object containing the MIDI data
            to analyze. Must have time signature and tempo information.

    Returns:
        np.ndarray: Array of beat times in seconds, sorted in ascending order.
            Shape: [n_beats], where n_beats is the total number of beats in the piece.

    Example:
        >>> midi = pretty_midi.PrettyMIDI('example.mid')
        >>> beats = extract_beats(midi)
        >>> print(beats[:5])  # First five beats
        [0.0, 0.5, 1.0, 1.5, 2.0]  # Times in seconds

    Note:
        - Beat positions are in absolute time (seconds from start)
        - The first beat is typically at time 0.0
        - Beat intervals may vary if there are tempo changes
    """
    # Get beat positions in seconds using PrettyMIDI's beat tracking
    return midi.get_beats()


def extract_downbeats(midi):
    """
    Extract downbeat positions (first beat of each bar) from a MIDI file.

    This function identifies the temporal positions of all downbeats (bar lines) in
    the MIDI file. Downbeats are crucial for:
    1. Determining bar boundaries
    2. Understanding musical structure
    3. Analyzing metric organization

    The function uses PrettyMIDI's downbeat tracking which:
    1. Analyzes time signatures to identify bar lengths
    2. Uses tempo information to calculate precise timings
    3. Returns downbeat positions in seconds from the start

    Args:
        midi (pretty_midi.PrettyMIDI): A PrettyMIDI object containing the MIDI data
            to analyze. Must have time signature and tempo information.

    Returns:
        np.ndarray: Array of downbeat times in seconds, sorted in ascending order.
            Shape: [n_bars], where n_bars is the total number of bars in the piece.

    Example:
        >>> midi = pretty_midi.PrettyMIDI('example.mid')
        >>> downbeats = extract_downbeats(midi)
        >>> print(downbeats[:3])  # First three bar positions
        [0.0, 2.0, 4.0]  # Times in seconds (assuming 4/4 time at 120 BPM)

    Note:
        - Downbeat positions are in absolute time (seconds from start)
        - The first downbeat is typically at time 0.0
        - Intervals between downbeats may vary with:
            - Different time signatures
            - Tempo changes
    """
    # Get downbeat positions in seconds using PrettyMIDI's downbeat tracking
    return midi.get_downbeats()


def group(midi, beats, items):
    """
    Groups musical events (items) into bars based on beat positions.

    This function organizes musical events (notes, tempo changes, etc.) into bars/measures
    by examining their temporal positions relative to beat boundaries. It's a crucial step
    in REMI (REvised MIDI) representation processing.

    Args:
        midi (pretty_midi.PrettyMIDI): A PrettyMIDI object containing the MIDI data.
            Used for time-to-tick conversion of beat positions.
        beats (np.ndarray): Array of beat positions (typically downbeats) in seconds.
            These define the bar boundaries for grouping events.
        items (list[Item]): List of musical events (notes, tempo changes, etc.) to be grouped.
            Each item must have 'start' attribute in ticks.

    Returns:
        list[list]: A list of bar groups. Each bar group is structured as:
            [bar_start_tick, *musical_events, bar_end_tick]
            where:
            - bar_start_tick: Start time of the bar in MIDI ticks
            - musical_events: All events (Item objects) that fall within the bar
            - bar_end_tick: End time of the bar in MIDI ticks

    Example:
        >>> midi = pretty_midi.PrettyMIDI('example.mid')
        >>> beats = extract_downbeats(midi)
        >>> items = [note_items + tempo_items]  # Combined musical events
        >>> bar_groups = group(midi, beats, items)
        >>> print(len(bar_groups))  # Number of bars
        4
    """
    groups = []
    # Iterate through consecutive pairs of beats to define bar boundaries
    for db1, db2 in zip(beats[:-1], beats[1:]):  # move bar by bar (bar start time (db1) and bar end time (db2))
        # Convert beat times from seconds to MIDI ticks for precise positioning
        db1, db2 = midi.time_to_tick(db1), midi.time_to_tick(db2)  # convert to ticks

        # Collect all events that start within the current bar
        insiders = []  # insiders are events inside the current bar
        for item in items:
            # Check if item starts within the current bar boundaries
            if (item.start >= db1) and (item.start < db2):  # if item is within the current bar
                insiders.append(item)

        # Construct the complete bar representation:
        # [bar_start_tick, *events_in_bar, bar_end_tick]
        overall = [db1] + insiders + [db2]  # overall is a complete remi event (bar start time, {Key, Chord, Tempo, Note}, bar end time)

        groups.append(overall)

    return groups


def read_note_tempo(midi):
    """
    Extracts and processes note and tempo events from a MIDI file, handling pedal effects and tempo changes.

    This function performs two main tasks:
    1. Note Processing:
        - Extracts notes from each instrument track
        - Processes sustain pedal events (MIDI CC #64)
        - Handles note timing considering pedal effects
        - Identifies instrument types (drum vs. program number)
    2. Tempo Processing:
        - Extracts tempo change events
        - Expands tempo changes to all beat positions
        - Ensures continuous tempo coverage throughout the piece

    Args:
        midi (pretty_midi.PrettyMIDI): A PrettyMIDI object containing the MIDI data
            to analyze. Must have note and tempo information.

    Returns:
        tuple[list[Item], list[Item]]: A tuple containing two lists:
            - note_items: List of Note Items with properties:
                - name: "Note"
                - start: Note start time in ticks
                - end: Note end time in ticks (considering pedal)
                - velocity: Note velocity (0-127)
                - pitch: MIDI note number (0-127)
                - instrument: Instrument name or program number
            - tempo_items: List of Tempo Items with properties:
                - name: "Tempo"
                - start: Tempo change time in ticks
                - end: None
                - velocity: None
                - pitch: Tempo value in BPM

    Example:
        >>> midi = pretty_midi.PrettyMIDI('example.mid')
        >>> note_items, tempo_items = read_note_tempo(midi)
        >>> print(f"Found {len(note_items)} notes and {len(tempo_items)} tempo changes")
        Found 500 notes and 24 tempo changes
    """
    # Process Notes
    note_items = []
    for instrument in midi.instruments:  # Process each instrument track
        # Extract sustain pedal events (MIDI CC #64)
        pedal_events = [event for event in instrument.control_changes if event.number == 64]
        pedal_pressed = False
        start = None
        pedals = []

        # Process pedal events to create pedal segments
        for e in pedal_events:
            if e.value >= 64 and not pedal_pressed:  # Pedal press detected (value >= 64 means pressed)
                pedal_pressed = True
                start = e.time
            elif e.value < 64 and pedal_pressed:  # Pedal release detected
                pedal_pressed = False
                pedals.append(Item(name="Pedal", start=start, end=e.time))
                start = e.time

        # Get and sort notes by start time and pitch
        notes = instrument.notes
        notes.sort(key=lambda x: (x.start, x.pitch))

        # Determine instrument type (drum or program number)
        if instrument.is_drum:
            instrument_name = "drum"
        else:
            instrument_name = instrument.program

        # Process each note, considering pedal effects
        pedal_idx = 0
        for note in notes:
            # Find overlapping pedal events for the current note
            pedal_candidates = [(i + pedal_idx, pedal) for i, pedal in enumerate(pedals[pedal_idx:]) if note.end >= pedal.start and note.start < pedal.end]

            # Select appropriate pedal or create dummy pedal
            if len(pedal_candidates) > 0:
                pedal_idx = pedal_candidates[0][0]  # Update pedal index for efficiency
                pedal = pedal_candidates[-1][1]  # Use the last overlapping pedal
            else:
                pedal = Item(name="Pedal", start=0, end=0)  # Dummy pedal if none found

            # Create note item with pedal-adjusted end time
            note_items.append(
                Item(
                    name="Note",
                    start=midi.time_to_tick(note.start),
                    end=midi.time_to_tick(max(note.end, pedal.end)),  # Extend note end if pedal is held
                    velocity=note.velocity,
                    pitch=note.pitch,
                    instrument=instrument_name,
                )
            )

    # Final sort of all notes across all instruments
    note_items.sort(key=lambda x: (x.start, x.pitch))

    # Process Tempo Changes
    tempo_items = []
    times, tempi = midi.get_tempo_changes()

    # Create initial tempo items from MIDI tempo change events
    for time, tempo in zip(times, tempi):
        tempo_items.append(Item(name="Tempo", start=time, end=None, velocity=None, pitch=int(tempo)))
    tempo_items.sort(key=lambda x: x.start)

    # Expand tempo changes to ensure coverage of all beats
    max_tick = midi.time_to_tick(midi.get_end_time())
    existing_ticks = {item.start: item.pitch for item in tempo_items}
    wanted_ticks = np.arange(0, max_tick + 1, DEFAULT_RESOLUTION)  # Generate ticks at regular intervals
    output = []

    # Fill in tempo values for all beats
    for tick in wanted_ticks:
        if tick in existing_ticks:
            # Use existing tempo change
            output.append(
                Item(
                    name="Tempo",
                    start=tick,
                    end=None,
                    velocity=None,
                    pitch=existing_ticks[tick],
                )
            )
        else:
            # Propagate previous tempo value
            output.append(
                Item(
                    name="Tempo",
                    start=tick,
                    end=None,
                    velocity=None,
                    pitch=output[-1].pitch,
                )
            )
    tempo_items = output

    return note_items, tempo_items


def quantize_midi(midi, note_items, resolution):
    """
    Quantizes MIDI note timings to a regular grid based on the specified resolution.

    Quantization is the process of adjusting musical events to align with a fixed
    temporal grid. This helps create a more precise rhythmic structure and can
    correct minor timing imperfections in the performance. The function:
    1. Creates a regular grid based on the resolution
    2. Shifts each note to the nearest grid point
    3. Maintains relative timing between note start and end points

    Args:
        midi (pretty_midi.PrettyMIDI): A PrettyMIDI object containing the MIDI data.
            Used for time-to-tick conversion and determining piece duration.
        note_items (list[Item]): List of Note Items to quantize. Each item must have:
            - start: Start time in ticks
            - end: End time in ticks
        resolution (int): The desired grid resolution in ticks.
            Higher values create a finer grid with more possible note positions.
            Typically matches DEFAULT_RESOLUTION from midi_constants.

    Note:
        - The function modifies note_items in place
        - Both start and end times are shifted by the same amount to preserve note duration
        - Grid spacing is determined by resolution/DEFAULT_POS_PER_QUARTER
        - Notes are shifted to the nearest grid point to minimize timing changes

    Example:
        >>> midi = pretty_midi.PrettyMIDI('example.mid')
        >>> note_items = [Item(start=505, end=750), Item(start=1200, end=1400)]
        >>> quantize_midi(midi, note_items, DEFAULT_RESOLUTION)
        >>> print(note_items[0].start)  # Will be aligned to nearest grid point
        512  # Assuming grid points at multiples of 256
    """
    # Calculate grid spacing based on resolution
    ticks = resolution / DEFAULT_POS_PER_QUARTER  # Convert resolution to grid spacing in ticks

    # Create quantization grid
    end_tick = midi.time_to_tick(midi.get_end_time())  # Get total length in ticks
    grids = np.arange(0, max(resolution, end_tick), ticks)  # Generate regular grid points

    # Quantize each note by shifting to nearest grid point
    for item in note_items:
        # Find the nearest grid point using binary search
        index = np.searchsorted(grids, item.start, side="right")
        if index > 0:
            index -= 1  # Adjust index to get the closest grid point

        # Calculate and apply the shift to align with grid
        shift = round(grids[index]) - item.start  # Calculate required shift
        item.start += shift  # Shift note start time
        item.end += shift  # Shift note end time to maintain duration


def group_items(midi, downbeats, items):
    """
    Groups and sorts musical events into bars based on their type and timing.

    This function performs two main operations:
    1. Sorts all musical events (notes, chords, tempo changes, etc.) based on a priority system
    2. Groups the sorted events into bars using downbeat positions

    The sorting priority is determined by:
    1. Time position (earlier events first)
    2. Event type (Key → Chord → Tempo → Note)
    3. Instrument type (drums last, other instruments by program number)
    4. Pitch value (lower to higher)

    Args:
        midi (pretty_midi.PrettyMIDI): A PrettyMIDI object containing the MIDI data.
            Used for time calculations and end time determination.
        downbeats (np.ndarray): Array of downbeat positions in seconds.
            These mark the beginning of each bar.
        items (list[Item]): List of musical events to be grouped. Each item must have:
            - name: Event type ("Key", "Chord", "Tempo", or "Note")
            - start: Start time
            - instrument: (for notes) Instrument name or program number
            - pitch: Pitch value

    Returns:
        list[list]: A list of bar groups, where each bar contains:
            [bar_start_tick, *sorted_events, bar_end_tick]

    Example:
        >>> midi = pretty_midi.PrettyMIDI('example.mid')
        >>> downbeats = extract_downbeats(midi)
        >>> items = [note_items + chord_items + tempo_items]
        >>> bars = group_items(midi, downbeats, items)
        >>> print(f"Grouped {len(items)} events into {len(bars)} bars")
        Grouped 1000 events into 32 bars
    """

    def _get_key(item):
        """
        Defines the sorting priority for musical events.

        Args:
            item (Item): The musical event to be sorted

        Returns:
            tuple: A sorting key with priorities:
                1. Start time (earliest first)
                2. Event type (Key → Chord → Tempo → Note)
                3. Instrument (-1 for drums, program number for others)
                4. Pitch value
        """
        type_priority = {"Key": 0, "Chord": 1, "Tempo": 2, "Note": 3}
        return (
            item.start,  # Primary sort by time position
            type_priority[item.name],  # Secondary sort by event type priority
            (-1 if item.instrument == "drum" else item.instrument),  # Tertiary sort by instrument
            item.pitch,  # Final sort by pitch value
        )

    # Sort all items according to the priority system
    items.sort(key=_get_key)

    # Prepare bar boundaries by adding the piece end time
    # This ensures the last bar has a proper end point
    downbeats = np.concatenate([downbeats, [midi.get_end_time()]])

    # Group the sorted items into bars using the downbeat positions
    bars_groups = group(midi, downbeats, items)

    return bars_groups


def extract_dominant_keys(groups):
    """
    Processes musical bars to determine and assign the dominant key for each bar.

    This function analyzes the key events within each bar and selects the most
    frequently occurring key as the dominant key for that bar. If a bar has no
    key events, it inherits the key from the previous bar to maintain harmonic
    continuity.

    The function modifies the groups in place by:
    1. Removing all key events from each bar
    2. Inserting a single dominant key event at the start of each bar
    3. Propagating keys to empty bars from previous bars

    Args:
        groups (list[list]): A list of bar groups, where each bar contains:
            [bar_start_tick, *musical_events, bar_end_tick]
            Musical events must be Item objects with properties:
            - name: Event type (e.g., "Key", "Note", etc.)
            - pitch: For key events, represents the key value
            - start: Start time of the event

    Returns:
        list[list]: The modified groups with single dominant key per bar.
            Each bar will have exactly one key event placed after the bar start time.

    Example:
        >>> # Bar with multiple key events: [bar_start, Key(C), Key(C), Key(G), bar_end]
        >>> # becomes: [bar_start, Key(C), bar_end] (C is most frequent)
        >>> groups = [[0, Item(name="Key", pitch="C"), Item(name="Key", pitch="C"),
        ...           Item(name="Key", pitch="G"), 480]]
        >>> groups = extract_dominant_keys(groups)
        >>> print(groups[0][1].pitch)  # Shows dominant key
        'C'
    """
    # Process each bar in the piece
    for i, bar in enumerate(groups):
        # Collect all key events in the current bar
        bar_keys = []
        for item in bar[1:-1]:  # Skip bar start/end times
            if item.name == "Key":
                bar_keys.append(item.pitch)  # Store key value
                bar.remove(item)  # Remove key event from bar

        # Count occurrences of each key in the bar
        key_counts = Counter(bar_keys)

        if len(key_counts) == 0:
            # If no keys in current bar, inherit key from previous bar
            # Skip for first bar (i == 0) as it has no previous bar
            if i > 0:
                bar.insert(1, groups[i - 1][1])  # Copy key from previous bar
            continue

        # Get the most common key in the bar
        most_common_key, _ = key_counts.most_common(1)[0]

        # Insert dominant key at start of bar (after bar start time)
        bar.insert(1, Item(name="Key", start=bar[0], end=bar[-1], velocity=None, pitch=most_common_key))  # Use bar start time  # Use bar end time

    return groups
