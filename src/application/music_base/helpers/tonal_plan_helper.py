import math

import numpy as np

import pretty_midi as pm

from src.domain.constants.encoder.harmony_constants import (
    sharp_to_natural,
    natural_to_sharp,
    get_major_minor_scales,
)

# Define pitch classes for diatonic scale
pitch_classes_diatonic = ["C", "D", "E", "F", "G", "A", "B"]


def find_dominant_instrument(midi: pm.PrettyMIDI) -> int:
    """
    Find the most dominant instrument in the MIDI file based on musical features.

    Args:
        midi: PrettyMIDI object containing the piece

    Returns:
        int: Index of the most dominant instrument
    """
    if not midi.instruments:
        raise ValueError("No instruments found in MIDI file")

    # Skip drum tracks
    melody_instruments = [i for i in midi.instruments if not i.is_drum]
    if not melody_instruments:
        raise ValueError("No melody instruments found in MIDI file")

    # Initialize scoring dictionary
    instrument_scores = {}

    for idx, instrument in enumerate(melody_instruments):
        # Skip empty instruments
        if not instrument.notes:
            continue

        # Calculate various musical features

        # 1. Note density (notes per second)
        duration = midi.get_end_time()
        note_density = len(instrument.notes) / duration if duration > 0 else 0

        # 2. Average velocity (loudness)
        avg_velocity = sum(note.velocity for note in instrument.notes) / len(instrument.notes)

        # 3. Pitch range (melodic spread)
        pitches = [note.pitch for note in instrument.notes]
        pitch_range = max(pitches) - min(pitches)

        # 4. Activity ratio (percentage of time with active notes)
        active_time = sum(note.end - note.start for note in instrument.notes)
        activity_ratio = active_time / duration if duration > 0 else 0

        # 5. Polyphony (average simultaneous notes)
        note_times = [(note.start, 1) for note in instrument.notes]
        note_times.extend((note.end, -1) for note in instrument.notes)
        note_times.sort()

        current_polyphony = 0
        max_polyphony = 0
        for _, change in note_times:
            current_polyphony += change
            max_polyphony = max(max_polyphony, current_polyphony)

        # Calculate weighted score
        # Weights can be adjusted based on importance of each feature
        score = (
            0.25 * note_density
            + 0.20 * (avg_velocity / 127)  # More notes suggest importance
            + 0.15 * (pitch_range / 88)  # Louder parts often more important
            + 0.25 * activity_ratio  # Wider range suggests melodic role
            + 0.15 * (max_polyphony / 12)  # More active parts usually important  # Harmonic importance
        )

        instrument_scores[idx] = {
            "score": score,
            "program": instrument.program,
            "is_drum": instrument.is_drum,
            "note_count": len(instrument.notes),
            "density": note_density,
            "velocity": avg_velocity,
            "range": pitch_range,
            "activity": activity_ratio,
            "polyphony": max_polyphony,
        }

    # Find instrument with highest score
    if not instrument_scores:
        raise ValueError("No valid instruments found for analysis")

    dominant_idx = max(instrument_scores.items(), key=lambda x: x[1]["score"])[0]

    return dominant_idx


def update_diatonic_pitch_set(dominant_instrument, beat_time, cs_b):
    """
    Update the diatonic pitch set based on notes at the current beat.
    Returns a new pitch set with accidentals adjusted based on recent notes.
    """
    # Create a copy to avoid modifying the original
    updated_cs_b = cs_b.copy()

    # Get all notes active at this beat
    active_notes = [note for note in dominant_instrument.notes if note.start < beat_time and note.end > beat_time]

    # Update pitch set based on active notes
    for note in active_notes:
        pitch_name = pm.note_number_to_name(note.pitch)[0:-1]
        if pitch_name in sharp_to_natural and pitch_name not in updated_cs_b:
            idx = updated_cs_b.index(sharp_to_natural[pitch_name])
            updated_cs_b[idx] = pitch_name
        elif pitch_name in natural_to_sharp and pitch_name not in updated_cs_b:
            idx = updated_cs_b.index(natural_to_sharp[pitch_name])
            updated_cs_b[idx] = pitch_name

    return updated_cs_b


def get_usual_diatonic_pitch_set(key_signature, mode):
    """Get the standard diatonic pitch set for a given key and mode."""
    major_scales, minor_scales = get_major_minor_scales()

    if mode == "major":
        scale = major_scales.get(key_signature)
    elif mode == "minor":
        scale = minor_scales.get(key_signature)
    else:
        raise ValueError("Mode must be either 'major' or 'minor'")

    # Extract just the note names without octave numbers
    pitch_set = [note[0] for note in scale[:-1]]  # Exclude last note (octave)

    # Sort according to standard order: C D E F G A B
    base_index = {note: idx for idx, note in enumerate(["C", "D", "E", "F", "G", "A", "B"])}
    return sorted(pitch_set, key=lambda note: base_index.get(note[0], float("inf")))


def ddiat(cs_b, S_b):
    """
    Calculate the diatonic pitch set distance between two sets.

    Args:
        cs_b: Current diatonic pitch set
        S_b: Standard pitch set for the key being evaluated

    Returns:
        float: Normalized distance between 0 and 1
    """
    distance = 0
    accidental_weight = 1.5  # Higher weight for accidental differences

    for curr_note, std_note in zip(cs_b, S_b):
        if curr_note != std_note:
            # Check if difference is due to accidental
            if curr_note[0] == std_note[0]:  # Same letter, different accidental
                distance += accidental_weight
            else:  # Different note entirely
                distance += 1

    return distance / (len(cs_b) * accidental_weight)  # Normalize to [0,1]


def evaluate_diatonic_pitch_set_distance(candidate_key, candidate_mode, cs_b):
    """
    Evaluate the distance between current pitch set and key's standard pitch set.

    Args:
        candidate_key: Key being evaluated
        candidate_mode: Mode of the key (major/minor)
        cs_b: Current diatonic pitch set from the music

    Returns:
        float: Normalized distance score between 0 and 1
    """
    # Get standard pitch set for candidate key
    S_b = get_usual_diatonic_pitch_set(candidate_key, candidate_mode)

    # Ensure both sets are in the same format
    cs_b = [note[0] if len(note) == 1 else note for note in cs_b]  # Handle natural notes

    # Calculate normalized distance
    distance = ddiat(cs_b, S_b)

    return distance


def tonality_anchoring(current_chord, next_chord, chord_notes):
    # Iterate over chords to calculate distances
    curr_chord = current_chord.split("/")[0]
    next_chord = next_chord.split("/")[0]

    curr_chord_pitch = curr_chord.split(":")[0]
    curr_chord_quality = curr_chord.split(":")[1]
    next_chord_pitch = next_chord.split(":")[0]
    next_chord_quality = next_chord.split(":")[1]

    curr_chord_notes = chord_notes[curr_chord_pitch][curr_chord_quality]
    next_chord_notes = chord_notes[next_chord_pitch][next_chord_quality]

    if len(curr_chord_notes) > 3:
        seventh_to_third = curr_chord_notes[3] == next_chord_notes[1]
    else:
        seventh_to_third = False

    third_to_tonic = curr_chord_notes[1] == next_chord_notes[0]
    root_to_root = curr_chord_notes[0] == next_chord_notes[0]

    if (root_to_root and third_to_tonic) or (third_to_tonic and seventh_to_third) or (root_to_root and seventh_to_third):
        return True
    return False


# def _compute_tonality_anchoring_cost(current_chord, next_chord, key_chords, chord_notes, previous_cost):
#     """Compute tonality anchoring cost based on V-I progressions"""
#     if current_chord == key_chords["V"] and next_chord is not None and next_chord == key_chords["I"]:
#         return 0 if tonality_anchoring(current_chord, next_chord, chord_notes) else min(20, previous_cost + 1)
#     return min(20, previous_cost + 1)


def compute_tonality_anchoring_cost(current_chord, next_chord, key_chords, chord_notes, previous_cost):
    """
    Compute tonality anchoring cost based on common cadential patterns.
    Returns 0 for strong cadences, scaled costs for weaker ones.
    """
    # Extract chord information
    if current_chord and next_chord:
        curr_root = current_chord.split(":")[0]
        curr_quality = current_chord.split(":")[1]
        next_root = next_chord.split(":")[0]
        next_quality = next_chord.split(":")[1]

        # Determine if we're in a major or minor key
        is_major = "I" in key_chords  # True for major keys, False for minor keys
        tonic_chord = key_chords["I"] if is_major else key_chords["i"]

        # Perfect Authentic Cadence (V-I or V-i)
        if current_chord == key_chords["V"] and next_chord == tonic_chord and tonality_anchoring(current_chord, next_chord, chord_notes):
            return 0

        # Imperfect Authentic Cadence (V-I or V-i with different inversions)
        elif curr_root == key_chords["V"].split(":")[0] and next_chord == tonic_chord:
            return 0.3

        # Plagal Cadence (IV-I or iv-i)
        elif (curr_root == key_chords["IV"].split(":")[0] or curr_root == key_chords["iv"].split(":")[0]) and next_chord == tonic_chord:
            return 0.5

        # Deceptive Cadence (V-vi in major or V-VI in minor)
        elif (
            current_chord == key_chords["V"]
            and next_root == key_chords["vi" if is_major else "VI"].split(":")[0]
            and next_quality == ("min" if is_major else "maj")
        ):
            return 0.7

        # Half Cadence (x-V)
        elif next_chord == key_chords["V"]:
            return 0.8

    # No recognized cadential pattern
    return min(20, previous_cost + 1)


def proximity(key_coordinates_curr, key_coordinates_cand, w=10):
    """
    Calculate the proximity between two keys based on their coordinates in Weber's table.
    Uses Euclidean distance with weights for vertical (fifth) and horizontal (third) movements.

    Args:
        key_coordinates_curr: List of (row, col) coordinates for current key
        key_coordinates_cand: List of (row, col) coordinates for candidate key
        w: Maximum distance threshold (default: 10)

    Returns:
        float: Weighted distance between keys, bounded by w
    """
    # Weight vertical movement (fifths) and horizontal movement (thirds) differently
    VERTICAL_WEIGHT = 1.0  # Movement by fifths
    HORIZONTAL_WEIGHT = 1.5  # Movement by thirds (slightly penalized)

    distances = []
    for row_curr, col_curr in key_coordinates_curr:
        for row_cand, col_cand in key_coordinates_cand:
            # Calculate weighted Euclidean distance
            vertical_dist = VERTICAL_WEIGHT * (row_cand - row_curr)
            horizontal_dist = HORIZONTAL_WEIGHT * (col_cand - col_curr)
            distance = math.sqrt(vertical_dist**2 + horizontal_dist**2)
            distances.append(distance)

    # Return minimum distance found, bounded by w
    return min(min(distances), w)


def calculate_proximity_at_beats(webers_table, current_key, candidate_key):
    """
    Calculate key proximity between two keys using Weber's table.

    Args:
        webers_table: 2D array representing Weber's table
        current_key: String representing current key (e.g., "C:major" or "A:minor")
        candidate_key: String representing candidate key

    Returns:
        float: Proximity cost between the two keys
    """

    def find_key_coordinates(key_str):
        pitch, mode = key_str.split(":")
        # Convert key to Weber's table format (uppercase for major, lowercase for minor)
        weber_key = pitch.upper() if mode == "major" else pitch.lower()

        coordinates = []
        # Search for all occurrences of the key in Weber's table
        for i, row in enumerate(webers_table):
            for j, cell in enumerate(row):
                if cell == weber_key:
                    coordinates.append((i, j))
        return coordinates

    # Get coordinates for both keys
    current_coords = find_key_coordinates(current_key)
    candidate_coords = find_key_coordinates(candidate_key)

    if not current_coords or not candidate_coords:
        raise ValueError(f"Could not find coordinates for keys: {current_key} or {candidate_key}")

    # Calculate proximity using the coordinates
    return proximity(current_coords, candidate_coords)


def compute_proximity_costs(beat, key_idx, candidate_key, candidate_keys, webers_table, D, proximity_costs, parameters):
    """Compute proximity costs between current key and all possible previous keys"""
    for prev_key_idx, (prev_pitch, prev_mode) in enumerate(candidate_keys):
        prev_key = f"{prev_pitch}:{prev_mode}"
        proximity_cost = calculate_proximity_at_beats(webers_table, candidate_key, prev_key)
        # Separate proximity cost from previous beat's cost
        proximity_costs[beat, prev_key_idx, key_idx] = parameters.gamma * (proximity_cost / parameters.w)
        total_cost = proximity_costs[beat, prev_key_idx, key_idx] + D[beat - 1, prev_key_idx]
        D[beat, key_idx] = min(D[beat, key_idx], total_cost)


def reconstruct_optimal_path(D, proximity_costs, num_beats, num_keys):
    optimal_plan = np.zeros(num_beats, dtype=int)
    optimal_plan[-1] = np.argmin(D[-1, :])

    for beat in range(num_beats - 2, -1, -1):
        prev_costs = D[beat, :] + proximity_costs[beat + 1, :, optimal_plan[beat + 1]]
        optimal_plan[beat] = np.argmin(prev_costs)

    return optimal_plan


def detect_modulation_points(D: np.ndarray, optimal_plan: np.ndarray, max_cost, modulation_threshold, min_key_duration) -> list:
    """
    Detect points of key modulation based on cost matrix analysis.
    
    Args:
        D: Cost matrix from dynamic programming (num_beats x num_keys)
        optimal_plan: Array of optimal key indices for each beat
        parameters: Algorithm parameters including modulation threshold
        
    Returns:
        list: List of tuples (beat_index, confidence) where modulations occur
    """
    modulation_points = []
    num_beats = len(optimal_plan)
    
    # Window size for cost smoothing
    window_size = 3
    
    # Keep track of the last modulation point
    last_modulation_beat = 0
    
    for beat in range(1, num_beats):
        # Skip if key hasn't changed
        if optimal_plan[beat] == optimal_plan[beat-1]:
            continue
            
        # Skip if too close to previous modulation
        if beat - last_modulation_beat < min_key_duration:
            continue
            
        # Compute cost difference around the potential modulation point
        prev_costs = D[max(0, beat-window_size):beat, optimal_plan[beat-1]]
        next_costs = D[beat:min(num_beats, beat+window_size), optimal_plan[beat]]
        
        # Average costs before and after the modulation
        prev_avg_cost = np.mean(prev_costs)
        next_avg_cost = np.mean(next_costs)
        
        # Compute modulation confidence based on cost difference
        cost_diff = abs(next_avg_cost - prev_avg_cost)
        confidence = cost_diff / max_cost  # Normalize by maximum cost
        
        # Only keep significant modulations
        if confidence > modulation_threshold:
            modulation_points.append((beat, confidence))
            last_modulation_beat = beat  # Update last modulation point
    
    return modulation_points