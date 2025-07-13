def get_pitch_classes():
    return ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def get_major_intervals():
    return [2, 2, 1, 2, 2, 2, 1]


def get_harmonic_minor_intervals():
    return [2, 1, 2, 2, 1, 3, 1]


def get_chord_qualities():
    return ["maj", "min", "dim", "aug", "dom7", "maj7", "min7", "N"]


def get_chord_maps():
    return {
        "maj": [0, 4, 7],  # Added missing third note (7 semitones from root)
        "min": [0, 3, 7],  # Added missing third note (7 semitones from root)
        "dim": [0, 3, 6],
        "aug": [0, 4, 8],
        "dom7": [0, 4, 7, 10],  # Added missing fifth
        "maj7": [0, 4, 7, 11],  # Added missing fifth
        "min7": [0, 3, 7, 10],  # Added missing fifth
    }


def get_chord_insiders():
    """Returns mapping of chord type to its characteristic/insider notes.

    Insider notes are notes that strongly reinforce the chord quality but are not part of
    the basic chord structure (root and third). For most chords, this is primarily the perfect fifth.

    The perfect fifth (7 semitones) is considered an insider for most chord types because:
    - It provides stability and reinforces the root
    - It's present in most common chord voicings
    - It helps define the chord's sonority

    The diminished chord is an exception, using the minor sixth (9 semitones) as its insider
    since it contains a diminished fifth rather than perfect fifth.

    The augmented chord has no insider notes since its augmented fifth already serves
    as its characteristic interval.

    Returns:
        dict: Mapping of chord type to list of semitone intervals representing insider notes
              relative to the root.

    Example:
        For a major chord, [7] represents:
        - 7: Perfect fifth above root
    """
    return {
        "maj": [7],  # Perfect fifth
        "min": [7],  # Perfect fifth
        "dim": [9],  # Minor sixth
        "aug": [],  # No insider notes due to augmented fifth
        "dom7": [7],  # Perfect fifth
        "maj7": [7],  # Perfect fifth
        "min7": [7],  # Perfect fifth
    }


def get_chord_outsiders_1():
    """Returns mapping of chord type to its first-level outsider notes.

    First-level outsiders are notes that are somewhat compatible with the chord
    but not part of its core structure. These typically include:
    - Major/minor second (2)
    - Perfect fourth (5)
    - Major sixth/minor seventh (9/8)

    Returns:
        dict: Mapping of chord type to list of semitone intervals representing
              first-level outsider notes relative to the root.

    Example:
        For a major chord, [2,5,9] represents:
        - 2: Major second
        - 5: Perfect fourth
        - 9: Major sixth
    """
    return {
        "maj": [2, 5, 9],
        "min": [2, 5, 8],
        "dim": [2, 5, 10],
        "aug": [2, 5, 9],
        "dom7": [2, 5, 9],
        "maj7": [2, 5, 9],
        "min7": [2, 5, 8],
    }


def get_chord_outsiders_2():
    """Returns mapping of chord type to its second-level outsider notes.

    Second-level outsiders are notes that create more dissonance with the chord.
    These typically include:
    - Minor second/major seventh (1/11)
    - Minor third/major third (3/4) when not part of chord
    - Tritone (6)
    - Augmented fifth/minor sixth (8/8)
    - Minor seventh/major sixth (10/9) when not part of chord

    Returns:
        dict: Mapping of chord type to list of semitone intervals representing
              second-level outsider notes relative to the root.

    Example:
        For a major chord, [1,3,6,8,10,11] represents:
        - 1: Minor second
        - 3: Minor third
        - 6: Tritone
        - 8: Augmented fifth
        - 10: Minor seventh
        - 11: Major seventh
    """
    return {
        "maj": [1, 3, 6, 8, 10, 11],
        "min": [1, 4, 6, 9, 11],
        "dim": [1, 4, 7, 8, 11],
        "aug": [1, 3, 6, 7, 10],
        "dom7": [1, 3, 6, 8, 11],
        "maj7": [1, 3, 6, 8, 10],
        "min7": [1, 4, 6, 9, 11],
    }


def generate_scale(root, intervals):
    """
    Generates a scale based on a root note and intervals.

    Args:
        root: Starting note of the scale
        intervals: List of semitone intervals between consecutive scale degrees

    Returns:
        list: Notes of the generated scale

    Example:
        generate_scale("C", [2, 2, 1, 2, 2, 2, 1]) returns C major scale
    """
    pitch_classes = get_pitch_classes()
    scale = [root]
    index = pitch_classes.index(root)
    for interval in intervals:
        index = (index + interval) % len(pitch_classes)
        scale.append(pitch_classes[index])
    return scale


def get_major_minor_scales():
    """
    Generate all major and minor scales based on the 12 pitch classes.

    Returns:
        tuple: (major_scales, minor_scales)
            - major_scales: dict mapping root note to major scale notes
            - minor_scales: dict mapping root note to harmonic minor scale notes
    """
    major_intervals = get_major_intervals()
    harmonic_minor_intervals = get_harmonic_minor_intervals()
    pitch_classes = get_pitch_classes()
    # Create major and minor scales dictionaries
    major_scales = {note: generate_scale(note, major_intervals) for note in pitch_classes}
    minor_scales = {note: generate_scale(note, harmonic_minor_intervals) for note in pitch_classes}
    return major_scales, minor_scales


def get_all_major_minor_keys():
    """
    Generate a list of all possible major and minor keys.

    Returns:
        list: List of tuples (root, scale_type) for all possible keys
              Example: [('C', 'major'), ('C', 'minor'), ...]
    """
    major_scales, minor_scales = get_major_minor_scales()
    major_keys = list(major_scales.keys())
    minor_keys = list(minor_scales.keys())
    candidate_keys = [(key, "major") for key in major_keys] + [(key, "minor") for key in minor_keys]
    return candidate_keys


def key_index():
    """
    Generate a list of all possible keys in format 'pitch:scale_type'.

    Returns:
        list: List of strings in format 'pitch:scale_type'
              (e.g., ['C:major', 'C#:major', ..., 'B:major', 'C:minor', ...])
    """
    pitch_classes = get_pitch_classes()

    # Generate a list with major keys first, then minor keys
    keys = [f"{pitch}:major" for pitch in pitch_classes] + [f"{pitch}:minor" for pitch in pitch_classes]

    return keys
