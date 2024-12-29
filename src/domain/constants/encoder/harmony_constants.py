def get_pitch_classes():
    return ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def get_major_intervals():
    return [2, 2, 1, 2, 2, 2, 1]


def get_harmonic_minor_intervals():
    return [2, 1, 2, 2, 1, 3, 1]


def get_chord_qualities():
    return ["maj", "min", "dim", "aug", "dom7", "maj7", "min7", "N"]


def get_chord_types():
    """Returns mapping of chord type to its common name"""
    return {
        "maj": "major",
        "min": "minor",
        "dim": "diminished",
        "aug": "augmented",
        "dom7": "dominant seventh",
        "maj7": "major seventh",
        "min7": "minor seventh",
        "N": "no chord",
    }


def get_chord_symbols():
    """Returns mapping of chord type to its standard symbol notation"""
    return {"maj": "", "min": "m", "dim": "°", "aug": "+", "dom7": "7", "maj7": "maj7", "min7": "m7", "N": "N"}  # Major is typically unmarked


# def get_chord_maps():
#     return {
#         "maj": [0, 4],
#         "min": [0, 3],
#         "dim": [0, 3, 6],
#         "aug": [0, 4, 8],
#         "dom7": [0, 4, 10],
#         "maj7": [0, 4, 11],
#         "min7": [0, 3, 10],
#     }


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


def generate_webers_table():
    """
    Generate Weber's table of key relationships based on the 12 pitch classes.
    The table represents relationships between major and minor keys where:
    - Uppercase letters represent major keys
    - Lowercase letters represent minor keys
    - Vertical axis represents movement by fifths (7 semitones)
    - Horizontal axis represents alternating relative minors (9 semitones down)
        and major thirds (4 semitones up)

    Returns:
        List[List[str]]: 2D array representing Weber's table (13x10)
    """
    pitch_classes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    def next_fifth(key):
        """Get next key in circle of fifths (up 7 semitones)"""
        idx = pitch_classes.index(key)
        return pitch_classes[(idx + 7) % 12]

    def get_relative_minor(major_key):
        """Get relative minor (down 3 semitones / up 9 semitones)"""
        idx = pitch_classes.index(major_key)
        return pitch_classes[(idx + 9) % 12].lower()

    def get_major_third(key):
        """Get major third relationship (up 4 semitones)"""
        idx = pitch_classes.index(key.upper())
        return pitch_classes[(idx + 4) % 12]

    # Initialize the table (13 rows x 10 columns)
    table = [[None] * 10 for _ in range(13)]

    # Start with C major in the middle row
    middle_row = 6
    current_key = "C"

    # Fill the vertical axis (circle of fifths)
    # Fill upward from middle
    for row in range(middle_row, -1, -1):
        table[row][0] = current_key
        current_key = next_fifth(current_key)

    # Fill downward from middle
    current_key = "F"  # Start with F for downward movement
    for row in range(middle_row + 1, 13):
        table[row][0] = current_key
        current_key = next_fifth(current_key)

    # Fill horizontal axis for each row
    for row in range(13):
        current_key = table[row][0]

        # Fill each row with alternating relative minor and major third relationships
        for col in range(10):
            if col == 0:
                continue  # Skip first column as it's already filled

            if col % 2 == 1:
                # Odd columns: relative minor
                table[row][col] = get_relative_minor(current_key)
            else:
                # Even columns: major third up from previous
                current_key = get_major_third(current_key)
                table[row][col] = current_key

    return table


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


# Function to create the V and I chords for major and minor keys
def get_chords(root, scale_type):
    """
    Create the chords for major and minor keys, including primary and secondary chords.

    Args:
        root: Root note of the key
        scale_type: Either "major" or "minor"

    Returns:
        dict: Dictionary containing primary and secondary chords:
            For major keys: I, IV, iv, V, vi
            For minor keys: i, iv, IV, V, VI

    Raises:
        ValueError: If scale_type is not "major" or "minor"
    """
    if root == "N":
        return {"N": "N:N"}  # Handle no-chord case

    major_scales, minor_scales = get_major_minor_scales()

    if scale_type == "major":
        scale = major_scales[root]
        return {"I": f"{root}:maj", "IV": f"{scale[3]}:maj", "iv": f"{scale[3]}:min", "V": f"{scale[4]}:dom7", "vi": f"{scale[5]}:min"}
    elif scale_type == "minor":
        scale = minor_scales[root]
        return {"i": f"{root}:min", "iv": f"{scale[3]}:min", "IV": f"{scale[3]}:maj", "V": f"{scale[4]}:dom7", "VI": f"{scale[5]}:maj"}
    else:
        raise ValueError(f"Invalid scale type: {scale_type}")


def get_all_major_minor_keys_chords():
    """
    Generate a complete mapping of all possible keys to their primary and secondary chords.

    Returns:
        dict: Nested dictionary structure:
            {
                "major": {
                    root: {chord_degree: chord_symbol, ...},
                    ...
                },
                "minor": {
                    root: {chord_degree: chord_symbol, ...},
                    ...
                }
            }
    """
    pitch_classes = get_pitch_classes()
    # Create separate dictionaries for major and minor keys
    major_keys_chords = {root: get_chords(root, "major") for root in pitch_classes}
    minor_keys_chords = {root: get_chords(root, "minor") for root in pitch_classes}
    key_chords = {
        "major": major_keys_chords,
        "minor": minor_keys_chords,
    }
    return key_chords


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


# Sharp to natural and natural to sharp mappings
sharp_to_natural = {"C#": "C", "D#": "D", "E#": "E", "F#": "F", "G#": "G", "A#": "A", "B#": "B"}

natural_to_sharp = {
    "C": "C#",
    "D": "D#",
    "E": "E#",  # E# is enharmonically F
    "F": "F#",
    "G": "G#",
    "A": "A#",
    "B": "B#",  # B# is enharmonically C, but depends on your context
}


def generate_chord_notes():
    """
    Generate a complete mapping of all possible chords and their constituent notes.

    The function creates chords for each root note using the following qualities:
    - Major triad (maj): root, major third, perfect fifth
    - Minor triad (min): root, minor third, perfect fifth
    - Diminished triad (dim): root, minor third, diminished fifth
    - Augmented triad (aug): root, major third, augmented fifth
    - Dominant seventh (dom7): root, major third, perfect fifth, minor seventh
    - Major seventh (maj7): root, major third, perfect fifth, major seventh
    - Minor seventh (min7): root, minor third, perfect fifth, minor seventh

    Returns:
        dict: Nested dictionary structure:
            {
                root_note: {
                    chord_quality: [list of notes in chord],
                    ...
                },
                ...
            }
    """
    pitch_classes = get_pitch_classes()

    # Get chord intervals from existing function to ensure consistency
    chord_intervals = get_chord_maps()

    def generate_chord(root_index, intervals):
        """
        Calculate notes for a specific chord.

        Args:
            root_index: Index of root note in pitch_classes
            intervals: List of semitone intervals from root

        Returns:
            list: Notes that make up the chord
        """
        return [pitch_classes[(root_index + interval) % len(pitch_classes)] for interval in intervals]

    # Generate chords for each root note and quality
    return {root: {quality: generate_chord(pitch_classes.index(root), intervals) for quality, intervals in chord_intervals.items()} for root in pitch_classes}
