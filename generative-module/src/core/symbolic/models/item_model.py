"""
This module defines the Item class, which represents musical items with temporal properties.
Items are used to store and manage musical elements that have a duration (start and end times),
such as notes, chords, and other time-bounded musical events.
"""


class Item(object):
    """
    A class representing a musical item with temporal and musical properties.

    The Item class is designed to store musical elements that have a duration and
    optional musical properties. It is particularly useful for representing:
    - Notes: With start time, end time, velocity, pitch, and instrument
    - Chords: With start time, end time, and pitch (chord symbol)
    - Key Changes: With start time, end time, and pitch (key signature)
    - Other temporal musical events

    Each item has these properties:
    - name: The type/category of the item (e.g., "Note", "Chord", "Key")
    - start: Start time in ticks
    - end: End time in ticks
    - velocity: Optional note velocity (0-127 for MIDI)
    - pitch: Optional pitch information (MIDI number or symbol)
    - instrument: Optional instrument identifier

    Examples:
        # Creating a note item
        note = Item(
            name="Note",
            start=480,    # Start at tick 480
            end=960,      # End at tick 960
            velocity=64,  # Medium velocity
            pitch=60,     # Middle C
            instrument=0  # Piano
        )

        # Creating a chord item
        chord = Item(
            name="Chord",
            start=0,
            end=1920,
            pitch="C:maj"  # C major chord
        )

    Note:
        All time values are in MIDI ticks, which can be converted to
        real time based on the MIDI file's resolution and tempo.
    """

    def __init__(self, name: str, start: int, end: int, velocity: int | None = None,
                 pitch: int | str | None = None, instrument: int | None = None):
        """
        Initialize a new Item instance.

        Args:
            name (str): The type/category of the item (e.g., "Note", "Chord", "Key")
            start (int): Start time in MIDI ticks
            end (int): End time in MIDI ticks
            velocity (int | None, optional): MIDI velocity (0-127). Defaults to None.
            pitch (int | str | None, optional): MIDI note number or symbol. Defaults to None.
            instrument (int | None, optional): MIDI program number. Defaults to None.

        Example:
            item = Item(
                name="Note",          # Item type is a musical note
                start=480,            # Start at tick 480
                end=960,              # End at tick 960 (duration = 480 ticks)
                velocity=64,          # Medium velocity
                pitch=60,             # Middle C
                instrument=0          # Piano
            )
        """
        self.name = name              # Type/category of the item
        self.start = start            # Start time in ticks
        self.end = end                # End time in ticks
        self.velocity = velocity      # Note velocity (if applicable)
        self.pitch = pitch           # Pitch information (note number or symbol)
        self.instrument = instrument  # Instrument identifier

    def __repr__(self) -> str:
        """
        Create a string representation of the Item instance.

        This method provides a detailed string representation of the item,
        useful for debugging, logging, and display purposes. The representation
        includes all item properties in a clear, readable format.

        Returns:
            str: A string in the format 'Item(name={}, start={}, end={}, velocity={}, pitch={}, instrument={})'

        Example:
            >>> item = Item("Note", 480, 960, 64, 60, 0)
            >>> print(item)
            Item(name=Note, start=480, end=960, velocity=64, pitch=60, instrument=0)
        """
        return 'Item(name={}, start={}, end={}, velocity={}, pitch={}, instrument={})'.format(
            self.name,        # Item type/category
            self.start,       # Start time
            self.end,         # End time
            self.velocity,    # Velocity value
            self.pitch,       # Pitch value
            self.instrument   # Instrument value
        )