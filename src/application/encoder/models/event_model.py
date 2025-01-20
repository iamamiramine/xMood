"""
This module defines the Event class, which represents musical events in a symbolic format.
Events are fundamental units used to represent various musical elements such as notes,
chords, time signatures, key signatures, and other musical features.
"""


class Event(object):
    """
    A class representing a musical event with its properties and string representation.

    The Event class is used to store and manage musical events in a standardized format.
    Each event has four main properties:
    - name: The type/category of the event (e.g., "Note", "Chord", "Key")
    - time: The timing information (can be None for events without temporal position)
    - value: The actual value/data of the event (e.g., pitch, chord quality)
    - text: A human-readable representation of the event

    Examples:
        # Creating a note event
        note_event = Event(name="Note", time=480, value="60", text="C4")

        # Creating a chord event
        chord_event = Event(name="Chord", time=None, value="C:maj", text="C major")

        # Creating a key signature event
        key_event = Event(name="Key", time=None, value="G:maj", text="G major")

    Note:
        The time property can be None for events that don't require temporal positioning
        (e.g., global key signatures or time signatures).
    """

    def __init__(self, name: str, time: int | None, value: str, text: str):
        """
        Initialize a new Event instance.

        Args:
            name (str): The type/category of the event (e.g., "Note", "Chord", "Key")
            time (int | None): Timing information in ticks, or None if not applicable
            value (str): The actual value/data of the event
            text (str): Human-readable representation of the event

        Example:
            event = Event(
                name="Note",          # Event type is a musical note
                time=480,             # Position at tick 480
                value="60",           # MIDI note number 60 (middle C)
                text="C4"             # Human-readable note name
            )
        """
        self.name = name      # Type/category of the event
        self.time = time      # Temporal position in ticks (if applicable)
        self.value = value    # Actual value/data of the event
        self.text = text      # Human-readable representation

    def __repr__(self) -> str:
        """
        Create a string representation of the Event instance.

        This method provides a detailed string representation of the event,
        useful for debugging, logging, and display purposes. The representation
        includes all event properties in a clear, readable format.

        Returns:
            str: A string in the format 'Event(name={}, time={}, value={}, text={})'

        Example:
            >>> event = Event("Note", 480, "60", "C4")
            >>> print(event)
            Event(name=Note, time=480, value=60, text=C4)
        """
        return 'Event(name={}, time={}, value={}, text={})'.format(
            self.name,    # Event type/category
            self.time,    # Temporal position
            self.value,   # Event value/data
            self.text     # Human-readable text
        )