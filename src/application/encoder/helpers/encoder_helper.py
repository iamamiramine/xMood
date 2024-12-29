from collections import Counter

import numpy as np

from src.application.encoder.models.item_model import Item

from src.domain.constants.encoder.midi_constants import (
    # discretization parameters
    DEFAULT_POS_PER_QUARTER,
    DEFAULT_RESOLUTION,
)


def extract_beats(midi):
    return midi.get_beats()


def extract_downbeats(midi):
    return midi.get_downbeats()


def group(midi, beats, items):
    groups = []
    for db1, db2 in zip(beats[:-1], beats[1:]):  # move bar by bar (bar start time (db1) and bar end time (db2))
        db1, db2 = midi.time_to_tick(db1), midi.time_to_tick(db2)  # convert to ticks
        insiders = []  # insiders are events inside the current bar
        for item in items:
            if (item.start >= db1) and (item.start < db2):  # if item is within the current bar
                insiders.append(item)
        overall = [db1] + insiders + [db2]  # overall is a complete remi event (bar start time, {Key, Chord, Tempo, Note}, bar end time)

        groups.append(overall)

    return groups


def trim_empty_groups(groups):
    # Trim empty groups from the beginning and end
    for idx in [0, -1]:
        while len(groups) > 0:
            group = groups[idx]  # group: self.groups[idx] = [bar start time, {Key, Chord, Tempo, Note}, bar end time]
            notes = [item for item in group[1:-1] if item.name == "Note"]
            if len(notes) == 0:
                groups.pop(idx)
            else:
                break
    return groups


def read_note_tempo(midi):
    """
    Read Note and Tempo Changes from MIDI
        Note and Tempo in this method are of Item class
        Since we decompose MIDI into channels, each channel will have only one instrument, and each instrument will have only one track
    """

    """Note"""
    note_items = []
    for instrument in midi.instruments:  # When midi is decomposed into channels, each channel will have only one instrument
        pedal_events = [event for event in instrument.control_changes if event.number == 64]
        pedal_pressed = False
        start = None
        pedals = []
        for e in pedal_events:
            if e.value >= 64 and not pedal_pressed:
                pedal_pressed = True
                start = e.time
            elif e.value < 64 and pedal_pressed:
                pedal_pressed = False
                pedals.append(Item(name="Pedal", start=start, end=e.time))
                start = e.time

        notes = instrument.notes  # Instrument notes are the same as Channel notes
        notes.sort(key=lambda x: (x.start, x.pitch))  # Sort notes according to start time and pitch values

        # Instrument Name According to General MIDI Program Number
        if instrument.is_drum:
            instrument_name = "drum"
        else:
            instrument_name = instrument.program

        # Add note item
        pedal_idx = 0
        for note in notes:
            pedal_candidates = [(i + pedal_idx, pedal) for i, pedal in enumerate(pedals[pedal_idx:]) if note.end >= pedal.start and note.start < pedal.end]
            if len(pedal_candidates) > 0:
                pedal_idx = pedal_candidates[0][0]
                pedal = pedal_candidates[-1][1]
            else:
                pedal = Item(name="Pedal", start=0, end=0)

            note_items.append(
                Item(
                    name="Note",
                    start=midi.time_to_tick(note.start),
                    end=midi.time_to_tick(max(note.end, pedal.end)),
                    velocity=note.velocity,
                    pitch=note.pitch,
                    instrument=instrument_name,
                )
            )

    note_items.sort(key=lambda x: (x.start, x.pitch))  # Another sort based on start time and pitch values

    """Tempo"""
    tempo_items = []
    times, tempi = midi.get_tempo_changes()
    for time, tempo in zip(times, tempi):
        tempo_items.append(Item(name="Tempo", start=time, end=None, velocity=None, pitch=int(tempo)))
    tempo_items.sort(key=lambda x: x.start)  # Sort Tempo based on start time

    # Expand tempo changes to all beats
    max_tick = midi.time_to_tick(midi.get_end_time())  # end time (tick) of a track
    existing_ticks = {item.start: item.pitch for item in tempo_items}  # existing tempo changes
    wanted_ticks = np.arange(0, max_tick + 1, DEFAULT_RESOLUTION)  # all ticks from 0 to max_tick at intervals of DEFAULT_RESOLUTION (all beats)
    output = []

    for tick in wanted_ticks:
        if tick in existing_ticks:
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
    ticks = resolution / DEFAULT_POS_PER_QUARTER
    # grid
    end_tick = midi.time_to_tick(midi.get_end_time())
    grids = np.arange(0, max(resolution, end_tick), ticks)
    # process
    for item in note_items:
        index = np.searchsorted(grids, item.start, side="right")
        if index > 0:
            index -= 1
        shift = round(grids[index]) - item.start
        item.start += shift
        item.end += shift


def group_items(midi, downbeats, items):
    def _get_key(item):
        type_priority = {"Key": 0, "Chord": 1, "Tempo": 2, "Note": 3}
        return (
            item.start,  # order by time
            type_priority[item.name],  # chord events first, then tempo events, then note events
            (-1 if item.instrument == "drum" else item.instrument),  # order by instrument
            item.pitch,  # order by note pitch
        )

    items.sort(key=_get_key)  # beats for position extraction

    # Bar Extraction
    downbeats = np.concatenate([downbeats, [midi.get_end_time()]])  # get end time and concatenate with bars
    bars_groups = group(midi, downbeats, items)

    return bars_groups


def extract_dominant_keys(groups):
    # Choosing dominant key per bar
    for i, bar in enumerate(groups):
        bar_keys = []
        for item in bar[1:-1]:
            if item.name == "Key":
                bar_keys.append(item.pitch)
                # remove key from bar
                bar.remove(item)
        # Count occurrences of each key
        key_counts = Counter(bar_keys)
        if len(key_counts) == 0:
            bar.insert(1, groups[i - 1][1])  # if no key in the bar, add the key from the previous bar
            continue
        # Get the most common key (key, count)
        most_common_key, _ = key_counts.most_common(1)[0]
        # Re-add key to the bar at the first index after the time of the bar
        bar.insert(1, Item(name="Key", start=bar[0], end=bar[-1], velocity=None, pitch=most_common_key))
    return groups
