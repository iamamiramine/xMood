import numpy as np
import pretty_midi
from pretty_midi.containers import TimeSignature
from pretty_midi import utilities as pm_utils

from src.domain.constants.encoder.harmony_constants import key_index
from src.application.encoder.models.vocab_model import RemiVocab
from src.application.encoder.models.event_model import Event

from src.domain.constants.encoder.token_constants import (
    # vocab_service keys
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
    EOS_TOKEN,
)

from src.domain.constants.encoder.midi_constants import (
    # discretization parameters
    DEFAULT_POS_PER_QUARTER,
    DEFAULT_VELOCITY_BINS,
    DEFAULT_DURATION_BINS,
    DEFAULT_TEMPO_BINS,
)


def _get_time_signature(midi, start):
    # This method assumes that time signature changes don't happen within a bar
    # which is a convention that commonly holds
    time_sig = None
    for curr_sig, next_sig in zip(midi.time_signature_changes[:-1], midi.time_signature_changes[1:]):
        if midi.time_to_tick(curr_sig.time) <= start and midi.time_to_tick(next_sig.time) > start:
            time_sig = curr_sig
            break
    if time_sig is None:
        if len(midi.time_signature_changes) > 0:
            time_sig = midi.time_signature_changes[-1]
        else:
            time_sig = TimeSignature(4, 4, 0.0)  # default value
    return time_sig


def _get_key_signature(midi, start):
    # This method assumes that time signature changes don't happen within a bar
    # which is a convention that commonly holds
    key_sig = None
    for curr_sig, next_sig in zip(midi.tonal_plan[:-1], midi.tonal_plan[1:]):
        if midi.time_to_tick(curr_sig.end) <= start and midi.time_to_tick(next_sig.start) > start:
            key_sig = curr_sig.pitch
            break
    if key_sig is None:
        if len(midi.tonal_plan) > 0:
            key_sig = midi.tonal_plan[-1].pitch
        else:
            key_sig = "C:major"  # default value
    return key_sig


def _get_ticks_per_bar(midi, start):
    time_sig = _get_time_signature(start)
    quarters_per_bar = 4 * time_sig.numerator / time_sig.denominator
    return midi.resolution * quarters_per_bar


def _get_positions_per_bar(midi, start=None, time_sig=None):
    if time_sig is None:
        time_sig = _get_time_signature(midi, start)
    quarters_per_bar = 4 * time_sig.numerator / time_sig.denominator
    positions_per_bar = int(DEFAULT_POS_PER_QUARTER * quarters_per_bar)
    return positions_per_bar


def tick_to_position(midi, tick):
    return round(tick / midi.resolution * DEFAULT_POS_PER_QUARTER)


def get_remi_events(midi, groups):
    events = []
    n_downbeat = 0
    current_chord = None
    current_tempo = None
    for i in range(len(groups)):  # iterating over all the MIDI Events
        bar_st, bar_et = (
            groups[i][0],
            groups[i][-1],
        )  # get start and end time of each bars
        n_downbeat += 1  # bar count
        positions_per_bar = _get_positions_per_bar(midi, bar_st)
        if positions_per_bar <= 0:
            raise ValueError("Invalid REMI file: There must be at least 1 position per bar.")

        # Adding Bar
        events.append(
            Event(
                name=BAR_KEY,
                time=None,
                value="{}".format(n_downbeat),
                text="{}".format(n_downbeat),
            )
        )

        # Adding Time Signature
        time_sig = _get_time_signature(midi, bar_st)
        events.append(
            Event(
                name=TIME_SIGNATURE_KEY,
                time=None,
                value="{}/{}".format(time_sig.numerator, time_sig.denominator),
                text="{}/{}".format(time_sig.numerator, time_sig.denominator),
            )
        )

        key_sig = _get_key_signature(midi, bar_st)
        events.append(
            Event(
                name=KEY_SIGNATURE_KEY,
                time=None,
                value="{}".format(key_sig),
                text="{}".format(key_sig),
            )
        )

        if current_chord is not None:
            events.append(
                Event(
                    name=POSITION_KEY,
                    time=0,
                    value="{}".format(0),
                    text="{}/{}".format(1, positions_per_bar),
                )
            )
            events.append(
                Event(
                    name=CHORD_KEY,
                    time=current_chord.start,
                    value=current_chord.pitch,
                    text="{}".format(current_chord.pitch),
                )
            )

        if current_tempo is not None:
            events.append(
                Event(
                    name=POSITION_KEY,
                    time=0,
                    value="{}".format(0),
                    text="{}/{}".format(1, positions_per_bar),
                )
            )
            tempo = current_tempo.pitch
            index = np.argmin(abs(DEFAULT_TEMPO_BINS - tempo))
            events.append(
                Event(
                    name=TEMPO_KEY,
                    time=current_tempo.start,
                    value=index,
                    text="{}/{}".format(tempo, DEFAULT_TEMPO_BINS[index]),
                )
            )

        quarters_per_bar = 4 * time_sig.numerator / time_sig.denominator
        ticks_per_bar = midi.resolution * quarters_per_bar
        flags = np.linspace(bar_st, bar_st + ticks_per_bar, positions_per_bar, endpoint=False)

        for item in groups[i][1:-1]:  # iterate over the items of each bar (excluding bar start and end time)
            """Position Event"""
            index = np.argmin(abs(flags - item.start))
            pos_event = Event(
                name=POSITION_KEY,
                time=item.start,
                value="{}".format(index),
                text="{}/{}".format(index + 1, positions_per_bar),
            )

            if item.name == "Note":
                """Note Event (as per REMI: Each note is followed by Positin Pitch, Velocity and Duration)"""
                events.append(pos_event)
                # instrument
                if item.instrument == "drum":
                    name = "drum"
                else:
                    name = pm_utils.program_to_instrument_name(item.instrument)
                events.append(
                    Event(
                        name=INSTRUMENT_KEY,
                        time=item.start,
                        value=name,
                        text="{}".format(name),
                    )
                )
                # pitch
                events.append(
                    Event(
                        name=PITCH_KEY,
                        time=item.start,
                        value=("drum_{}".format(item.pitch) if name == "drum" else item.pitch),
                        text="{}".format(pm_utils.note_number_to_name(item.pitch)),
                    )
                )
                # velocity
                velocity_index = np.argmin(abs(DEFAULT_VELOCITY_BINS - item.velocity))
                events.append(
                    Event(
                        name=VELOCITY_KEY,
                        time=item.start,
                        value=velocity_index,
                        text="{}/{}".format(item.velocity, DEFAULT_VELOCITY_BINS[velocity_index]),
                    )
                )
                # duration
                duration = tick_to_position(midi, item.end - item.start)
                index = np.argmin(abs(DEFAULT_DURATION_BINS - duration))
                events.append(
                    Event(
                        name=DURATION_KEY,
                        time=item.start,
                        value=index,
                        text="{}/{}".format(duration, DEFAULT_DURATION_BINS[index]),
                    )
                )

            elif item.name == "Chord":
                """Chord Event (as per REMI+)"""
                if current_chord is None or item.pitch != current_chord.pitch:
                    events.append(pos_event)
                    events.append(
                        Event(
                            name=CHORD_KEY,
                            time=item.start,
                            value=item.pitch,
                            text="{}".format(item.pitch),
                        )
                    )
                    current_chord = item

            elif item.name == "Tempo":
                """Tempo Event (as per REMI+)"""
                if current_tempo is None or item.pitch != current_tempo.pitch:
                    events.append(pos_event)
                    tempo = item.pitch
                    index = np.argmin(abs(DEFAULT_TEMPO_BINS - tempo))
                    events.append(
                        Event(
                            name=TEMPO_KEY,
                            time=item.start,
                            value=index,
                            text="{}/{}".format(tempo, DEFAULT_TEMPO_BINS[index]),
                        )
                    )
                    current_tempo = item
    readable_events = [f"{e.name}_{e.value}" for e in events]

    return events, readable_events


def remi2midi(events, bpm=120, time_signature=(4, 4), polyphony_limit=16):
    vocab = RemiVocab()

    def _get_time(bar, position, bpm=120, positions_per_bar=48):
        abs_position = bar * positions_per_bar + position
        beat = abs_position / DEFAULT_POS_PER_QUARTER
        return beat / bpm * 60

    def _get_time(reference, bar, pos):
        time_sig = reference["time_sig"]
        num, denom = time_sig.numerator, time_sig.denominator
        # Quarters per bar, assuming 4 quarters per whole note
        qpb = 4 * num / denom
        ref_pos = reference["pos"]
        d_bars = bar - ref_pos[0]
        d_pos = (pos - ref_pos[1]) + d_bars * qpb * DEFAULT_POS_PER_QUARTER
        d_quarters = d_pos / DEFAULT_POS_PER_QUARTER
        # Convert quarters to seconds
        dt = d_quarters / reference["tempo"] * 60
        return reference["time"] + dt

    # time_sigs = [event.split('_')[-1].split('/') for event in events if f"{TIME_SIGNATURE_KEY}_" in event]
    # time_sigs = [(int(num), int(denom)) for num, denom in time_sigs]

    tempo_changes = [event for event in events if f"{TEMPO_KEY}_" in event]
    print(tempo_changes, flush=True)
    if len(tempo_changes) > 0:
        bpm = DEFAULT_TEMPO_BINS[int(tempo_changes[0].split("_")[-1])]

    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    num, denom = time_signature
    pm.time_signature_changes.append(pretty_midi.TimeSignature(num, denom, 0))
    current_time_sig = pm.time_signature_changes[0]

    instruments = {}

    # Use implicit timeline: keep track of last tempo/time signature change event
    # and calculate time difference relative to that
    last_tl_event = {"time": 0, "pos": (0, 0), "time_sig": current_time_sig, "tempo": bpm}

    bar = -1
    n_notes = 0
    polyphony_control = {}
    for i, event in enumerate(events):
        if event == EOS_TOKEN:
            break

        if not bar in polyphony_control:
            polyphony_control[bar] = {}

        if f"{BAR_KEY}_" in events[i]:
            # Next bar is starting
            bar += 1
            polyphony_control[bar] = {}

            if i + 1 < len(events) and f"{TIME_SIGNATURE_KEY}_" in events[i + 1]:
                num, denom = events[i + 1].split("_")[-1].split("/")
                num, denom = int(num), int(denom)
                current_time_sig = last_tl_event["time_sig"]
                if num != current_time_sig.numerator or denom != current_time_sig.denominator:
                    time = _get_time(last_tl_event, bar, 0)
                    time_sig = pretty_midi.TimeSignature(num, denom, time)
                    pm.time_signature_changes.append(time_sig)
                    last_tl_event["time"] = time
                    last_tl_event["pos"] = (bar, 0)
                    last_tl_event["time_sig"] = time_sig

            if i + 2 < len(events) and f"{KEY_SIGNATURE_KEY}_" in events[i + 2]:
                key = events[i + 2].split("_")[-1]
                key = key_index().index(key)
                time = _get_time(last_tl_event, bar, 0)
                key_sig = pretty_midi.KeySignature(key, time)
                pm.key_signature_changes.append(key_sig)

        elif i + 1 < len(events) and f"{POSITION_KEY}_" in events[i] and f"{TEMPO_KEY}_" in events[i + 1]:
            position = int(events[i].split("_")[-1])
            tempo_idx = int(events[i + 1].split("_")[-1])
            tempo = DEFAULT_TEMPO_BINS[tempo_idx]

            if tempo != last_tl_event["tempo"]:
                time = _get_time(last_tl_event, bar, position)
                last_tl_event["time"] = time
                last_tl_event["pos"] = (bar, position)
                last_tl_event["tempo"] = tempo

        elif (
            i + 4 < len(events)
            and f"{POSITION_KEY}_" in events[i]
            and f"{INSTRUMENT_KEY}_" in events[i + 1]
            and f"{PITCH_KEY}_" in events[i + 2]
            and f"{VELOCITY_KEY}_" in events[i + 3]
            and f"{DURATION_KEY}_" in events[i + 4]
        ):
            # get position
            position = int(events[i].split("_")[-1])
            if not position in polyphony_control[bar]:
                polyphony_control[bar][position] = {}

            # get instrument
            instrument_name = events[i + 1].split("_")[-1]
            if instrument_name not in polyphony_control[bar][position]:
                polyphony_control[bar][position][instrument_name] = 0
            elif polyphony_control[bar][position][instrument_name] >= polyphony_limit:
                # If number of notes exceeds polyphony limit, omit this note
                continue

            if instrument_name not in instruments:
                if instrument_name == "drum":
                    instrument = pretty_midi.Instrument(0, is_drum=True)
                else:
                    program = pretty_midi.instrument_name_to_program(instrument_name)
                    instrument = pretty_midi.Instrument(program)
                instruments[instrument_name] = instrument
            else:
                instrument = instruments[instrument_name]

            # get pitch
            pitch = int(events[i + 2].split("_")[-1])
            # get velocity
            velocity_index = int(events[i + 3].split("_")[-1])
            velocity = min(127, DEFAULT_VELOCITY_BINS[velocity_index])
            # get duration
            duration_index = int(events[i + 4].split("_")[-1])
            duration = DEFAULT_DURATION_BINS[duration_index]
            # create not and add to instrument
            start = _get_time(last_tl_event, bar, position)
            end = _get_time(last_tl_event, bar, position + duration)
            note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end)
            instrument.notes.append(note)
            n_notes += 1
            polyphony_control[bar][position][instrument_name] += 1

    for instrument in instruments.values():
        pm.instruments.append(instrument)
    return pm
