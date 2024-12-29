import numpy as np
import torch
import re
import logging

from pretty_midi import utilities as pm_utils

from src.application.encoder.helpers.remi_helper import _get_time_signature, _get_key_signature, _get_positions_per_bar
from src.application.encoder.models.event_model import Event
from src.application.encoder.models.vocab_model import DescriptionVocab, RemiVocab
from src.domain.constants.encoder.midi_constants import (
    DEFAULT_NOTE_DENSITY_BINS,
    DEFAULT_POS_PER_QUARTER,
    DEFAULT_MEAN_VELOCITY_BINS,
    DEFAULT_MEAN_PITCH_BINS,
    DEFAULT_MEAN_DURATION_BINS,
)
from src.domain.constants.encoder.token_constants import (
    NOTE_DENSITY_KEY,
    TIME_SIGNATURE_KEY,
    BAR_KEY,
    BOS_TOKEN,
    EOS_TOKEN,
    KEY_SIGNATURE_KEY,
    MEAN_VELOCITY_KEY,
    MEAN_PITCH_KEY,
    MEAN_DURATION_KEY,
    INSTRUMENT_KEY,
    CHORD_KEY,
)

desc_vocab = DescriptionVocab()
remi_vocab = RemiVocab()


# TODO: Write documentation for this method
# TODO: Add more symbolic features
def get_description(
    midi,
    groups,
    omit_time_sig=False,
    omit_instruments=False,
    omit_chords=False,
    omit_meta=False,
):
    events = []
    n_downbeat = 0
    current_chord = None

    for i in range(len(groups)):
        bar_st, bar_et = groups[i][0], groups[i][-1]
        n_downbeat += 1
        time_sig = _get_time_signature(midi, bar_st)
        key_sig = _get_key_signature(midi, bar_st)
        positions_per_bar = _get_positions_per_bar(midi, time_sig=time_sig)
        if positions_per_bar <= 0:
            raise ValueError("Invalid REMI file: There must be at least 1 position in each bar.")

        events.append(Event(name=BAR_KEY, time=None, value="{}".format(n_downbeat), text="{}".format(n_downbeat)))

        if not omit_time_sig:
            events.append(
                Event(
                    name=TIME_SIGNATURE_KEY,
                    time=None,
                    value="{}/{}".format(time_sig.numerator, time_sig.denominator),
                    text="{}/{}".format(time_sig.numerator, time_sig.denominator),
                )
            )

        events.append(
            Event(
                name=KEY_SIGNATURE_KEY,
                time=None,
                value="{}".format(key_sig),
                text="{}".format(key_sig),
            )
        )

        if not omit_meta:
            notes = [item for item in groups[i][1:-1] if item.name == "Note"]
            n_notes = len(notes)
            velocities = np.array([item.velocity for item in notes])
            pitches = np.array([item.pitch for item in notes])
            durations = np.array([item.end - item.start for item in notes])

            note_density = n_notes / positions_per_bar
            index = np.argmin(abs(DEFAULT_NOTE_DENSITY_BINS - note_density))
            events.append(Event(name=NOTE_DENSITY_KEY, time=None, value=index, text="{:.2f}/{:.2f}".format(note_density, DEFAULT_NOTE_DENSITY_BINS[index])))

            # will be 0 if there's no notes
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

            # will be 0 if there's no notes
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

            # will be 1 if there's no notes
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

        if not omit_instruments:
            instruments = set([item.instrument for item in notes])
            for instrument in instruments:
                instrument = pm_utils.program_to_instrument_name(instrument) if instrument != "drum" else "drum"
                events.append(Event(name=INSTRUMENT_KEY, time=None, value=instrument, text=instrument))

        if not omit_chords:
            chords = [item for item in groups[i][1:-1] if item.name == "Chord"]
            if len(chords) == 0 and current_chord is not None:
                chords = [current_chord]
            elif len(chords) > 0:
                if chords[0].start > bar_st and current_chord is not None:
                    chords.insert(0, current_chord)
                current_chord = chords[-1]

            for chord in chords:
                events.append(Event(name=CHORD_KEY, time=None, value=chord.pitch, text="{}".format(chord.pitch)))

    return [f"{e.name}_{e.value}" for e in events]


def preprocess_description(desc, desc_vocab=desc_vocab):
    desc = "\n".join(re.findall(r"<[^>]+>", desc.strip()))
    desc = re.sub(r"[<>]", "", desc)
    desc = desc.replace("_Drums", "_drum")
    desc = desc.split("\n")
    check_description(desc, desc_vocab=desc_vocab)
    return desc


def check_description(desc, desc_vocab=desc_vocab):
    desc_ids = desc_vocab.encode(desc)
    tokens = desc_vocab.decode(desc_ids)
    if len(desc) != len(tokens):
        logging.error("Number of tokens was different after decoding, not sure what happened.")
    for desc_token, decoded_token in zip(desc, tokens):
        if desc_token != decoded_token:
            logging.error(f"Unable to encode token '{desc_token}' (was encoded to '{decoded_token}')")

    # TODO: check if the description is valid
    # check if it has the right order (meta tokens -> instruments -> chords)
    # check if it has the right meta tokens in the right order


def estimate_number_of_tokens(desc, desc_vocab=desc_vocab):
    desc_events = preprocess_description(desc, desc_vocab=desc_vocab)
    time_signatures = [tuple(int(x) for x in event.split("_")[-1].split("/")) for event in desc_events if f"{TIME_SIGNATURE_KEY}_" in event]
    note_densities = [int(event.split("_")[-1]) for event in desc_events if f"{NOTE_DENSITY_KEY}_" in event]
    num_quarters = [num / denom * 4 for num, denom in time_signatures]
    densities = [DEFAULT_NOTE_DENSITY_BINS[d] for d in note_densities]

    expected_notes = DEFAULT_POS_PER_QUARTER * sum(q * d for q, d in zip(num_quarters, densities))
    n_bars = len(time_signatures)

    # it takes 5 tokens for every note, each bar usually has 6 token at the beginning
    return 5 * expected_notes + 6 * n_bars


def make_example_from_description(description: str, desc_vocab=desc_vocab, remi_vocab=remi_vocab):
    desc_events = preprocess_description(description, desc_vocab=desc_vocab)
    desc_bars = [i for i, event in enumerate(desc_events) if f"{BAR_KEY}_" in event]
    assert len(desc_bars) < 512, "The maximum number of allowed bars is 511."

    desc_bar_ids = torch.zeros(len(desc_events), dtype=torch.int)
    desc_bar_ids[desc_bars] = 1
    desc_bar_ids = torch.cumsum(desc_bar_ids, dim=0)

    zero = torch.tensor([0], dtype=torch.int)

    desc_ids = torch.tensor(desc_vocab.encode([BOS_TOKEN] + desc_events + [EOS_TOKEN]), dtype=torch.int)
    desc_bar_ids = torch.cat([zero, desc_bar_ids, zero])

    input_ids = torch.tensor(remi_vocab.encode([BOS_TOKEN]), dtype=torch.int)
    position_ids = torch.tensor([0], dtype=torch.int)
    bar_ids = torch.tensor([0], dtype=torch.int)

    return {
        "description": desc_ids,
        "desc_bar_ids": desc_bar_ids,
        "input_ids": input_ids,
        "position_ids": position_ids,
        "bar_ids": bar_ids,
    }
