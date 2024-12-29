import torch

from src.domain.constants.encoder.token_constants import (
    BAR_KEY,
    POSITION_KEY,
    BOS_TOKEN,
    EOS_TOKEN,
    INSTRUMENT_KEY,
    CHORD_KEY,
    TIME_SIGNATURE_KEY,
    NOTE_DENSITY_KEY,
    MEAN_PITCH_KEY,
    MEAN_VELOCITY_KEY,
    MEAN_DURATION_KEY,
)


def get_bars(events, include_ids=False):
    bars = [i for i, event in enumerate(events) if f"{BAR_KEY}_" in event]

    if include_ids:
        bar_ids = torch.bincount(torch.tensor(bars, dtype=torch.int), minlength=len(events))
        bar_ids = torch.cumsum(bar_ids, dim=0)

        return bars, bar_ids
    else:
        return bars


def get_positions(events):
    events = [f"{POSITION_KEY}_0" if f"{BAR_KEY}_" in event else event for event in events]
    position_events = [event if f"{POSITION_KEY}_" in event else None for event in events]

    positions = [int(pos.split("_")[-1]) if pos is not None else None for pos in position_events]

    if positions[0] is None:
        positions[0] = 0
    for i in range(1, len(positions)):
        if positions[i] is None:
            positions[i] = positions[i - 1]
    positions = torch.tensor(positions, dtype=torch.int)

    return positions


def mask_bar_tokens(events, bar_token_mask="<mask>"):
    events = [bar_token_mask if f"{BAR_KEY}_" in token else token for token in events]
    return events


def get_bos_eos_events(vocab, tuple_size=8):
    bos_event = torch.tensor(vocab.encode([BOS_TOKEN]), dtype=torch.long)
    eos_event = torch.tensor(vocab.encode([EOS_TOKEN]), dtype=torch.long)
    return bos_event, eos_event


def preprocess_description(desc, instruments=True, chords=True, meta=True):
    valid_keys = {
        BAR_KEY: True,
        INSTRUMENT_KEY: instruments,
        CHORD_KEY: chords,
        TIME_SIGNATURE_KEY: meta,
        NOTE_DENSITY_KEY: meta,
        MEAN_PITCH_KEY: meta,
        MEAN_VELOCITY_KEY: meta,
        MEAN_DURATION_KEY: meta,
    }
    return [token for token in desc if len(token.split("_")) == 0 or valid_keys[token.split("_")[0]]]
