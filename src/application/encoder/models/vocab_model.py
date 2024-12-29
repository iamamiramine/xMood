from collections import Counter

import torch

from pretty_midi.utilities import program_to_instrument_name

from src.domain.constants.encoder.harmony_constants import get_all_major_minor_keys, get_pitch_classes, get_chord_qualities
from src.domain.constants.encoder.midi_constants import (
    DEFAULT_VELOCITY_BINS,
    DEFAULT_DURATION_BINS,
    DEFAULT_TEMPO_BINS,
    DEFAULT_POS_PER_QUARTER,
    DEFAULT_NOTE_DENSITY_BINS,
    DEFAULT_MEAN_VELOCITY_BINS,
    DEFAULT_MEAN_PITCH_BINS,
    DEFAULT_MEAN_DURATION_BINS,
    MAX_BAR_LENGTH,
    MAX_N_BARS,
)


from src.domain.constants.encoder.token_constants import (
    PAD_TOKEN,
    UNK_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    MASK_TOKEN,
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
    NOTE_DENSITY_KEY,
    MEAN_PITCH_KEY,
    MEAN_VELOCITY_KEY,
    MEAN_DURATION_KEY,
)


class Tokens:
    def get_instrument_tokens(key=INSTRUMENT_KEY):
        tokens = [f"{key}_{program_to_instrument_name(i)}" for i in range(128)]
        tokens.append(f"{key}_drum")
        return tokens

    def get_chord_tokens(key=CHORD_KEY):
        qualities = get_chord_qualities()
        pitch_classes = get_pitch_classes()

        chords = [f"{root}:{quality}" for root in pitch_classes for quality in qualities]
        # chords = [f'{root}:{quality}:{degree}' for root in pitch_classes for quality in qualities for degree in ['1', '3', '5', '7']]
        chords.append("N:N")

        tokens = [f"{key}_{chord}" for chord in chords]
        return tokens

    def get_time_signature_tokens(key=TIME_SIGNATURE_KEY):
        denominators = [2, 4, 8, 16]
        time_sigs = [f"{p}/{q}" for q in denominators for p in range(1, MAX_BAR_LENGTH * q + 1)]
        tokens = [f"{key}_{time_sig}" for time_sig in time_sigs]
        return tokens

    def get_key_signature_tokens(key=KEY_SIGNATURE_KEY):
        key_classes = get_all_major_minor_keys()

        keys = [f"{key_classes[i][0]}:{key_classes[i][1]}" for i, _ in enumerate(key_classes)]

        tokens = [f"{key}_{key_sig}" for key_sig in keys]
        return tokens

    def get_midi_tokens(
        instrument_key=INSTRUMENT_KEY,
        key_signature_key=KEY_SIGNATURE_KEY,
        time_signature_key=TIME_SIGNATURE_KEY,
        pitch_key=PITCH_KEY,
        velocity_key=VELOCITY_KEY,
        duration_key=DURATION_KEY,
        tempo_key=TEMPO_KEY,
        bar_key=BAR_KEY,
        position_key=POSITION_KEY,
    ):
        instrument_tokens = Tokens.get_instrument_tokens(instrument_key)

        pitch_tokens = [f"{pitch_key}_{i}" for i in range(128)] + [f"{pitch_key}_drum_{i}" for i in range(128)]
        velocity_tokens = [f"{velocity_key}_{i}" for i in range(len(DEFAULT_VELOCITY_BINS))]
        duration_tokens = [f"{duration_key}_{i}" for i in range(len(DEFAULT_DURATION_BINS))]
        tempo_tokens = [f"{tempo_key}_{i}" for i in range(len(DEFAULT_TEMPO_BINS))]
        bar_tokens = [f"{bar_key}_{i}" for i in range(MAX_N_BARS)]
        position_tokens = [f"{position_key}_{i}" for i in range(MAX_BAR_LENGTH * 4 * DEFAULT_POS_PER_QUARTER)]

        time_sig_tokens = Tokens.get_time_signature_tokens(time_signature_key)
        key_sig_tokens = Tokens.get_key_signature_tokens(key_signature_key)

        return (
            key_sig_tokens
            + time_sig_tokens
            + tempo_tokens
            + instrument_tokens
            + pitch_tokens
            + velocity_tokens
            + duration_tokens
            + bar_tokens
            + position_tokens
        )



class Vocab:
    def __init__(
        self,
        counter,
        specials=[
            PAD_TOKEN,
            UNK_TOKEN,
            BOS_TOKEN,
            EOS_TOKEN,
            MASK_TOKEN,
        ],
        unk_token=UNK_TOKEN,
    ):
        # Initialize dictionaries for token-to-index (stoi) and index-to-token (itos)
        self.specials = specials
        self.unk_token = unk_token
        self.stoi = {}
        self.itos = []

        # Add special tokens first
        for i, token in enumerate(specials):
            self.stoi[token] = i
            self.itos.append(token)

        # Add the rest of the tokens from the counter
        for token, _ in counter.items():
            if token not in self.stoi:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)

        # Set default index for unknown tokens
        if unk_token in specials:
            self.default_index = self.stoi[unk_token]
        else:
            self.default_index = -1  # Handle cases without unknown token

    def to_i(self, token):
        # Return the index of the token, or the index of the unknown token
        return self.stoi.get(token, self.default_index)

    def to_s(self, idx):
        # Return the token corresponding to the index
        if idx >= len(self.itos):
            return self.unk_token
        else:
            return self.itos[idx]

    def __len__(self):
        # Return the size of the vocabulary
        return len(self.itos)

    def encode(self, seq):
        # Encode a sequence of tokens into their indices
        return [self.to_i(token) for token in seq]

    def decode(self, seq):
        # Decode a sequence of indices into their corresponding tokens
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()
        return [self.to_s(idx) for idx in seq]


class RemiVocab(Vocab):
    def __init__(self):
        midi_tokens = Tokens.get_midi_tokens()
        chord_tokens = Tokens.get_chord_tokens()
        key_tokens = Tokens.get_key_signature_tokens()

        self.tokens = midi_tokens + chord_tokens + key_tokens

        counter = Counter(self.tokens)
        super().__init__(counter)


class DescriptionVocab(Vocab):
    def __init__(self):
        key_sig_tokens = Tokens.get_key_signature_tokens()
        time_sig_tokens = Tokens.get_time_signature_tokens()
        instrument_tokens = Tokens.get_instrument_tokens()
        chord_tokens = Tokens.get_chord_tokens()

        bar_tokens = [f"Bar_{i}" for i in range(MAX_N_BARS)]
        density_tokens = [f"{NOTE_DENSITY_KEY}_{i}" for i in range(len(DEFAULT_NOTE_DENSITY_BINS))]
        velocity_tokens = [f"{MEAN_VELOCITY_KEY}_{i}" for i in range(len(DEFAULT_MEAN_VELOCITY_BINS))]
        pitch_tokens = [f"{MEAN_PITCH_KEY}_{i}" for i in range(len(DEFAULT_MEAN_PITCH_BINS))]
        duration_tokens = [f"{MEAN_DURATION_KEY}_{i}" for i in range(len(DEFAULT_MEAN_DURATION_BINS))]

        self.tokens = (
            key_sig_tokens + time_sig_tokens + instrument_tokens + chord_tokens + density_tokens + velocity_tokens + pitch_tokens + duration_tokens + bar_tokens
        )

        counter = Counter(self.tokens)
        super().__init__(counter)
