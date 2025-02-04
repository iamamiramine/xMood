# Standard library imports
from collections import Counter

# PyTorch imports
import torch

# Third-party music processing
from pretty_midi.utilities import program_to_instrument_name

# Constants - Harmony
from domain.constants.encoder.harmony_constants import get_all_major_minor_keys, get_pitch_classes, get_chord_qualities

# Constants - MIDI parameters
from domain.constants.encoder.midi_constants import (
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

# Constants - Token types
from domain.constants.encoder.token_constants import (
    # Special tokens
    PAD_TOKEN,
    UNK_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    MASK_TOKEN,
    # Musical feature tokens
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
    # Statistical feature tokens
    NOTE_DENSITY_KEY,
    MEAN_PITCH_KEY,
    MEAN_VELOCITY_KEY,
    MEAN_DURATION_KEY,
    # Emotion tokens
    ANGER_KEY,
    LOVE_KEY,
    SADNESS_KEY,
    JOY_KEY,
    SURPRISE_KEY,
    ADMIRATION_KEY,
    AMUSEMENT_KEY,
    ANNOYANCE_KEY,
    APPROVAL_KEY,
    CARING_KEY,
    CONFUSION_KEY,
    CURIOSITY_KEY,
    DESIRE_KEY,
    DISAPPOINTMENT_KEY,
    DISAPPROVAL_KEY,
    DISGUST_KEY,
    EMBARRASSMENT_KEY,
    EXCITEMENT_KEY,
    FEAR_KEY,
    GRATITUDE_KEY,
    GRIEF_KEY,
    NERVOUSNESS_KEY,
    NEUTRAL_KEY,
    OPTIMISM_KEY,
    PRIDE_KEY,
    REALIZATION_KEY,
    RELIEF_KEY,
    REMORSE_KEY,
)


class Tokens:
    """
    A utility class for generating various types of tokens used in music representation.

    This class provides static methods to create standardized token strings for different
    musical elements such as instruments, chords, time signatures, and key signatures.
    Each token is prefixed with its type (e.g., 'instrument_piano', 'chord_C:maj')
    to maintain clear categorization in the vocabulary.
    """

    @staticmethod
    def get_instrument_tokens(key=INSTRUMENT_KEY):
        """
        Generate tokens for all possible MIDI instruments plus drums.

        Creates tokens for all 128 standard MIDI program numbers, mapping them to
        their instrument names, and adds a special token for drums. Each token
        is prefixed with the instrument key.

        Args:
            key (str, optional): Prefix for instrument tokens. Defaults to INSTRUMENT_KEY.

        Returns:
            list[str]: List of instrument tokens in format f"{key}_{instrument_name}".
                      Example: ["instrument_acoustic_grand_piano", "instrument_drum"]
        """
        # Create tokens for all 128 MIDI programs using their instrument names
        tokens = [f"{key}_{program_to_instrument_name(i)}" for i in range(128)]
        # Add special token for drums
        tokens.append(f"{key}_drum")
        return tokens

    @staticmethod
    def get_chord_tokens(key=CHORD_KEY):
        """
        Generate tokens for all possible chord combinations.

        Combines pitch classes (e.g., C, C#, D) with chord qualities (e.g., maj, min)
        to create tokens for all possible chords. Also includes a special "N:N" token
        for no chord or undefined chord situations.

        Args:
            key (str, optional): Prefix for chord tokens. Defaults to CHORD_KEY.

        Returns:
            list[str]: List of chord tokens in format f"{key}_{root}:{quality}".
                      Examples: ["chord_C:maj", "chord_F#:min", "chord_N:N"]
        """
        # Get all possible chord components
        qualities = get_chord_qualities()  # Get chord qualities (maj, min, etc.)
        pitch_classes = get_pitch_classes()  # Get pitch classes (C, C#, D, etc.)

        # Create all possible combinations of root notes and qualities
        chords = [f"{root}:{quality}" for root in pitch_classes for quality in qualities]
        # Add special token for no chord or undefined chord
        chords.append("N:N")

        # Prefix all chord combinations with the chord key
        tokens = [f"{key}_{chord}" for chord in chords]
        return tokens

    @staticmethod
    def get_time_signature_tokens(key=TIME_SIGNATURE_KEY):
        """
        Generate tokens for all supported time signatures.

        Creates tokens for common time signatures using standard denominators (2,4,8,16)
        and numerators up to the maximum bar length. This covers all standard and most
        compound time signatures used in music.

        Args:
            key (str, optional): Prefix for time signature tokens. Defaults to TIME_SIGNATURE_KEY.

        Returns:
            list[str]: List of time signature tokens in format f"{key}_{numerator}/{denominator}".
                      Examples: ["time_4/4", "time_3/4", "time_6/8"]
        """
        # Standard time signature denominators
        denominators = [2, 4, 8, 16]  # Common time signature denominators
        # Generate all valid time signatures up to MAX_BAR_LENGTH
        time_sigs = [f"{p}/{q}" for q in denominators for p in range(1, MAX_BAR_LENGTH * q + 1)]
        # Prefix with time signature key
        tokens = [f"{key}_{time_sig}" for time_sig in time_sigs]
        return tokens

    @staticmethod
    def get_key_signature_tokens(key=KEY_SIGNATURE_KEY):
        """
        Generate tokens for all possible musical key signatures.

        Creates tokens for all major and minor keys using the standard
        key signature format (e.g., C:maj, A:min). This covers all
        possible key signatures in Western music.

        Args:
            key (str, optional): Prefix for key signature tokens. Defaults to KEY_SIGNATURE_KEY.

        Returns:
            list[str]: List of key signature tokens in format f"{key}_{root}:{mode}".
                      Examples: ["key_C:maj", "key_A:min"]
        """
        # Get all possible major and minor keys
        key_classes = get_all_major_minor_keys()  # List of (root, mode) tuples

        # Create key signature strings from root and mode
        keys = [f"{key_classes[i][0]}:{key_classes[i][1]}" for i, _ in enumerate(key_classes)]

        # Prefix with key signature key
        tokens = [f"{key}_{key_sig}" for key_sig in keys]
        return tokens

    @staticmethod
    def get_midi_tokens(
        instrument_key=INSTRUMENT_KEY,
        time_signature_key=TIME_SIGNATURE_KEY,
        pitch_key=PITCH_KEY,
        velocity_key=VELOCITY_KEY,
        duration_key=DURATION_KEY,
        tempo_key=TEMPO_KEY,
        bar_key=BAR_KEY,
        position_key=POSITION_KEY,
    ):
        """
        Generate all MIDI-related tokens needed for music representation.

        Combines various types of tokens (instruments, pitches, velocities, etc.)
        into a complete set of tokens needed to represent MIDI music data.

        Args:
            instrument_key (str, optional): Prefix for instrument tokens
            time_signature_key (str, optional): Prefix for time signature tokens
            pitch_key (str, optional): Prefix for pitch tokens
            velocity_key (str, optional): Prefix for velocity tokens
            duration_key (str, optional): Prefix for duration tokens
            tempo_key (str, optional): Prefix for tempo tokens
            bar_key (str, optional): Prefix for bar tokens
            position_key (str, optional): Prefix for position tokens

        Returns:
            list[str]: Combined list of all MIDI-related tokens
        """
        # Generate instrument tokens (including drums)
        instrument_tokens = Tokens.get_instrument_tokens(instrument_key)

        # Generate tokens for MIDI note properties
        pitch_tokens = [f"{pitch_key}_{i}" for i in range(128)] + [f"{pitch_key}_drum_{i}" for i in range(128)]  # Regular and drum pitches
        velocity_tokens = [f"{velocity_key}_{i}" for i in range(len(DEFAULT_VELOCITY_BINS))]  # Note velocities
        duration_tokens = [f"{duration_key}_{i}" for i in range(len(DEFAULT_DURATION_BINS))]  # Note durations
        tempo_tokens = [f"{tempo_key}_{i}" for i in range(len(DEFAULT_TEMPO_BINS))]  # Tempo markings

        # Generate tokens for musical structure
        bar_tokens = [f"{bar_key}_{i}" for i in range(MAX_N_BARS)]  # Bar numbers
        position_tokens = [f"{position_key}_{i}" for i in range(MAX_BAR_LENGTH * 4 * DEFAULT_POS_PER_QUARTER)]  # Positions within bars

        # Get time signature tokens
        time_sig_tokens = Tokens.get_time_signature_tokens(time_signature_key)

        # Combine all token types in a specific order
        return time_sig_tokens + tempo_tokens + instrument_tokens + pitch_tokens + velocity_tokens + duration_tokens + bar_tokens + position_tokens


class Vocab:
    """
    A vocabulary class for managing token-to-index and index-to-token mappings.

    This class provides functionality for:
    1. Creating a vocabulary from a collection of tokens
    2. Converting between tokens and their numerical indices
    3. Handling special tokens (PAD, UNK, BOS, EOS, MASK)
    4. Encoding and decoding sequences of tokens

    The vocabulary maintains two primary data structures:
    - stoi (str -> int): Maps tokens to their indices
    - itos (int -> str): Maps indices to their tokens

    Special tokens are given priority and assigned the lowest indices.
    """

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
        """
        Initialize the vocabulary with tokens and special tokens.

        Args:
            counter (Counter): Collection of tokens to include in the vocabulary
            specials (list[str], optional): List of special tokens to add first.
                                          Defaults to [PAD, UNK, BOS, EOS, MASK].
            unk_token (str, optional): Token to use for unknown tokens.
                                     Defaults to UNK_TOKEN.

        Note:
            Special tokens are assigned the lowest indices in the vocabulary.
            The unknown token's index is used as the default for out-of-vocabulary tokens.
        """
        # Initialize mapping dictionaries
        self.specials = specials  # Store special tokens for reference
        self.unk_token = unk_token  # Token to use for unknown words
        self.stoi = {}  # String to Index mapping
        self.itos = []  # Index to String mapping

        # Add special tokens first to ensure they get the lowest indices
        for i, token in enumerate(specials):
            self.stoi[token] = i  # Assign index to token
            self.itos.append(token)  # Store token at index

        # Add remaining tokens from the counter
        for token, _ in counter.items():
            if token not in self.stoi:  # Skip if token is already in vocabulary
                self.stoi[token] = len(self.itos)  # Assign next available index
                self.itos.append(token)  # Store token at index

        # Set up handling for unknown tokens
        if unk_token in specials:
            self.default_index = self.stoi[unk_token]  # Use UNK token index
        else:
            self.default_index = -1  # Use -1 if no UNK token specified

    def to_i(self, token):
        """
        Convert a token to its corresponding index.

        Args:
            token (str): Token to convert to index

        Returns:
            int: Index of the token, or default_index if token is not in vocabulary
        """
        return self.stoi.get(token, self.default_index)

    def to_s(self, idx):
        """
        Convert an index to its corresponding token.

        Args:
            idx (int): Index to convert to token

        Returns:
            str: Token at the given index, or unknown token if index is out of range
        """
        if idx >= len(self.itos):
            return self.unk_token
        else:
            return self.itos[idx]

    def __len__(self):
        """
        Get the size of the vocabulary.

        Returns:
            int: Total number of tokens in the vocabulary (including special tokens)
        """
        return len(self.itos)

    def encode(self, seq):
        """
        Encode a sequence of tokens into their corresponding indices.

        Args:
            seq (list[str]): Sequence of tokens to encode

        Returns:
            list[int]: List of indices corresponding to the input tokens

        Note:
            Out-of-vocabulary tokens are mapped to the default_index
        """
        return [self.to_i(token) for token in seq]

    def decode(self, seq):
        """
        Decode a sequence of indices back into tokens.

        Args:
            seq (list[int] or torch.Tensor): Sequence of indices to decode

        Returns:
            list[str]: List of tokens corresponding to the input indices

        Note:
            If input is a torch.Tensor, it is first converted to a numpy array
        """
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()  # Convert tensor to numpy array
        return [self.to_s(idx) for idx in seq]


class RemiVocab(Vocab):
    """
    A specialized vocabulary class for REMI (REvamped MIDI) representation of music.

    This class creates a comprehensive vocabulary for tokenizing MIDI music data,
    including:
    1. MIDI-specific tokens (notes, velocities, durations, etc.)
    2. Chord tokens for harmonic representation
    3. Key signature tokens for tonal context

    The vocabulary follows the REMI format, which represents music as a sequence
    of tokens that capture both the musical content and structure. This allows
    for effective encoding of musical pieces while preserving their musical meaning.

    Inherits from:
        Vocab: Base vocabulary class providing token-index conversion functionality
    """

    def __init__(self):
        """
        Initialize the REMI vocabulary with all necessary musical tokens.

        The initialization process:
        1. Generates MIDI-related tokens (instruments, notes, timing, etc.)
        2. Generates chord tokens for harmonic representation
        3. Generates key signature tokens for tonal context
        4. Combines all tokens and creates the vocabulary mapping

        Note:
            The token order (MIDI -> chord -> key) is maintained for consistency
            and to ensure deterministic token indices across model usage.
        """
        # Generate MIDI-specific tokens
        # Includes time signatures, tempo, instruments, pitches, velocities, durations, bars, positions
        midi_tokens = Tokens.get_midi_tokens()

        # Generate chord tokens
        # Represents all possible chord combinations (e.g., C:maj, F#:min)
        chord_tokens = Tokens.get_chord_tokens()

        # Generate key signature tokens
        # Represents all possible musical keys (e.g., C:maj, A:min)
        key_tokens = Tokens.get_key_signature_tokens()

        # Combine all tokens in a specific order
        # Order matters for consistent token indices across model usage
        self.tokens = midi_tokens + chord_tokens + key_tokens

        # Create counter from combined tokens
        # Convert token list to Counter for vocabulary creation
        counter = Counter(self.tokens)

        # Initialize parent Vocab class
        # This sets up the token-to-index and index-to-token mappings
        super().__init__(counter)


class SymbolicFeaturesVocab(Vocab):
    """
    A specialized vocabulary class for musical piece symbolic features and statistical features.

    This class creates a vocabulary that combines multiple aspects of musical symbolic features:
    1. Musical Context:
       - Key signatures (e.g., C:maj, A:min)
       - Time signatures (e.g., 4/4, 3/4)
       - Instruments (e.g., piano, violin)
       - Chords (e.g., C:maj, F#:min)

    2. Statistical Features:
       - Note density (notes per time unit)
       - Mean velocity (average note loudness)
       - Mean pitch (average note height)
       - Mean duration (average note length)

    3. Structural Elements:
       - Bar markers for temporal organization

    Inherits from:
        Vocab: Base vocabulary class providing token-index conversion functionality
    """

    def __init__(self):
        """
        Initialize the symbolic features vocabulary with musical and statistical tokens.

        The initialization process combines tokens in the following order:
        1. Musical context tokens (key, time signature, instruments, chords)
        2. Statistical feature tokens (density, velocity, pitch, duration)
        3. Structural tokens (bar markers)

        The order is maintained for consistent token indices across model usage.
        """
        # Generate musical context tokens
        # Key signature tokens (e.g., "key_C:maj", "key_A:min")
        key_sig_tokens = Tokens.get_key_signature_tokens()
        # Time signature tokens (e.g., "time_4/4", "time_3/4")
        time_sig_tokens = Tokens.get_time_signature_tokens()
        # Instrument tokens (e.g., "instrument_piano", "instrument_violin")
        instrument_tokens = Tokens.get_instrument_tokens()
        # Chord tokens (e.g., "chord_C:maj", "chord_F#:min")
        chord_tokens = Tokens.get_chord_tokens()

        # Generate statistical feature tokens
        # Bar position markers (e.g., "Bar_0", "Bar_1", ...)
        bar_tokens = [f"{BAR_KEY}_{i}" for i in range(MAX_N_BARS)]
        # Position within bar markers (e.g., "Position_0", "Position_1", ...)
        position_tokens = [f"{POSITION_KEY}_{i}" for i in range(MAX_BAR_LENGTH * 4 * DEFAULT_POS_PER_QUARTER)]  # Positions within bars
        # Note density tokens (number of notes per time unit)
        density_tokens = [f"{NOTE_DENSITY_KEY}_{i}" for i in range(len(DEFAULT_NOTE_DENSITY_BINS))]
        # Mean velocity tokens (average note loudness)
        velocity_tokens = [f"{MEAN_VELOCITY_KEY}_{i}" for i in range(len(DEFAULT_MEAN_VELOCITY_BINS))]
        # Mean pitch tokens (average note height)
        pitch_tokens = [f"{MEAN_PITCH_KEY}_{i}" for i in range(len(DEFAULT_MEAN_PITCH_BINS))]
        # Mean duration tokens (average note length)
        duration_tokens = [f"{MEAN_DURATION_KEY}_{i}" for i in range(len(DEFAULT_MEAN_DURATION_BINS))]

        # Combine all tokens in a specific order
        # Order: Musical context -> Statistical features -> Structural markers
        self.tokens = (
            key_sig_tokens
            + time_sig_tokens  # Musical key context
            + instrument_tokens  # Rhythmic structure
            + chord_tokens  # Timbral information
            + density_tokens  # Harmonic content
            + velocity_tokens  # Note density statistics
            + pitch_tokens  # Dynamic level statistics
            + duration_tokens  # Pitch height statistics
            + bar_tokens  # Note length statistics  # Structural organization
            + position_tokens  # Temporal organization
        )

        # Create counter and initialize vocabulary
        # Convert token list to Counter for vocabulary creation
        counter = Counter(self.tokens)
        # Initialize parent Vocab class with the combined tokens
        super().__init__(counter)


class EmotionVocab(Vocab):
    """
    A specialized vocabulary class for representing musical emotions and their temporal progression.

    This class creates a vocabulary that combines:
    1. Structural Elements:
       - Bar markers for temporal organization of emotions

    2. Emotion Categories (0.0000-1.0000 intensity for each):
       - All standard emotions from the emotion classification model
       Each emotion intensity is represented with 4 decimal precision

    Each emotion is represented with 4 decimal precision (0.0000-1.0000),
    allowing for precise emotion intensity representation at each bar.

    Inherits from:
        Vocab: Base vocabulary class providing token-index conversion functionality
    """

    def __init__(self):
        """
        Initialize the emotion vocabulary with structural and emotion intensity tokens.

        The initialization process:
        1. Creates bar position tokens for temporal organization
        2. Creates tokens for each emotion category with 4 decimal precision
        3. Combines tokens in a specific order for consistent indexing

        Token formats:
        - Bar tokens: "Bar_0", "Bar_1", etc.
        - Emotion tokens: "{emotion}_{0.0000}" to "{emotion}_{1.0000}" for each emotion

        Note:
            The decimal range (0.0000-1.0000) allows for fine-grained emotion intensity
            representation, where 0.0000 represents absence and 1.0000 represents maximum
            intensity of that emotion.
        """
        # Generate structural tokens
        bar_tokens = [f"Bar_{i}" for i in range(MAX_N_BARS)]

        # Function to generate emotion tokens with 4 decimal precision
        def generate_emotion_tokens(emotion_key):
            return [f"{emotion_key}_{i/10000:.4f}" for i in range(10001)]

        # Generate emotion intensity tokens for all emotions
        emotion_tokens = []
        for emotion_key in [
            ANGER_KEY,
            LOVE_KEY,
            SADNESS_KEY,
            JOY_KEY,
            SURPRISE_KEY,
            ADMIRATION_KEY,
            AMUSEMENT_KEY,
            ANNOYANCE_KEY,
            APPROVAL_KEY,
            CARING_KEY,
            CONFUSION_KEY,
            CURIOSITY_KEY,
            DESIRE_KEY,
            DISAPPOINTMENT_KEY,
            DISAPPROVAL_KEY,
            DISGUST_KEY,
            EMBARRASSMENT_KEY,
            EXCITEMENT_KEY,
            FEAR_KEY,
            GRATITUDE_KEY,
            GRIEF_KEY,
            NERVOUSNESS_KEY,
            NEUTRAL_KEY,
            OPTIMISM_KEY,
            PRIDE_KEY,
            REALIZATION_KEY,
            RELIEF_KEY,
            REMORSE_KEY,
        ]:
            emotion_tokens.extend(generate_emotion_tokens(emotion_key))

        # Combine all tokens in a specific order
        self.tokens = bar_tokens + emotion_tokens

        # Create counter and initialize vocabulary
        counter = Counter(self.tokens)
        super().__init__(counter)
