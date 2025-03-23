# Standard library imports
import torch

# Constants - Token types
from domain.constants.encoder.token_constants import (
    # Special tokens
    BOS_TOKEN,
    EOS_TOKEN,
    # Musical feature tokens
    BAR_KEY,
    POSITION_KEY,
)


def get_bars(events, include_ids=False):
    """
    Extract bar positions and optionally generate bar IDs from a sequence of REMI events.

    This function serves two purposes:
    1. Locate all bar marker events in the sequence
    2. Optionally generate cumulative bar IDs for each position in the sequence

    The function identifies bar positions by looking for events that contain the BAR_KEY
    prefix (e.g., 'Bar_1', 'Bar_2', etc.). When include_ids is True, it also generates
    a tensor where each position is labeled with its corresponding bar number.

    Args:
        events (list[str]): List of REMI event strings to process
        include_ids (bool, optional): Whether to generate cumulative bar IDs.
            If True, returns both bar positions and bar IDs tensor.
            If False, returns only bar positions. Defaults to False.

    Returns:
        Union[list[int], tuple[list[int], torch.Tensor]]:
            - If include_ids=False: List of indices where bar markers occur
            - If include_ids=True: Tuple containing:
                - list[int]: Indices where bar markers occur
                - torch.Tensor: Tensor of bar IDs for each position (shape: [len(events)])

    Example:
        >>> events = ['Bar_1', 'Note_C4', 'Bar_2', 'Note_E4']
        >>> positions = get_bars(events)
        >>> print(positions)
        [0, 2]  # Indices where 'Bar_' events occur

        >>> positions, bar_ids = get_bars(events, include_ids=True)
        >>> print(bar_ids)
        tensor([1, 1, 2, 2])  # Each position labeled with its bar number

    Note:
        The bar IDs tensor uses cumulative sum to create a sequence where each
        position between bar markers gets assigned the current bar number.
    """
    # Find indices of all bar marker events
    # List comprehension creates a list of indices where BAR_KEY appears in event strings
    bars = [i for i, event in enumerate(events) if f"{BAR_KEY}_" in event]

    if include_ids:
        # Create a binary tensor marking bar positions (1 where bars occur, 0 elsewhere)
        # bincount converts the bar indices into a binary occurrence vector
        bar_ids = torch.bincount(torch.tensor(bars, dtype=torch.int), minlength=len(events))

        # Compute cumulative sum to create bar numbers
        # This makes all positions between bar i and bar i+1 have value i+1
        bar_ids = torch.cumsum(bar_ids, dim=0)

        return bars, bar_ids
    else:
        return bars


def get_positions(events):
    """
    Extract and normalize position values from a sequence of REMI events.

    This function processes a sequence of REMI events to create a continuous sequence
    of position values, handling bar markers and missing positions. It performs three
    main steps:
    1. Initializes positions at bar markers to 0
    2. Extracts position values from position events
    3. Forward-fills missing positions with the last known position

    The function ensures that:
    - Each bar marker gets position 0
    - Missing positions are filled with the previous position value
    - All positions are converted to a tensor of integers

    Args:
        events (list[str]): List of REMI event strings to process. Events can be:
            - Bar markers (e.g., 'Bar_1')
            - Position markers (e.g., 'Position_32')
            - Other event types (treated as None for position)

    Returns:
        torch.Tensor: Integer tensor containing position values for each event.
            Shape: [len(events)]

    Example:
        >>> events = ['Bar_1', 'Position_0', 'Note_C4', 'Position_12', 'Bar_2']
        >>> positions = get_positions(events)
        >>> print(positions)
        tensor([0, 0, 0, 12, 0])

    Note:
        - Position values at bar markers are always set to 0
        - If the first event has no position, it's initialized to 0
        - Missing positions are filled with the last valid position
        - This creates a continuous sequence suitable for timing calculations
    """
    # Replace bar markers with position 0 events
    # This ensures every bar starts at position 0
    events = [f"{POSITION_KEY}_0" if f"{BAR_KEY}_" in event else event for event in events]

    # Extract position events, replacing non-position events with None
    # This creates a list of either position events or None values
    position_events = [event if f"{POSITION_KEY}_" in event else None for event in events]

    # Convert position events to integer values
    # For each position event, extract the number after the underscore
    # If the event is None, keep it as None
    positions = [int(pos.split("_")[-1]) if pos is not None else None for pos in position_events]

    # Initialize first position to 0 if it's None
    # This ensures we have a starting point for forward-filling
    if positions[0] is None:
        positions[0] = 0

    # Forward-fill missing positions
    # Iterate through positions and replace None values with the previous position
    for i in range(1, len(positions)):
        if positions[i] is None:
            positions[i] = positions[i - 1]

    # Convert list of positions to tensor
    # This creates a continuous sequence of integer position values
    positions = torch.tensor(positions, dtype=torch.int)

    return positions


def mask_bar_tokens(events, bar_token_mask="<mask>"):
    """
    Replace all bar marker tokens in a sequence with a mask token.

    This function is typically used in training scenarios where we want to:
    1. Hide bar position information from the model
    2. Create masked sequences for prediction tasks
    3. Standardize bar marker representation

    The function identifies bar markers using the BAR_KEY prefix and replaces them
    with a specified mask token, leaving all other events unchanged.

    Args:
        events (list[str]): List of REMI event strings to process. Events can be:
            - Bar markers (e.g., 'Bar_1', 'Bar_2')
            - Any other REMI events (left unchanged)
        bar_token_mask (str, optional): Token to use for masking bar markers.
            Defaults to "<mask>".

    Returns:
        list[str]: New list where all bar marker tokens are replaced with the mask token.
            The length of the output list equals the length of the input list.

    Example:
        >>> events = ['Bar_1', 'Note_C4', 'Bar_2', 'Note_E4']
        >>> masked = mask_bar_tokens(events)
        >>> print(masked)
        ['<mask>', 'Note_C4', '<mask>', 'Note_E4']

        >>> # Using custom mask token
        >>> masked = mask_bar_tokens(events, bar_token_mask='[MASK]')
        >>> print(masked)
        ['[MASK]', 'Note_C4', '[MASK]', 'Note_E4']

    Note:
        - The function creates a new list rather than modifying the input
        - Only tokens starting with BAR_KEY are masked
        - The mask token can be customized for different use cases
    """
    # Create a new list with masked bar tokens
    # For each token:
    # - If it's a bar marker (starts with BAR_KEY), replace with mask token
    # - Otherwise, keep the original token
    events = [bar_token_mask if f"{BAR_KEY}_" in token else token for token in events]

    return events


def get_bos_eos_events(vocab, tuple_size: int = 8):
    """
    Create beginning-of-sequence (BOS) and end-of-sequence (EOS) event tensors.

    This function converts special BOS and EOS tokens into their corresponding tensor
    representations using the provided vocabulary. These tokens are essential for:
    1. Marking sequence boundaries in training data
    2. Controlling generation start/end in inference
    3. Providing positional context to the model

    The function performs two main steps for each token:
    1. Encodes the token string into vocabulary indices using the provided vocab
    2. Converts the indices into PyTorch tensors with appropriate data type

    Args:
        vocab (Vocab): Vocabulary object with encode/decode methods for converting
            between tokens and indices.
        tuple_size (int, optional): Size of event tuples. This parameter is kept
            for backward compatibility but is not used in current implementation.
            Defaults to 8.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - bos_event: Tensor of shape [1] containing the BOS token index
            - eos_event: Tensor of shape [1] containing the EOS token index
            Both tensors are of type torch.long (int64)

    Example:
        >>> vocab = RemiVocab()  # Initialize vocabulary
        >>> bos, eos = get_bos_eos_events(vocab)
        >>> print(bos, eos)
        tensor([1]) tensor([2])  # Example indices for BOS and EOS tokens

    Note:
        - The tensors are created on CPU but can be moved to GPU if needed
        - The function assumes BOS_TOKEN and EOS_TOKEN are defined in the vocabulary
        - The returned tensors are 1-dimensional with a single value each
    """
    # Convert BOS token to tensor
    # 1. Wrap token in list for vocab.encode which expects a sequence
    # 2. Convert to tensor of type long (required for embedding layers)
    bos_event = torch.tensor(vocab.encode([BOS_TOKEN]), dtype=torch.long)

    # Convert EOS token to tensor
    # Same process as BOS token
    eos_event = torch.tensor(vocab.encode([EOS_TOKEN]), dtype=torch.long)

    return bos_event, eos_event
