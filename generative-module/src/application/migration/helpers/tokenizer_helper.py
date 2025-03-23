import os
from pathlib import Path
from typing import List, Dict, Optional, Union, Set

import torch
import json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

from application.encoder.models.vocab_model import RemiVocab, SymbolicFeaturesVocab, MoodsVocab, EmotionVocab
from domain.constants.encoder.token_constants import (
    PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, MASK_TOKEN,
    INSTRUMENT_KEY, PITCH_KEY, VELOCITY_KEY, DURATION_KEY, TEMPO_KEY,
    BAR_KEY, POSITION_KEY, KEY_SIGNATURE_KEY, TIME_SIGNATURE_KEY, CHORD_KEY
)
from application.migration.models.migration_model import TokenizerParameters


def collect_vocab_tokens() -> List[str]:
    """Collect all tokens from all vocabulary classes"""
    all_tokens = set()
    special_tokens = {PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, MASK_TOKEN}
    
    # Add special tokens first
    all_tokens.update(special_tokens)
    
    # Add tokens from each vocabulary
    remi_vocab = RemiVocab()
    symb_vocab = SymbolicFeaturesVocab()
    moods_vocab = MoodsVocab()
    emotion_vocab = EmotionVocab()
    
    # Add all tokens from RemiVocab
    for i in range(len(remi_vocab)):
        token = remi_vocab.to_s(i)
        all_tokens.add(token)
    
    # Add all tokens from SymbolicFeaturesVocab
    for i in range(len(symb_vocab)):
        token = symb_vocab.to_s(i)
        all_tokens.add(token)
    
    # Add all tokens from MoodsVocab
    for i in range(len(moods_vocab)):
        token = moods_vocab.to_s(i)
        all_tokens.add(token)
    
    # Add all tokens from EmotionVocab
    for i in range(len(emotion_vocab)):
        token = emotion_vocab.to_s(i)
        all_tokens.add(token)
    
    return list(all_tokens)


def write_tokens_to_file(tokens: List[str], output_file: str = "music_tokens.txt") -> str:
    """Write tokens to a file, one per line"""
    with open(output_file, "w", encoding="utf-8") as f:
        for token in tokens:
            f.write(f"{token}\n")
    return output_file


def train_tokenizer(params: TokenizerParameters) -> Tokenizer:
    """Train a new tokenizer on the music vocabulary"""
    # Collect all tokens
    all_tokens = collect_vocab_tokens()
    
    # Write tokens to a file
    token_file = write_tokens_to_file(
        all_tokens, 
        os.path.join(params.output_dir, "music_tokens.txt")
    )
    
    # Make sure output directory exists
    os.makedirs(params.output_dir, exist_ok=True)
    
    # Create and train tokenizer
    if params.use_bpe:
        tokenizer = Tokenizer(models.BPE())
    else:
        tokenizer = Tokenizer(models.Unigram())
    
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    
    # Add special tokens if not provided
    special_tokens = params.special_tokens
    if not special_tokens:
        special_tokens = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, MASK_TOKEN]
    
    trainer = trainers.BpeTrainer(
        vocab_size=params.vocab_size,
        special_tokens=special_tokens
    )
    
    # Train the tokenizer
    tokenizer.train([token_file], trainer)
    
    # Add post-processor for BOS/EOS tokens
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
        pair=f"{BOS_TOKEN} $A {EOS_TOKEN} $B:1 {EOS_TOKEN}:1",
        special_tokens=[
            (BOS_TOKEN, tokenizer.token_to_id(BOS_TOKEN)),
            (EOS_TOKEN, tokenizer.token_to_id(EOS_TOKEN)),
        ],
    )
    
    # Save the tokenizer
    tokenizer_path = os.path.join(params.output_dir, params.tokenizer_filename)
    tokenizer.save(tokenizer_path)
    
    return tokenizer


def convert_to_hf_tokenizer(tokenizer_path: str) -> PreTrainedTokenizerFast:
    """Convert a tokenizers tokenizer to a Hugging Face tokenizer"""
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_path,
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        pad_token=PAD_TOKEN,
        unk_token=UNK_TOKEN,
        mask_token=MASK_TOKEN
    )
    
    # Save vocabulary mapping
    vocab_mapping = {
        "remi": create_vocab_mapping(RemiVocab(), hf_tokenizer),
        "symbolic": create_vocab_mapping(SymbolicFeaturesVocab(), hf_tokenizer),
        "moods": create_vocab_mapping(MoodsVocab(), hf_tokenizer),
        "emotions": create_vocab_mapping(EmotionVocab(), hf_tokenizer)
    }
    
    # Save mapping alongside tokenizer
    tokenizer_dir = os.path.dirname(tokenizer_path)
    with open(os.path.join(tokenizer_dir, "vocab_mapping.json"), "w") as f:
        json.dump(vocab_mapping, f, indent=2)
    
    return hf_tokenizer


def create_vocab_mapping(custom_vocab, hf_tokenizer) -> Dict[str, int]:
    """Create a mapping between custom vocabulary tokens and HF tokenizer IDs"""
    mapping = {}
    
    for i in range(len(custom_vocab)):
        token = custom_vocab.to_s(i)
        if token in hf_tokenizer.get_vocab():
            mapping[token] = hf_tokenizer.get_vocab()[token]
        else:
            # Token might be split into subwords, handle this case
            token_ids = hf_tokenizer.encode(token, add_special_tokens=False)
            if token_ids:
                mapping[token] = token_ids[0]  # Use first subword as identifier
    
    return mapping


def create_and_save_tokenizer(params: TokenizerParameters) -> str:
    """Create, train and save both tokenizers and HF tokenizer"""
    # First create the base tokenizer
    tokenizer = train_tokenizer(params)
    
    # Convert to HF tokenizer
    tokenizer_path = os.path.join(params.output_dir, params.tokenizer_filename)
    hf_tokenizer = convert_to_hf_tokenizer(tokenizer_path)
    
    # Save HF tokenizer
    hf_tokenizer_dir = os.path.join(params.output_dir, "hf_tokenizer")
    os.makedirs(hf_tokenizer_dir, exist_ok=True)
    hf_tokenizer.save_pretrained(hf_tokenizer_dir)
    
    return hf_tokenizer_dir 