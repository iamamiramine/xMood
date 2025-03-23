import os
import json
import shutil
from pathlib import Path
from typing import Dict, Optional, Union, List

import torch
import torch.nn as nn
from transformers import (
    T5Config,
    T5ForConditionalGeneration,
    BartConfig,
    BartForConditionalGeneration,
    PreTrainedTokenizerFast,
    EncoderDecoderConfig,
    EncoderDecoderModel
)

from application.generator.models.generator_model import MIDIGeneratorModule
from application.projection.models.projection_model import ProjectorModule
from application.migration.models.migration_model import ModelExportParameters


class MusicGenerationPipeline(nn.Module):
    """Combined model that handles both projection and generation in a single pipeline"""
    
    def __init__(
        self, 
        projector_model: nn.Module, 
        generator_model: nn.Module,
        tokenizer
    ):
        super().__init__()
        self.projector = projector_model
        self.generator = generator_model
        self.tokenizer = tokenizer
        
    def forward(self, input_text, mood=None, features=None, generation_params=None):
        """Complete pipeline from text input to music generation"""
        # Default generation parameters
        if generation_params is None:
            generation_params = {
                "max_length": 256,
                "max_bars": 16,
                "temp": 0.8
            }
            
        # 1. Project text to symbolic and latent features
        proj_input = {
            "input_text": input_text,
            "mood": mood,
            "features": features
        }
        
        # Get symbolic and latent features from projector
        with torch.no_grad():
            projection_output = self.projector(proj_input)
            symbolic_features = projection_output.get("symbolic_features")
            latent_features = projection_output.get("latent_features")
        
        # 2. Generate music using these features
        generation_input = {
            "bar_symbolic": symbolic_features,
            "latents": latent_features
        }
        
        if mood is not None:
            generation_input["moods"] = mood
            
        # Generate the music sequence
        with torch.no_grad():
            generation_output = self.generator.sample(
                batch=generation_input,
                max_length=generation_params["max_length"],
                max_bars=generation_params["max_bars"],
                temp=generation_params["temp"]
            )
            
        return {
            "sequences": generation_output["sequences"],
            "bar_ids": generation_output["bar_ids"],
            "position_ids": generation_output["position_ids"],
            "symbolic_features": symbolic_features,
            "latent_features": latent_features
        }
        
    def decode_output(self, output):
        """Convert generated token sequences to text tokens"""
        sequences = output["sequences"]
        return [self.tokenizer.decode(seq) for seq in sequences]


class MusicT5Model(T5ForConditionalGeneration):
    """Custom T5 model with capabilities for music generation"""
    
    def __init__(self, config):
        super().__init__(config)
        self.mood_embedding = None
        self.feature_embedding = None
        
    def add_mood_conditioning(self, num_moods: int):
        """Add an embedding layer for mood conditioning"""
        self.mood_embedding = nn.Embedding(num_moods, self.config.d_model)
        
    def add_feature_conditioning(self, num_features: int):
        """Add an embedding layer for feature conditioning"""
        self.feature_embedding = nn.Embedding(num_features, self.config.d_model)
        
    def add_projection_layer(self):
        """Add a projection layer for combined embeddings"""
        self.projection = nn.Linear(self.config.d_model * 3, self.config.d_model)


def create_t5_config(params: ModelExportParameters) -> T5Config:
    """Create a T5 configuration for music generation"""
    config = T5Config(
        vocab_size=params.config_overrides.get("vocab_size", 50000),
        d_model=params.d_model,
        d_kv=params.config_overrides.get("d_kv", 64),
        d_ff=params.d_ff,
        num_layers=params.num_encoder_layers,
        num_decoder_layers=params.num_decoder_layers,
        num_heads=params.num_heads,
        relative_attention_num_buckets=params.config_overrides.get("relative_attention_num_buckets", 32),
        dropout_rate=params.config_overrides.get("dropout_rate", 0.1),
        layer_norm_epsilon=params.config_overrides.get("layer_norm_epsilon", 1e-6),
        initializer_factor=params.config_overrides.get("initializer_factor", 1.0),
        feed_forward_proj=params.config_overrides.get("feed_forward_proj", "relu"),
        is_encoder_decoder=True,
        use_cache=True,
        pad_token_id=params.config_overrides.get("pad_token_id", 0),
        eos_token_id=params.config_overrides.get("eos_token_id", 1),
        decoder_start_token_id=params.config_overrides.get("decoder_start_token_id", 0),
    )
    
    return config


def create_bart_config(params: ModelExportParameters) -> BartConfig:
    """Create a BART configuration for music generation"""
    config = BartConfig(
        vocab_size=params.config_overrides.get("vocab_size", 50000),
        d_model=params.d_model,
        encoder_layers=params.num_encoder_layers,
        decoder_layers=params.num_decoder_layers,
        encoder_attention_heads=params.num_heads,
        decoder_attention_heads=params.num_heads,
        encoder_ffn_dim=params.d_ff,
        decoder_ffn_dim=params.d_ff,
        dropout=params.config_overrides.get("dropout", 0.1),
        attention_dropout=params.config_overrides.get("attention_dropout", 0.1),
        activation_dropout=params.config_overrides.get("activation_dropout", 0.1),
        activation_function=params.config_overrides.get("activation_function", "gelu"),
        pad_token_id=params.config_overrides.get("pad_token_id", 0),
        eos_token_id=params.config_overrides.get("eos_token_id", 1),
        bos_token_id=params.config_overrides.get("bos_token_id", 0),
        is_encoder_decoder=True,
    )
    
    return config


def load_generator_weights(
    model_path: Path, 
    target_model,
    mapping_dict: Optional[Dict] = None
) -> None:
    """
    Load weights from your MIDIGeneratorModule to the target HF model
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location="cpu")
    source_state_dict = checkpoint["state_dict"]
    
    # Create target state dict
    target_state_dict = target_model.state_dict()
    
    # If no mapping is provided, create a default mapping
    if mapping_dict is None:
        mapping_dict = {
            # Map encoder components
            "encoder.*.attention.self.query": "encoder.block.*.layer.0.SelfAttention.q",
            "encoder.*.attention.self.key": "encoder.block.*.layer.0.SelfAttention.k",
            "encoder.*.attention.self.value": "encoder.block.*.layer.0.SelfAttention.v",
            "encoder.*.attention.output.dense": "encoder.block.*.layer.0.SelfAttention.o",
            "encoder.*.intermediate.dense": "encoder.block.*.layer.1.DenseReluDense.wi",
            "encoder.*.output.dense": "encoder.block.*.layer.1.DenseReluDense.wo",
            # Map decoder components
            "decoder.*.attention.self.query": "decoder.block.*.layer.0.SelfAttention.q",
            "decoder.*.attention.self.key": "decoder.block.*.layer.0.SelfAttention.k",
            "decoder.*.attention.self.value": "decoder.block.*.layer.0.SelfAttention.v",
            "decoder.*.attention.output.dense": "decoder.block.*.layer.0.SelfAttention.o",
            "decoder.*.crossattention.query": "decoder.block.*.layer.1.EncDecAttention.q",
            "decoder.*.crossattention.key": "decoder.block.*.layer.1.EncDecAttention.k",
            "decoder.*.crossattention.value": "decoder.block.*.layer.1.EncDecAttention.v",
            "decoder.*.crossattention.output.dense": "decoder.block.*.layer.1.EncDecAttention.o",
            "decoder.*.intermediate.dense": "decoder.block.*.layer.2.DenseReluDense.wi",
            "decoder.*.output.dense": "decoder.block.*.layer.2.DenseReluDense.wo",
            # Map embeddings
            "in_layer.weight": "shared.weight",
            "out_layer.weight": "lm_head.weight",
        }
    
    # Apply weight mapping
    # This is a simplified implementation - a real one would handle patterns and complex mappings
    for source_key, source_param in source_state_dict.items():
        # Find matching target key
        for source_pattern, target_pattern in mapping_dict.items():
            if source_key.startswith(source_pattern.split("*")[0]):
                target_key = target_pattern.replace("*", source_key.split(".")[-2])
                if target_key in target_state_dict:
                    # Check if shapes match
                    if source_param.shape == target_state_dict[target_key].shape:
                        target_state_dict[target_key] = source_param
                        print(f"Mapped {source_key} -> {target_key}")
                    else:
                        print(f"Shape mismatch: {source_key} {source_param.shape} -> {target_key} {target_state_dict[target_key].shape}")
    
    # Load mapped weights to target model
    missing, unexpected = target_model.load_state_dict(target_state_dict, strict=False)
    print(f"Loaded weights with {len(missing)} missing and {len(unexpected)} unexpected")


def create_new_model(params: ModelExportParameters, tokenizer: PreTrainedTokenizerFast) -> torch.nn.Module:
    """Create a new model based on the specified type"""
    # Setup config overrides with tokenizer info
    params.config_overrides["vocab_size"] = len(tokenizer)
    params.config_overrides["pad_token_id"] = tokenizer.pad_token_id
    params.config_overrides["eos_token_id"] = tokenizer.eos_token_id
    params.config_overrides["bos_token_id"] = tokenizer.bos_token_id if hasattr(tokenizer, "bos_token_id") else tokenizer.pad_token_id
    
    if params.model_type.lower() == "t5":
        config = create_t5_config(params)
        model = MusicT5Model(config)
        
        # Add custom conditioning
        # These would be added based on your specific conditioning needs
        model.add_mood_conditioning(100)  # Example: 100 mood types
        model.add_feature_conditioning(1000)  # Example: 1000 feature types
        model.add_projection_layer()
        
    elif params.model_type.lower() == "bart":
        config = create_bart_config(params)
        model = BartForConditionalGeneration(config)
        
    elif params.model_type.lower() == "custom":
        # Create a custom encoder-decoder model
        from transformers import BertConfig
        
        encoder_config = BertConfig(
            vocab_size=len(tokenizer),
            hidden_size=params.d_model,
            num_hidden_layers=params.num_encoder_layers,
            num_attention_heads=params.num_heads,
            intermediate_size=params.d_ff,
            max_position_embeddings=1024,
            position_embedding_type="relative_key_query",
        )
        
        decoder_config = BertConfig(
            vocab_size=len(tokenizer),
            hidden_size=params.d_model,
            num_hidden_layers=params.num_decoder_layers,
            num_attention_heads=params.num_heads,
            intermediate_size=params.d_ff,
            max_position_embeddings=1024,
            position_embedding_type="relative_key_query",
            is_decoder=True,
            add_cross_attention=True,
        )
        
        config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
        model = EncoderDecoderModel(config)
        
    else:
        raise ValueError(f"Unsupported model type: {params.model_type}")
    
    return model


def load_existing_model(params: ModelExportParameters, tokenizer: PreTrainedTokenizerFast) -> torch.nn.Module:
    """Load an existing music generation model and map to HF model"""
    if not params.source_checkpoint:
        raise ValueError("Source checkpoint must be provided for loading existing model")
    
    # Create a new model
    model = create_new_model(params, tokenizer)
    
    # Load weights from source checkpoint
    load_generator_weights(params.source_checkpoint, model)
    
    return model


def create_extension_files(params: ModelExportParameters, extension_dir: Path) -> None:
    """Create extension files for text-generation-webui"""
    # Ensure extension directory exists
    os.makedirs(extension_dir, exist_ok=True)
    
    # Create script.py
    script_path = extension_dir / "script.py"
    with open(script_path, "w") as f:
        f.write("""
import os
import time
import json
import gradio as gr
import torch
import numpy as np
from pathlib import Path
from modules import shared
from modules.logging_colors import logger

# Try to import music-specific libraries
try:
    import pretty_midi
    MIDI_AVAILABLE = True
except ImportError:
    MIDI_AVAILABLE = False
    logger.warning("pretty_midi not found. MIDI export will be disabled.")

# Global parameters for music generation
music_params = {
    "mood": "",
    "features": {
        "tempo": 120,
        "key": "C:maj",
        "time_signature": "4/4",
        "instrumentation": "piano",
        "style": "classical"
    },
    "generation": {
        "max_length": 512,
        "max_bars": 16,
        "temperature": 0.8
    }
}

# Models
PROJECTOR = None
GENERATOR = None
PIPELINE = None

def load_pipeline_models(model_dir):
    """Load the projector and generator models from the model directory"""
    global PROJECTOR, GENERATOR, PIPELINE
    
    try:
        from application.projection.models.projection_model import ProjectorModule
        from application.generator.models.generator_model import MIDIGeneratorModule
        
        # Check if we have a pipeline configuration
        pipeline_config_path = Path(model_dir) / "pipeline_config.json"
        if pipeline_config_path.exists():
            logger.info("Loading music generation pipeline...")
            
            # Load the configuration
            with open(pipeline_config_path, "r") as f:
                config = json.load(f)
            
            # Load projector model
            projector_path = Path(model_dir) / "projector.pt"
            if projector_path.exists():
                logger.info(f"Loading projector from {projector_path}")
                # You would need to adjust this based on your actual loading logic
                # This is a placeholder that assumes you have loading functions
                PROJECTOR = ProjectorModule.load_from_checkpoint(projector_path)
            
            # Load generator model
            generator_path = Path(model_dir) / "generator.pt"
            if generator_path.exists():
                logger.info(f"Loading generator from {generator_path}")
                # Similarly, adjust this based on your actual loading logic
                GENERATOR = MIDIGeneratorModule.load_from_checkpoint(generator_path)
            
            # Create a simple pipeline object if both models are loaded
            if PROJECTOR is not None and GENERATOR is not None:
                PIPELINE = {
                    "projector": PROJECTOR,
                    "generator": GENERATOR
                }
                logger.info("Music generation pipeline loaded successfully")
                return True
    except ImportError as e:
        logger.error(f"Failed to import music generation modules: {e}")
    except Exception as e:
        logger.error(f"Error loading music generation pipeline: {e}")
    
    return False

def input_modifier(string):
    """Add conditioning to the prompt."""
    global music_params
    
    # Extract features as a formatted string
    features_text = json.dumps(music_params["features"])
    
    # Format with mood and features tags
    mood_condition = f"[MOOD={music_params['mood']}]" if music_params['mood'] else ""
    feature_condition = f"[FEATURES={features_text}]" if features_text else ""
    
    # Add conditioning to prompt
    conditioned_prompt = f"{mood_condition} {feature_condition} {string}".strip()
    
    # Log what we're doing
    logger.info(f"Conditioning prompt with mood: {music_params['mood']} and features")
    
    return conditioned_prompt

def output_modifier(string):
    """Process the generated output, potentially converting to MIDI."""
    global PIPELINE, music_params
    
    # Check if the output contains music tokens (rough check)
    if "Bar_" in string and "Position_" in string:
        try:
            # If we have our pipeline loaded, we can try to convert properly
            if PIPELINE is not None and MIDI_AVAILABLE:
                # Parse the tokens from the generated text
                tokens = string.strip().split()
                
                # Convert tokens to MIDI (implementation would depend on your tokenizer)
                # This is a placeholder for your actual conversion logic
                midi_data = convert_tokens_to_midi(tokens)
                
                # Save to file
                timestamp = int(time.time())
                output_path = f"outputs/generated_{timestamp}.mid"
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                midi_data.write(output_path)
                
                # Return original text + link to download
                return f"{string}\\n\\n[Download MIDI]({output_path})"
            else:
                return f"{string}\\n\\n(MIDI conversion not available - install pretty_midi or check pipeline)"
        except Exception as e:
            return f"{string}\\n\\nError converting to MIDI: {str(e)}"
    
    return string

def convert_tokens_to_midi(tokens):
    """Convert REMI tokens to a MIDI file"""
    # This is a placeholder for your actual conversion logic
    # You would implement your REMI → MIDI conversion here
    
    # Example implementation (replace with your actual logic)
    midi_data = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)
    
    # Add a few example notes
    note = pretty_midi.Note(
        velocity=100, 
        pitch=60, 
        start=0, 
        end=1
    )
    piano.notes.append(note)
    midi_data.instruments.append(piano)
    
    return midi_data

def ui():
    """Create the user interface for music generation parameters"""
    # Load models if they haven't been loaded
    current_model_dir = shared.model_path
    if current_model_dir and (PIPELINE is None):
        load_pipeline_models(current_model_dir)
    
    with gr.Accordion("Music Generation Parameters", open=False):
        with gr.Row():
            mood = gr.Dropdown(
                choices=["Happy", "Sad", "Energetic", "Calm", "Dramatic", "Mysterious", "Romantic"],
                label="Mood",
                value=""
            )
        
        with gr.Row():
            with gr.Column():
                tempo = gr.Slider(40, 240, value=120, step=1, label="Tempo (BPM)")
                key = gr.Dropdown(
                    choices=["C:maj", "C:min", "C#:maj", "C#:min", "D:maj", "D:min", 
                            "D#:maj", "D#:min", "E:maj", "E:min", "F:maj", "F:min",
                            "F#:maj", "F#:min", "G:maj", "G:min", "G#:maj", "G#:min",
                            "A:maj", "A:min", "A#:maj", "A#:min", "B:maj", "B:min"],
                    label="Key Signature",
                    value="C:maj"
                )
                time_signature = gr.Dropdown(
                    choices=["4/4", "3/4", "6/8", "5/4", "7/8"],
                    label="Time Signature",
                    value="4/4"
                )
            
            with gr.Column():
                instrumentation = gr.Dropdown(
                    choices=["piano", "guitar", "strings", "orchestra", "electronic", "drums", "bass"],
                    label="Main Instrument",
                    value="piano"
                )
                style = gr.Dropdown(
                    choices=["classical", "jazz", "rock", "pop", "electronic", "ambient", "film_score"],
                    label="Musical Style",
                    value="classical"
                )
                max_bars = gr.Slider(4, 32, value=16, step=1, label="Maximum Bars")
        
        with gr.Row():
            temperature = gr.Slider(0.1, 2.0, value=0.8, step=0.1, label="Temperature")
            
        with gr.Row():
            generation_button = gr.Button("Apply Music Parameters")
        
        def update_music_params(mood_val, tempo_val, key_val, time_sig_val, instrument_val, 
                               style_val, max_bars_val, temp_val):
            global music_params
            
            music_params.update({
                "mood": mood_val,
                "features": {
                    "tempo": tempo_val,
                    "key": key_val,
                    "time_signature": time_sig_val,
                    "instrumentation": instrument_val,
                    "style": style_val
                },
                "generation": {
                    "max_length": 512,
                    "max_bars": max_bars_val,
                    "temperature": temp_val
                }
            })
            
            return f"Music parameters updated: Mood={mood_val}, Tempo={tempo_val}, Key={key_val}"
            
        generation_button.click(
            update_music_params,
            inputs=[mood, tempo, key, time_signature, instrumentation, style, max_bars, temperature],
            outputs=[gr.Textbox(label="Status")]
        )
    
    return [mood, tempo, key, time_signature, instrumentation, style, max_bars, temperature]
""")
    
    # Create config.json
    config_path = extension_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump({
            "name": "Music Generator",
            "version": "1.0",
            "description": "Generate music using a transformer encoder-decoder model conditioned on moods and features",
            "author": "Your Name",
            "api": {
                "supports_dynamic_prompts": True,
                "supports_conditioning": True
            }
        }, f, indent=4)
    
    # Create requirements.txt
    requirements_path = extension_dir / "requirements.txt"
    with open(requirements_path, "w") as f:
        f.write("""
pretty_midi>=0.2.9
midi2audio>=0.1.1
numpy>=1.20.0
""")

    # Create a simple README.md
    readme_path = extension_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write("""# Music Generator Extension for Text Generation WebUI

This extension allows you to generate music using a specialized music generation model.

## Features

- Condition music generation on mood and musical parameters
- Generate MIDI files directly from text prompts
- Control musical parameters like tempo, key, and style

## Usage

1. Select a music generation model
2. Set your desired mood and musical parameters
3. Enter a text prompt describing the music you want
4. Generate!
5. Download the resulting MIDI file

## Requirements

- pretty_midi
- midi2audio (optional, for audio playback)
""")


def export_model(
    params: ModelExportParameters, 
    tokenizer: PreTrainedTokenizerFast,
    model: Optional[torch.nn.Module] = None
) -> Path:
    """
    Export model for text-generation-webui
    
    Args:
        params: Model export parameters
        tokenizer: Tokenizer to use with the model
        model: Pre-created model (optional)
        
    Returns:
        Path to exported model directory
    """
    os.makedirs(params.output_dir, exist_ok=True)
    
    # Create or load model
    if model is None:
        if params.source_checkpoint:
            print(f"Loading existing checkpoint from {params.source_checkpoint}")
            
            # Check if we're loading a pipeline or individual models
            if "pipeline_config.json" in os.listdir(Path(params.source_checkpoint).parent):
                # Load pipeline configuration
                with open(Path(params.source_checkpoint).parent / "pipeline_config.json", "r") as f:
                    pipeline_config = json.load(f)
                    
                # Load projector model
                projector_path = pipeline_config.get("projector_checkpoint")
                if projector_path:
                    projector = ProjectorModule.load_from_checkpoint(projector_path)
                else:
                    # Create a dummy projector if none provided
                    projector = create_dummy_projector()
                    
                # Load generator model
                generator_path = pipeline_config.get("generator_checkpoint", str(params.source_checkpoint))
                generator = MIDIGeneratorModule.load_from_checkpoint(generator_path)
                
                # Create pipeline model
                model = MusicGenerationPipeline(projector, generator, tokenizer)
            else:
                # Just load a regular model
                model = load_existing_model(params, tokenizer)
        else:
            print(f"Creating new model of type {params.model_type}")
            model = create_new_model(params, tokenizer)
    
    # Save the model
    print(f"Saving model to {params.output_dir}")
    
    # For the pipeline model, we need special handling
    if isinstance(model, MusicGenerationPipeline):
        # Save generator and projector separately
        torch.save(model.generator.state_dict(), os.path.join(params.output_dir, "generator.pt"))
        torch.save(model.projector.state_dict(), os.path.join(params.output_dir, "projector.pt"))
        
        # Save pipeline configuration
        pipeline_config = {
            "type": "music_generation_pipeline",
            "generator_model_type": type(model.generator).__name__,
            "projector_model_type": type(model.projector).__name__
        }
        
        with open(os.path.join(params.output_dir, "pipeline_config.json"), "w") as f:
            json.dump(pipeline_config, f, indent=4)
    else:
        # Standard model saving
        model.save_pretrained(params.output_dir)
    
    # Save tokenizer
    tokenizer.save_pretrained(params.output_dir)
    
    # If webui_dir is provided, copy model to webui models directory
    if params.webui_dir:
        webui_model_dir = os.path.join(params.webui_dir, "models", params.model_name)
        os.makedirs(webui_model_dir, exist_ok=True)
        
        # Copy model files
        for file in os.listdir(params.output_dir):
            src = os.path.join(params.output_dir, file)
            dst = os.path.join(webui_model_dir, file)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
            elif os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
        
        # Create extension
        if params.convert_format:
            extension_dir = os.path.join(params.webui_dir, "extensions", "music_generator")
            create_extension_files(params, Path(extension_dir))
    
    return Path(params.output_dir)


def create_dummy_projector():
    """Create a simple projector for testing"""
    return ProjectorModule(d_model=512) 