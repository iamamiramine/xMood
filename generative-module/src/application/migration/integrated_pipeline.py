#!/usr/bin/env python3
"""
Integrated Pipeline Migration Tool

This script provides an easy way to migrate your music generation models
to text-generation-webui using the integrated pipeline approach.
"""

import os
import sys
import argparse
from pathlib import Path

from application.migration.services.migration_service import migrate_to_webui
from application.migration.models.migration_model import (
    TokenizerParameters,
    DatasetParameters,
    ModelExportParameters,
    ProjectionModelParameters,
    MigrationConfig
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Migrate music generation models to text-generation-webui using the integrated pipeline approach"
    )
    
    # Basic arguments
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to store intermediate files")
    parser.add_argument("--webui_dir", type=str, required=True,
                        help="Directory where text-generation-webui is installed")
    parser.add_argument("--model_name", type=str, default="music_generation_model",
                        help="Name for the exported model")
    
    # Dataset arguments
    parser.add_argument("--midi_dir", type=str, default="",
                        help="Directory containing MIDI files")
    parser.add_argument("--processed_dir", type=str, default="",
                        help="Directory containing processed MIDI data")
    parser.add_argument("--skip_dataset", action="store_true",
                        help="Skip dataset preparation")
    
    # Model arguments
    parser.add_argument("--projection_checkpoint", type=str, default=None,
                        help="Path to projection model checkpoint")
    parser.add_argument("--generator_checkpoint", type=str, default=None,
                        help="Path to generator model checkpoint")
    parser.add_argument("--model_type", type=str, default="custom",
                        choices=["t5", "bart", "custom"],
                        help="Type of model to create")
    
    # Tokenizer arguments
    parser.add_argument("--vocab_size", type=int, default=50000,
                        help="Vocabulary size for tokenizer")
    parser.add_argument("--skip_tokenizer", action="store_true",
                        help="Skip tokenizer creation")
    parser.add_argument("--existing_tokenizer_path", type=str, default=None,
                        help="Path to existing tokenizer")
    
    # Projection model arguments
    parser.add_argument("--proj_d_model", type=int, default=512,
                        help="Hidden size for projection model")
    parser.add_argument("--proj_encoder_layers", type=int, default=6,
                        help="Number of encoder layers for projection model")
    parser.add_argument("--proj_decoder_layers", type=int, default=6,
                        help="Number of decoder layers for projection model")
    
    # Generator model arguments
    parser.add_argument("--gen_d_model", type=int, default=512,
                        help="Hidden size for generator model")
    parser.add_argument("--gen_encoder_layers", type=int, default=6,
                        help="Number of encoder layers for generator model")
    parser.add_argument("--gen_decoder_layers", type=int, default=12,
                        help="Number of decoder layers for generator model")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set up basic paths
    output_dir = Path(args.output_dir)
    
    # Prepare configuration defaults (can be customized further)
    config = {
        "output_dir": str(output_dir),
        "webui_dir": args.webui_dir,
        "model_name": args.model_name,
        "projection_checkpoint": args.projection_checkpoint,
        "generator_checkpoint": args.generator_checkpoint,
        "model_type": args.model_type,
        "midi_dir": args.midi_dir,
        "processed_dir": args.processed_dir,
        "integrated_pipeline": True
    }
    
    # Run integrated pipeline migration
    print("Starting integrated pipeline migration...")
    
    try:
        result = migrate_to_webui(**config)
        print("\nMigration completed successfully!")
        print(f"Model exported to: {result['model_dir']}")
        print(f"Extension created at: {args.webui_dir}/extensions/music_generator")
        
        # Print some useful information
        if result.get("integrated_pipeline"):
            print("\nThis model uses the integrated pipeline approach:")
            print("- User prompts are projected to symbolic and latent features")
            print("- These features are used to condition music generation")
            print("- All UI controls are available in the Music Generation Parameters section")
    
    except Exception as e:
        print(f"Error during migration: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 