# pa-ai

# Parameter Setup

## Overview

The PA-AI-2 pipeline now includes a unified parameter setup system that allows you to configure all services with a single API call. This system is based on the analysis from the [BaseModel Parameters Analysis Report](BaseModel_Parameters_Analysis_Report.md) and provides a streamlined way to set common parameters once and then specify service-specific parameters.

## Key Features

✅ **Common Parameter Management**: Set parameters once that apply to all services  
✅ **Service-Specific Configuration**: Override or add parameters for specific services  
✅ **Automatic Parameter Merging**: Intelligent merging of common and specific parameters  
✅ **YAML Export**: Generate configuration files with unique IDs  
✅ **Parameter Validation**: Built-in validation and warning system  
✅ **Default Value Injection**: Automatic addition of required defaults  

## API Endpoint

### POST `/pipeline/setup_parameters`

Set up parameters for pipeline configuration and export to YAML file.

**Request Body**: `ParameterSetupRequest`

```json
{
  "common_parameters": {
    "device": "cuda",
    "batch_size": 64,
    "context_size": 1024,
    "dataset_name": "EMOPIA",
    "num_workers": 8,
    "pin_memory": true,
    "max_bars": 256,
    "max_positions": 2048,
    "bar_token_idx": 0,
    "train_val_test_split": [0.8, 0.1, 0.1],
    "load_latent": true,
    "load_symb": true,
    "load_emotions": true,
    "load_global_features": false,
    "load_text_prompts": false,
    "encode": true,
    "caption": false,
    "d_model": 768,
    "dropout": 0.1,
    "lr": 5e-5,
    "max_steps": 50000,
    "warmup_steps": 2000
  },
  "service_parameters": {
    "symbolic": {
      "encode_dataset": {
        "processed_dir": "datasets/processed/symbolic",
        "add_position_tokens": true,
        "omit_time_sig": false
      }
    },
    "latent": {
      "train_vae": {
        "n_codes": 2048,
        "n_groups": 8,
        "encoder_layers": 8,
        "decoder_layers": 8,
        "beta": 0.25
      }
    }
  },
  "config_name": "my_pipeline_config",
  "description": "Custom pipeline configuration for experiment XYZ",
  "global_config": {
    "seed": 12345,
    "output_root": "output/experiment/"
  }
}
```

**Response**: `ParameterSetupResponse`

```json
{
  "status": "success",
  "message": "Parameters setup completed successfully",
  "data": {
    "config_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "config_name": "my_pipeline_config",
    "config_file_path": "shared/config/my_pipeline_config_a1b2c3d4.yaml",
    "created_at": "2024-01-15T10:30:00Z",
    "validation_passed": true,
    "validation_warnings": [],
    "summary": {
      "common_parameters_count": 25,
      "service_parameters_configured": ["symbolic", "latent"],
      "total_functions_configured": 2
    }
  }
}
```

## Common Parameters

Based on the [BaseModel Parameters Analysis Report](BaseModel_Parameters_Analysis_Report.md), these parameters are used by 5+ models:

### **Highly Common Parameters (Used by 8+ models)**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `device` | `str` | `"cuda"` | Device to use for training/inference |
| `batch_size` | `int` | `32` | Batch size for processing |
| `context_size` | `int` | `512` | Context size for sequences |
| `dataset_name` | `str` | `"EMOPIA"` | Name of the dataset |
| `num_workers` | `int` | `4` | Number of workers for data loading |
| `pin_memory` | `bool` | `true` | Use pinned memory for data loading |
| `max_bars` | `int` | `512` | Maximum number of bars |
| `max_positions` | `int` | `1024` | Maximum position embeddings |
| `bar_token_idx` | `int` | `0` | Bar token index |
| `train_val_test_split` | `tuple` | `[0.7, 0.2, 0.1]` | Train/validation/test split |

### **Data Loading Configuration Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `load_latent` | `bool` | `false` | Load latent representations |
| `load_symb` | `bool` | `true` | Load symbolic representations |
| `load_emotions` | `bool` | `false` | Load emotion labels |
| `load_global_features` | `bool` | `false` | Load global features |
| `load_text_prompts` | `bool` | `false` | Load text prompts |
| `encode` | `bool` | `false` | Encode data |
| `caption` | `bool` | `false` | Generate captions |

### **Training Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `d_model` | `int` | `512` | Model dimension |
| `dropout` | `float` | `0.1` | Dropout rate |
| `lr` | `float` | `1e-4` | Learning rate |
| `max_steps` | `int` | `100000` | Maximum training steps |
| `warmup_steps` | `int` | `4000` | Warmup steps |

## Service-Specific Parameters

### **Symbolic Service**

#### `encode_dataset`
- `processed_dir`: Directory for processed files
- `add_position_tokens`: Add position tokens to features
- `omit_time_sig`: Omit time signature features
- `omit_instruments`: Omit instrument features
- `omit_chords`: Omit chord features
- `omit_meta`: Omit meta features
- `encodings_out_dir`: Output directory for encoded files
- `overwrite_existing`: Overwrite existing files

#### `tokenize_remi_dataset`
- `tokens_out_dir`: Output directory for tokens
- `vocab_path`: Path to vocabulary file
- `create_vocab`: Create vocabulary if not exists
- `max_bars_per_context`: Maximum bars per context
- `max_contexts_per_file`: Maximum contexts per file

### **Latent Service**

#### `train_vae`
- `n_codes`: Number of VQ codes
- `n_groups`: Number of VQ groups
- `encoder_layers`: Number of encoder layers
- `decoder_layers`: Number of decoder layers
- `encoder_ffn_dim`: Encoder FFN dimension
- `decoder_ffn_dim`: Decoder FFN dimension
- `beta`: Beta parameter for VQ-VAE
- `disable_vq`: Disable vector quantization
- `use_wandb`: Use Weights & Biases logging

#### `generate_latent_representations`
- `output_dir`: Output directory
- `save_latents`: Save latent representations
- `save_codes`: Save VQ codes
- `max_files`: Maximum files to process

### **Generator Service**

#### `train`
- `num_attention_heads`: Number of attention heads
- `d_latent`: Latent dimension
- `intermediate_size`: Intermediate layer size
- `vocab_size`: Vocabulary size
- `lr_schedule`: Learning rate schedule
- `max_epochs`: Maximum training epochs
- `gpus`: Number of GPUs
- `accumulate_grad_batches`: Gradient accumulation batches

#### `generate_from_midi`
- `latent_midi`: Path to latent MIDI file
- `symbolic_midi`: Path to symbolic MIDI file
- `emotions_midi`: Path to emotions MIDI file
- `output_folder`: Output folder for generated files
- `output_name`: Name of output file
- `max_n_tokens`: Maximum number of tokens
- `temperature`: Temperature for sampling

### **Multimodal Service**

#### `train`
- `image_dim`: Image feature dimension
- `text_dim`: Text feature dimension
- `fusion_dim`: Fusion dimension
- `fusion_type`: Fusion type
- `num_layers`: Number of layers
- `num_heads`: Number of attention heads
- `use_kl_annealing`: Use KL annealing
- `load_images`: Load images flag

#### `generate_representations`
- `output_folder`: Output folder
- `weights_path`: Path to model weights
- `config_path`: Path to model config
- `num_bars`: Number of bars to generate

## Usage Examples

### 1. Basic Setup with Common Parameters Only

```python
import requests

request_data = {
    "common_parameters": {
        "device": "cuda",
        "batch_size": 64,
        "dataset_name": "EMOPIA",
        "num_workers": 8,
        "lr": 1e-4
    },
    "service_parameters": {},
    "config_name": "basic_config"
}

response = requests.post(
    "http://localhost:8000/pipeline/setup_parameters",
    json=request_data
)
```

### 2. Full Pipeline Setup

```python
request_data = {
    "common_parameters": {
        "device": "cuda",
        "batch_size": 64,
        "context_size": 1024,
        "dataset_name": "EMOPIA",
        "num_workers": 8,
        "pin_memory": True,
        "max_bars": 256,
        "load_latent": True,
        "load_symb": True,
        "load_emotions": True,
        "encode": True,
        "d_model": 768,
        "dropout": 0.1,
        "lr": 5e-5,
        "max_steps": 50000,
        "warmup_steps": 2000
    },
    "service_parameters": {
        "symbolic": {
            "encode_dataset": {
                "add_position_tokens": True,
                "encodings_out_dir": "output/encoded/symbolic"
            }
        },
        "latent": {
            "train_vae": {
                "n_codes": 2048,
                "n_groups": 8,
                "encoder_layers": 8,
                "decoder_layers": 8,
                "beta": 0.25,
                "use_wandb": True
            }
        },
        "generator": {
            "train": {
                "num_attention_heads": 16,
                "d_latent": 256,
                "vocab_size": 400,
                "lr_schedule": "cosine",
                "max_epochs": 150,
                "gpus": 2
            }
        },
        "multimodal": {
            "train": {
                "image_dim": 768,
                "text_dim": 1024,
                "fusion_type": "cross_attention",
                "num_layers": 6,
                "use_kl_annealing": True
            }
        }
    },
    "config_name": "full_pipeline_config",
    "description": "Complete pipeline configuration for EMOPIA dataset"
}
```

### 3. Using the Example Script

```bash
# Run the example script
python example_parameter_setup.py
```

This will create a comprehensive configuration file with all services configured.

## Parameter Merging Logic

The system uses the following merging logic:

1. **Start with common parameters**: All services inherit common parameters
2. **Apply service-specific parameters**: Override common parameters with service-specific ones
3. **Add function-specific defaults**: Inject required defaults for specific functions
4. **Validate configuration**: Ensure all parameters are valid for the target BaseModel

### Example Merging Process

```python
# Common parameters
common = {
    "device": "cuda",
    "batch_size": 64,
    "lr": 1e-4
}

# Service-specific parameters
specific = {
    "batch_size": 128,  # Override common
    "n_codes": 1024     # Add specific
}

# Result after merging
merged = {
    "device": "cuda",     # From common
    "batch_size": 128,    # Overridden by specific
    "lr": 1e-4,          # From common
    "n_codes": 1024,     # From specific
    "max_epochs": 100    # Added as function default
}
```

## Configuration File Output

The generated YAML configuration file will be saved to `shared/config/` with the format:
```
{config_name}_{config_id[:8]}.yaml
```

Example output structure:
```yaml
metadata:
  version: "1.0"
  schema: "pa-ai-unified-config-v1"
  description: "Configuration created at 2024-01-15T10:30:00Z"
  generated_by: "PipelineParameterSetupService"

global:
  device: "cuda"
  seed: 42
  output_root: "output/"

paths:
  ROOT_OUTPUT: "output"
  DATASETS_PATH: "datasets"
  DATASET_NAME: "EMOPIA"

midi:
  pos_per_quarter: 12
  resolution: 480
  max_bars: 256

symbolic_service:
  encode_dataset:
    device: "cuda"
    batch_size: 64
    dataset_name: "EMOPIA"
    add_position_tokens: true
    encodings_out_dir: "output/encoded/symbolic"

latent_service:
  train_vae:
    device: "cuda"
    batch_size: 64
    n_codes: 2048
    n_groups: 8
    beta: 0.25
```

## Best Practices

1. **Set common parameters first**: Configure all shared parameters in `common_parameters`
2. **Use service-specific parameters for overrides**: Only specify parameters that differ from common ones
3. **Provide meaningful names**: Use descriptive `config_name` and `description` fields
4. **Validate your configuration**: Check the response for validation warnings
5. **Store configuration files**: Keep generated YAML files for reproducibility
6. **Use environment-specific settings**: Adjust `global_config`, `paths_config`, and `midi_config` for different environments

## Error Handling

The system provides comprehensive error handling:

- **Validation errors**: Invalid parameter values or types
- **Configuration errors**: Missing required parameters
- **File system errors**: Issues creating or writing configuration files
- **Service errors**: Problems with service-specific parameter merging

Always check the response status and handle errors appropriately in your application.

## Integration with Existing Pipeline

Once you have a configuration file, you can use it with the existing pipeline endpoints:

```python
# Use the generated configuration file
pipeline_request = {
    "pipeline_name": "my_experiment",
    "config_path": "shared/config/my_pipeline_config_a1b2c3d4.yaml",
    "services": ["symbolic", "latent", "generator"],
    "parallel_execution": False
}

response = requests.post(
    "http://localhost:8000/pipeline/execute_pipeline",
    json=pipeline_request
)
```