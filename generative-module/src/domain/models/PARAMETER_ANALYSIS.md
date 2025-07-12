# Common Parameter Analysis for PA-AI-2 Pipeline

## Overview
This document analyzes common parameters across the pipeline services (dataloader → encoder → feature extraction → generator → multimodal mapping) to identify inconsistencies and establish a unified parameter hierarchy.

## Common Parameters Identified

### 1. Dataset Configuration
| Parameter | Dataloader | Encoder | Feature Extraction | Generator | Multimodal | Inconsistency |
|-----------|------------|---------|-------------------|-----------|------------|---------------|
| `dataset_name` | `"DATASET_NAME"` | `"DATASET_NAME"` | `"DATASET_NAME"` | `"DATASET_NAME"` | `"DATASET_NAME"` | ✅ Consistent |

### 2. Model Architecture
| Parameter | Dataloader | Encoder | Feature Extraction | Generator | Multimodal | Inconsistency |
|-----------|------------|---------|-------------------|-----------|------------|---------------|
| `context_size` | `DEFAULT_CONTEXT_SIZE` | `DEFAULT_CONTEXT_SIZE` | `DEFAULT_CONTEXT_SIZE` / `-1` | `DEFAULT_CONTEXT_SIZE` | `DEFAULT_CONTEXT_SIZE` | ⚠️ LatentRepresentation uses -1 |
| `max_positions` | `DEFAULT_MAX_POSITIONS` | `DEFAULT_MAX_POSITIONS` | `DEFAULT_MAX_POSITIONS` | `DEFAULT_MAX_POSITIONS` | `DEFAULT_MAX_POSITIONS` | ✅ Consistent |
| `max_bars` | `MAX_N_BARS` | `MAX_N_BARS` | `MAX_N_BARS` | `16` | `MAX_N_BARS` | ❌ Generator uses 16 |
| `d_model` | N/A | N/A | `DEFAULT_D_MODEL` | `DEFAULT_D_MODEL` | `DEFAULT_D_MODEL` | ✅ Consistent |

### 3. Training Configuration
| Parameter | Dataloader | Encoder | Feature Extraction | Generator | Multimodal | Inconsistency |
|-----------|------------|---------|-------------------|-----------|------------|---------------|
| `batch_size` | `DEFAULT_BATCH_SIZE` | `DEFAULT_BATCH_SIZE` | `DEFAULT_BATCH_SIZE` | `DEFAULT_BATCH_SIZE` | `DEFAULT_BATCH_SIZE` | ✅ Consistent |
| `num_workers` | `2` | `4` | `4` | `4` | `4` | ❌ Dataloader uses 2 |
| `pin_memory` | `True` | N/A | `True` | `True` | `True` | ✅ Consistent |
| `device` | N/A | `"cuda"` | `"cuda"` | `"cuda"` | `"cuda"` | ✅ Consistent |

### 4. Data Processing
| Parameter | Dataloader | Encoder | Feature Extraction | Generator | Multimodal | Inconsistency |
|-----------|------------|---------|-------------------|-----------|------------|---------------|
| `max_bars_per_context` | `-1` | `-1` | `-1` | `-1` | `-1` | ✅ Consistent |
| `max_contexts_per_file` | `-1` | `-1` | N/A | `-1` | `-1` | ✅ Consistent |
| `bar_token_mask` | `None` | `None` | `None` | `None` | `None` | ✅ Consistent |
| `bar_token_idx` | `BOS_TOKEN_ID` | `BOS_TOKEN_ID` | `BOS_TOKEN_ID` | `BOS_TOKEN_ID` | `BOS_TOKEN_ID` | ✅ Consistent |
| `train_val_test_split` | `(0.7, 0.2, 0.1)` | N/A | `(0.7, 0.2, 0.1)` | `(0.7, 0.2, 0.1)` | `(0.7, 0.2, 0.1)` | ✅ Consistent |

### 5. Data Loading Flags
| Parameter | Dataloader | Encoder | Feature Extraction | Generator | Multimodal | Inconsistency |
|-----------|------------|---------|-------------------|-----------|------------|---------------|
| `load_latent` | `True` | N/A | `False` | `True` | `True` | ❌ Feature extraction uses False |
| `load_symb` | `True` | N/A | `False` | `True` | `True` | ❌ Feature extraction uses False |
| `load_emotions` | `True` | N/A | `False` | `True` | `True` | ❌ Feature extraction uses False |
| `load_global_features` | `False` | N/A | `False` | `False` | `True` | ❌ Multimodal uses True |
| `load_text_prompts` | `False` | N/A | `False` | `False` | `True` | ❌ Multimodal uses True |
| `encode` | `False` | N/A | `False` | `False` | `False` | ✅ Consistent |
| `caption` | `False` | N/A | `False` | `False` | `False` | ✅ Consistent |

### 6. Training Parameters (for training models)
| Parameter | Feature Extraction | Generator | Multimodal | Inconsistency |
|-----------|-------------------|-----------|------------|---------------|
| `lr` | `DEFAULT_LEARNING_RATE` | `DEFAULT_LEARNING_RATE` | `DEFAULT_LEARNING_RATE` | ✅ Consistent |
| `lr_schedule` | `"sqrt_decay"` | `"sqrt_decay"` | `"cosine"` | ❌ Multimodal uses "cosine" |
| `warmup_steps` | `DEFAULT_WARMUP_STEPS` | `DEFAULT_WARMUP_STEPS` | `DEFAULT_WARMUP_STEPS` | ✅ Consistent |
| `max_steps` | `DEFAULT_MAX_STEPS` | `DEFAULT_MAX_STEPS` | `DEFAULT_MAX_STEPS` | ✅ Consistent |
| `max_epochs` | `100` | `100` | `100` | ✅ Consistent |
| `weight_decay` | `DEFAULT_WEIGHT_DECAY` | `DEFAULT_WEIGHT_DECAY` | N/A | ✅ Consistent |
| `dropout` | `DEFAULT_DROPOUT` | `DEFAULT_DROPOUT` | `DEFAULT_DROPOUT` | ✅ Consistent |

### 7. Checkpoint Configuration
| Parameter | Feature Extraction | Generator | Multimodal | Inconsistency |
|-----------|-------------------|-----------|------------|---------------|
| `checkpoint_dir` | `"output/checkpoints/vae"` | `"output/checkpoints/generator"` | `"output/checkpoints/multimodal_mapping"` | ❌ Different paths |
| `load_from_checkpoint` | `False` | `False` | `False` | ✅ Consistent |
| `checkpoint_path` | `None` | `None` | `None` | ✅ Consistent |
| `weights_path` | `None` | `None` | `None` | ✅ Consistent |
| `config_path` | `None` | `None` | `None` | ✅ Consistent |

### 8. Validation and Logging
| Parameter | Feature Extraction | Generator | Multimodal | Inconsistency |
|-----------|-------------------|-----------|------------|---------------|
| `val_check_interval` | `DEFAULT_VAL_CHECK_INTERVAL` | `DEFAULT_VAL_CHECK_INTERVAL` | `DEFAULT_VAL_CHECK_INTERVAL` | ✅ Consistent |
| `log_every_n_steps` | `DEFAULT_LOG_EVERY_N_STEPS` | `DEFAULT_LOG_EVERY_N_STEPS` | `DEFAULT_LOG_EVERY_N_STEPS` | ✅ Consistent |
| `save_top_k` | `DEFAULT_SAVE_TOP_K` | `DEFAULT_SAVE_TOP_K` | `DEFAULT_SAVE_TOP_K` | ✅ Consistent |
| `limit_val_batches` | `DEFAULT_VALIDATION_BATCHES` | `DEFAULT_VALIDATION_BATCHES` | `DEFAULT_VALIDATION_BATCHES` | ✅ Consistent |
| `num_sanity_val_steps` | `DEFAULT_SANITY_VAL_STEPS` | `DEFAULT_SANITY_VAL_STEPS` | `DEFAULT_SANITY_VAL_STEPS` | ✅ Consistent |
| `every_n_train_steps` | `DEFAULT_CHECKPOINT_EVERY_N_STEPS` | `1000` | `1000` | ❌ Different values |

## Key Inconsistencies Identified

### Critical Issues
1. **`max_bars`**: Generator uses `16` while others use `MAX_N_BARS` (512)
2. **`num_workers`**: Dataloader uses `2` while others use `4`
3. **Data loading flags**: Feature extraction models have opposite defaults for `load_*` flags
4. **`lr_schedule`**: Multimodal uses `"cosine"` while others use `"sqrt_decay"`
5. **`every_n_train_steps`**: Feature extraction uses `DEFAULT_CHECKPOINT_EVERY_N_STEPS` (500) while others use `1000`

### Architectural Issues
1. **Checkpoint directories**: Each service has its own hardcoded path
2. **Context size**: LatentRepresentation uses `-1` for full context by default

## Proposed Hierarchy

Following the pipeline flow: **dataloader → encoder → feature extraction → generator → multimodal**

### Base Classes Structure
```
BaseParameters
├── DatasetParameters (dataset_name, device, etc.)
├── ModelArchitectureParameters (d_model, context_size, max_positions, etc.)
├── TrainingParameters (batch_size, lr, epochs, etc.)
├── DataProcessingParameters (max_bars_per_context, bar_token_*, etc.)
├── DataLoadingParameters (load_*, encode, caption)
└── CheckpointParameters (checkpoint_dir, load_from_checkpoint, etc.)
```

### Parameter Ownership by Service
- **Dataloader**: Defines all basic data processing parameters
- **Encoder**: Inherits from dataloader, adds encoding-specific parameters
- **Feature Extraction**: Inherits from encoder, adds training parameters
- **Generator**: Inherits from feature extraction, overrides specific model parameters
- **Multimodal**: Inherits from generator, adds multimodal-specific parameters 