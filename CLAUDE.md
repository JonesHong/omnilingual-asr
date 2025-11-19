# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Omnilingual ASR is an open-source multilingual speech recognition system supporting 1,600+ languages. Built on fairseq2, it provides three model families (W2V, CTC, LLM) with sizes ranging from 300M to 7B parameters.

## Development Setup

### Installation

```bash
# Install development dependencies
pip install -e ".[dev,data]"

# Install system dependency (required for audio processing)
# macOS
brew install libsndfile

# Linux
sudo apt-get install libsndfile1
```

### Verification

```bash
# Verify installation
python -c "from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline; print('Success!')"

# Run tests
pytest
```

## Common Commands

### Code Quality

```bash
# Format code (must pass before committing)
isort . && black .

# Lint checks
mypy  # Type checking
flake8 .  # Style checking

# Run all linters in sequence (as CI does)
isort --check . && black --check . && flake8 . && mypy
```

### Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_audio_utils.py

# Run tests with markers (slow tests are marked with @pytest.mark.slow)
pytest -m "not slow"  # Skip slow tests
pytest -m slow  # Run only slow tests

# Run with verbose output
pytest -v
```

### Training & Evaluation

```bash
# Set output directory for artifacts
export OUTPUT_DIR="/path/to/artifact/directory"

# Training examples (see workflows/recipes/wav2vec2/asr/README.md)
python -m workflows.recipes.wav2vec2.asr $OUTPUT_DIR --config-file workflows/recipes/wav2vec2/asr/configs/ctc-finetune.yaml

# Evaluation
python -m workflows.recipes.wav2vec2.asr.eval $OUTPUT_DIR --config-file workflows/recipes/wav2vec2/asr/eval/configs/fleurs-mls-mini.yaml
```

### Data Preparation

```bash
# Quick test with 2 languages from FLEURS (~5-10 minutes)
python workflows/dataprep/hf_dataset_ingestion_example.py run_short /path/to/output/dir

# Full example with MLS + FLEURS subset (~90 minutes)
python workflows/dataprep/hf_dataset_ingestion_example.py run_full /path/to/output/dir

# Verify dataloader
python -m workflows.dataprep.dataloader_example \
  --dataset_path="root_ds/all_asr/version=0" \
  --split="train" \
  --num_iterations=10
```

## Architecture Overview

### Model Hierarchy

The project implements a hierarchical model architecture built on Wav2Vec2:

1. **W2V (SSL Models)**: Audio → CNN Feature Extractor (320x downsample) → Transformer → Audio Embeddings
2. **CTC Models**: W2V Encoder → Linear Projection → Vocabulary Logits (parallel prediction)
3. **LLM Models**: W2V Encoder → Linear Projection → Llama Decoder → Vocabulary Logits (autoregressive)

**Key Characteristics**:
- All models operate on 16kHz mono audio
- CTC: Fast, non-autoregressive, ideal for on-device transcription
- LLM: Best quality, supports language conditioning and zero-shot with context examples
- LLM+ZS variant requires exactly 10 context examples (audio-text pairs)

### Dataset Architecture (Plugin Pattern)

The dataset system uses a composable plugin architecture separating storage and task concerns:

**Storage Backends** (`src/omnilingual_asr/datasets/storage/`):
- `MixtureParquetStorage`: Production backend for parquet files with weighted sampling
- `ManifestStorage`: Alternative manifest-based storage (reference implementation)

**Task Backends** (`src/omnilingual_asr/datasets/tasks/`):
- `AsrTask`: Returns `Seq2SeqBatch` (audio + text) for ASR training
- `SslTask`: Returns `SequenceBatch` (audio only) for self-supervised learning

**Interfaces** (`src/omnilingual_asr/datasets/interfaces/`):
- `StorageInterface`: Defines data loading contract
- `TaskInterface`: Defines task-specific processing contract

This separation allows mixing and matching storage formats with different training objectives.

### Asset System (fairseq2 Integration)

Models, tokenizers, and datasets are managed via YAML asset cards in `src/omnilingual_asr/cards/`:

```yaml
# Example model card
name: omniASR_CTC_300M
model_family: wav2vec2_asr
model_arch: 300m
checkpoint: https://dl.fbaipublicfiles.com/mms/omniASR-CTC-300M.pt
tokenizer_ref: omniASR_tokenizer
```

**Usage Pattern**: Create asset card → Reference by name in code/configs → fairseq2 handles loading/caching

**Model Storage**: Auto-downloaded to `~/.cache/fairseq2/assets/` on first use

### Parquet Dataset Structure

Datasets are partitioned by `corpus/split/language` for efficient filtering:

```
dataset_root/version=0/
├── corpus=mls/split=train/language=deu_Latn/part-*.parquet
└── corpus=fleurs/split=dev/language=fra_Latn/part-*.parquet
```

**Schema**:
- `text`: Normalized transcription
- `audio_bytes`: Compressed audio as list<int8> (16kHz mono, flac/ogg)
- `audio_size`: Decoded waveform size (used for duration filtering and batching)
- `corpus`, `split`, `language`: Partition keys for filtering

**Rationale**: Partition filtering enables weighted mixture sampling and language-specific evaluation

## Key Constraints & Design Decisions

### Model Input Constraints

- **Audio Length**: Currently limited to 40 seconds for inference (unlimited support planned)
- **LLM+ZS Context**: Requires exactly 10 context examples; repeat examples if fewer available
- **Sample Rate**: All audio must be 16kHz mono (automatic resampling in pipeline)

### Language Codes

Languages use ISO 639-3 + script format: `{language_code}_{script}` (e.g., `eng_Latn`, `cmn_Hans`)

**Reference**: Full list in `src/omnilingual_asr/models/wav2vec2_llama/lang_ids.py`

```python
from omnilingual_asr.models.wav2vec2_llama.lang_ids import supported_langs
print(len(supported_langs))  # 1600+
```

### Data Processing

- **Row Group Size**: Parquet files written with `row_group_size=100` to balance memory footprint and shuffling efficiency
- **Audio Format**: Use `pa.list_(pa.int8())` instead of `pa.binary()` to avoid copying during pyarrow→pandas conversion
- **Text Normalization**: Language-specific processing in `workflows/dataprep/text_tools.py`

### fairseq2 Recipe System

Recipes are pre-configured training/evaluation workflows combining models, datasets, and hyperparameters:

- Training recipes: `workflows/recipes/wav2vec2/asr/`
- Evaluation recipes: `workflows/recipes/wav2vec2/asr/eval/`
- Configuration: YAML files in respective `configs/` directories
- Execution: Python module invocation with config file and output directory

## Code Style

- Python 3.10+ required
- PEP 8 compliance with Black formatter (88 character line limit)
- Type hints required for function signatures
- Strict mypy checking enabled (see `[tool.mypy]` in pyproject.toml)
- 4 spaces for indentation

## CI/CD

GitHub Actions workflow (`.github/workflows/lint_and_test.yaml`) runs on PRs:
1. isort check
2. black check
3. flake8 lint
4. mypy type check
5. pytest test suite

All checks must pass before merging.

## Documentation References

- Inference guide: `src/omnilingual_asr/models/inference/README.md`
- Model architectures: `src/omnilingual_asr/models/README.md`
- Data preparation: `workflows/dataprep/README.md`
- Training recipes: `workflows/recipes/wav2vec2/asr/README.md`
- Asset system: `src/omnilingual_asr/cards/README.md`
