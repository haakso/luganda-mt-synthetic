# luganda-mt-synthetic

Research pipeline for evaluating LLMs on Luganda→English machine translation.
Candidate models (7–8B base, 4-bit quantized) are scored with COMET (wmt22-comet-da) and results logged to MLFlow.

## Prerequisites

- Docker with [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- A HuggingFace account with the [Llama 3.1 license accepted](https://huggingface.co/meta-llama/Llama-3.1-8B)
- `huggingface_hub` installed on the host machine (`pip install huggingface_hub`) for the download script

## Candidate models

| `--model` key | HuggingFace repo | Notes |
|---|---|---|
| `llama3` | `meta-llama/Llama-3.1-8B` | Gated — requires accepted license |
| `mistral` | `mistralai/Mistral-7B-v0.3` | |
| `aya` | `CohereLabs/aya-23-8B` | |
| `gemma` | `google/gemma-7b` | |

## Setup

1. Clone the repo
2. Copy `.env.example` to `.env` and fill in your values (`HF_TOKEN`, `MODEL_CACHE_DIR`, `DATA_DIR`, `OUTPUT_DIR`, `MLFLOW_TRACKING_URI`)
3. Download model weights into `MODEL_CACHE_DIR` (skips models already present):
   ```bash
   python scripts/download_models.py
   ```
4. Build the image (downloads COMET at build time — requires internet):
   ```bash
   docker compose build
   ```

## Test data format

The test set at `DATA_DIR/test_initial/pairs.jsonl` must be newline-delimited JSON with the following fields:

```json
{
  "text_id": "har_s0005",
  "luganda": "Luganda sentence here",
  "english": "English translation here",
  "dataset_origin": "makerere2024",
  "is_synthetic": false,
  "derived_from": null,
  "seed_group": null
}
```

## Running Evaluation

```bash
docker compose run evaluate --model mistral --output /results/mistral_results.jsonl
```

Optional flags:
- `--batch_size` (default: 8)
- `--output` path to write per-sentence JSONL to the `/results` volume

## MLFlow metrics

Each run logs:

| Metric | Description |
|---|---|
| `comet_mean` | COMET score across the full test set |
| `comet_mean_<dataset_origin>` | Per-origin breakdown, e.g. `comet_mean_makerere2024` |
| `num_examples_<dataset_origin>` | Example count per origin (logged as a param) |

If `--output` is provided, the results file includes all original fields plus `hypothesis` and per-sentence `comet` scores.
