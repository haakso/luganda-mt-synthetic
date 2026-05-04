# luganda-mt-synthetic

Luganda is an extremely low-resource language with limited parallel corpora, making neural machine translation challenging for standard approaches. This project investigates whether synthetic data augmentation can meaningfully improve LG→EN translation quality when combined with parameter-efficient fine-tuning. We benchmark four 7–8B parameter LLMs (Gemma-7B, Mistral-7B, Llama-3.1-8B, Aya-23-8B) zero-shot on a 2,000-pair test set drawn from three corpora, scoring with COMET (wmt22-comet-da). Zero-shot performance is modest across all models (COMET 0.325–0.356), with no single model dominant. We then fine-tune the Aya-23-8B base model with QLoRA on 1× and 5× synthetic training sets, finding that even the 1× condition yields a large gain (COMET 0.544, +0.219) and the 5× condition brings the model to 0.611 (+0.286 over zero-shot), with consistent improvements across all three source corpora.

## Architecture

### Overall pipeline

![Overall pipeline diagram](figures/pipeline.png)

### QLoRA fine-tuning

![QLoRA fine-tuning diagram](figures/qlora.png)

---

## Prerequisites

- [uv](https://docs.astral.sh/uv/) for dependency management (host-side download script + local dev)
- Docker with [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for evaluation and fine-tuning runs
- A HuggingFace account with the [Llama 3.1 license accepted](https://huggingface.co/meta-llama/Llama-3.1-8B)

## Codebase map

| Path | Description |
|---|---|
| `scripts/evaluate.py` | Main evaluation harness: loads a candidate model in 4-bit, runs batched greedy inference on the test set, scores with COMET, and logs results to MLFlow |
| `scripts/finetune.py` | Sequential QLoRA fine-tuning of Aya 23 8B on the 1x and 5x synthetic conditions; each condition trains, evaluates with COMET, and logs to a nested MLFlow run under a single parent |
| `scripts/download_models.py` | Downloads all candidate model weights from HuggingFace into `MODEL_CACHE_DIR`; skips models whose target directory already exists |
| `Dockerfile` | CUDA 12.8 + Python 3.11 image managed by uv; bakes in the COMET checkpoint and xlm-roberta encoder at build time so the container is fully air-gapped at runtime |
| `docker-compose.yml` | Defines `evaluate` and `finetune` services with GPU passthrough and volume mounts for models, data, and results |
| `pyproject.toml` | Python project definition and dependencies (managed by uv); torch is sourced from the cu128 wheel index |
| `run_overnight.sh` | Convenience script that runs the full `finetune` pipeline unattended and tees output to `results/finetune_overnight.log` |
| `Dataset/` | Training, validation, evaluation, test, and synthetic data files (see dataset README for provenance) |
| `results/` | Inference outputs (`.jsonl`), LoRA adapter checkpoints, and run logs; gitignored except adapter configs |
| `ai-usage-logs/` | Claude Code session transcripts for this project |

## Candidate models

| `--model` key | HuggingFace repo | Notes |
|---|---|---|
| `llama3` | `meta-llama/Llama-3.1-8B` | Gated — requires accepted license |
| `mistral` | `mistralai/Mistral-7B-v0.3` | |
| `aya` | `CohereLabs/aya-23-8B` | Also used as the fine-tuning base model |
| `gemma` | `google/gemma-7b` | |

## Setup

1. Clone the repo
2. Copy `.env.example` to `.env` and fill in your values (`HF_TOKEN`, `MODEL_CACHE_DIR`, `DATA_DIR`, `OUTPUT_DIR`, `MLFLOW_TRACKING_URI`)
3. Download model weights into `MODEL_CACHE_DIR` (skips models already present):
   ```bash
   uv run scripts/download_models.py
   ```
4. Build the image (downloads COMET and xlm-roberta at build time — requires internet):
   ```bash
   docker compose build
   ```

## Test data format

The test set at `DATA_DIR/test.json` must be a JSON array of objects with the following fields:

```json
[
  {
    "text_id": "har_s0005",
    "luganda": "Luganda sentence here",
    "english": "English translation here",
    "dataset_origin": "makerere2024",
    "is_synthetic": false,
    "derived_from": null,
    "seed_group": null
  }
]
```

Training, validation, evaluation, and synthetic data files in `Dataset/` follow the same schema.

## Running Evaluation

```bash
docker compose run evaluate --model mistral --output /results/mistral_results.jsonl
```

Optional flags:
- `--batch_size` (default: 8)
- `--limit N` — run on only the first N examples (smoke test)
- `--output` path to write per-sentence JSONL to the `/results` volume

## Running Fine-tuning

The finetuning script trains Aya 23 8B with QLoRA on the 1x and 5x synthetic conditions sequentially, then evaluates each adapter with COMET. A failure in one condition does not abort the other.

**Overnight run** (uses paths from `docker-compose.yml`):

```bash
bash run_overnight.sh
```

**Manual run** (smoke test with `--limit` to verify the pipeline end-to-end):

```bash
docker compose run finetune \
  --model_path /models/aya-23-8B \
  --data_1x /data/synthetic_1x.json \
  --data_5x /data/synthetic_5x.json \
  --val_data /data/validation.json \
  --eval_data /data/evaluation.json \
  --output_dir /results/lora_smoke \
  --limit 50
```

Key optional flags:
- `--epochs` (default: 5)
- `--batch_size` (default: 4)
- `--grad_accum` (default: 8)
- `--lora_rank` (default: 16)
- `--limit N` — cap train/val/eval to first N examples (smoke test)

Adapter checkpoints are saved to `output_dir/lora_1x/` and `output_dir/lora_5x/`. Per-sentence evaluation results are written alongside them as `lora_1x_eval_results.jsonl` and `lora_5x_eval_results.jsonl`.

## MLFlow metrics

Each evaluation run logs:

| Metric | Description |
|---|---|
| `comet_mean` | COMET score across the full test set |
| `comet_mean_<dataset_origin>` | Per-origin breakdown, e.g. `comet_mean_makerere2024` |
| `num_examples_<dataset_origin>` | Example count per origin (logged as a param) |

Fine-tuning runs create a parent run (`finetune_lora`) with two nested child runs (`lora_1x`, `lora_5x`), each logging the same metrics plus training loss and hyperparameters.

If `--output` is provided for evaluation, or automatically for fine-tuning, the results file includes all original fields plus `hypothesis` and per-sentence `comet` scores.

## Results

### Zero-shot evaluation (2,000-pair test set)

COMET scores (wmt22-comet-da) broken down by source corpus. Bold marks the model selected for fine-tuning.

| Model | Overall | harvard | kimrichies | makerere |
|---|---|---|---|---|
| Gemma-7B | 0.354 | 0.350 | 0.355 | 0.357 |
| Mistral-7B | 0.356 | 0.349 | 0.357 | 0.366 |
| Llama-3.1-8B | 0.349 | 0.341 | 0.350 | 0.346 |
| **Aya-23-8B** | **0.325** | **0.327** | **0.324** | **0.327** |

### QLoRA fine-tuning (3 epochs, synthetic-only training)

Baseline is zero-shot Aya-23-8B. Δ row shows gain from zero-shot to the 5× condition.

| Condition | Overall | harvard | kimrichies | makerere |
|---|---|---|---|---|
| Aya (zero-shot) | 0.325 | 0.327 | 0.324 | 0.327 |
| + QLoRA 1× | 0.544 | 0.550 | 0.544 | 0.531 |
| **+ QLoRA 5×** | **0.611** | **0.615** | **0.610** | **0.566** |
| Δ (ZS → 5×) | +0.286 | +0.288 | +0.286 | +0.239 |

---

## AI Usage
In a first session, Claude Code was used as a code generation and iteration assistant. Starting from a detailed spec prompt, it scaffolded the  project from scratch, generating a Dockerfile, docker-compose configuration, evaluation script, and supporting files. From there, I issued targeted instructions to refine specific code blocks, update data schemas, add models to the registry, and keep documentation up to date. In a separate session on Professor joelawalsh01's account, Claude Code was used to migrate the project to uv and update GPU targets for Blackwell architecture. A final session on joelawalsh01's account added a QLoRA finetuning script. `claude_session_log.md` in `ai-usage-logs` contains the transcript of the first seesion conducted on haakso's device. Transcripts for the remaining sessions conducted on joelawalsh01's device are not included.
