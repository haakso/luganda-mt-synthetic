"""
Download candidate models from HuggingFace into MODEL_CACHE_DIR.

Usage:
    HF_TOKEN=hf_... MODEL_CACHE_DIR=/path/to/models python scripts/download_models.py

Re-running is safe: existing non-empty directories are skipped.
"""
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

MODELS = [
    ("meta-llama/Llama-3.1-8B",    "Llama-3.1-8B"),
    ("mistralai/Mistral-7B-v0.3",   "Mistral-7B-v0.3"),
    ("google/gemma-7b",             "gemma-7b"),
    ("CohereLabs/aya-23-8B",        "aya-23-8B"),
]


def main():
    cache_dir = os.environ.get("MODEL_CACHE_DIR")
    if not cache_dir:
        print("ERROR: MODEL_CACHE_DIR is not set.", file=sys.stderr)
        sys.exit(1)

    hf_token = os.environ.get("HF_TOKEN")
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    for repo_id, local_name in MODELS:
        target = cache_path / local_name

        if target.exists() and any(target.iterdir()):
            print(f"[skip] {local_name} — directory already exists and is non-empty")
            continue

        print(f"[download] {repo_id} -> {target}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target),
            token=hf_token,
        )
        print(f"[done] {local_name}")


if __name__ == "__main__":
    main()
