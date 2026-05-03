"""
generate_synthetic_luganda.py
------------------------------
Uses Gemini (via prompting) to generate synthetic English/Luganda sentence pairs
from a seed dataset (training.json).

Outputs:
  - synthetic_1x.json  : same number of pairs as seed data
  - synthetic_5x.json  : 5x the number of pairs as seed data

Saves progress after every batch so it can resume if interrupted.
"""

from google import genai

import json
import math
import os
import random
import re
import time

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED_FILE         = "training.json"
OUTPUT_1X         = "synthetic_1x.json"
OUTPUT_5X         = "synthetic_5x.json"
CHECKPOINT_1X     = "checkpoint_1x.json"
CHECKPOINT_5X     = "checkpoint_5x.json"

API_KEY           = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
MODEL_NAME        = "gemini-2.5-flash"

BATCH_SIZE        = 30
EXAMPLES_PER_CALL = 15
MAX_RETRIES       = 5
RETRY_DELAY       = 65

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

client = genai.Client(api_key=API_KEY)


# ---------------------------------------------------------------------------
# Load seed data
# ---------------------------------------------------------------------------

def load_seed(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data):,} seed records from {path}")
    return data


# ---------------------------------------------------------------------------
# text_id generators
# ---------------------------------------------------------------------------

def make_1x_id(n: int) -> str:
    return f"gem1_s{n:05d}"

def make_5x_id(n: int) -> str:
    return f"gem5_s{n:06d}"


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint(path: str) -> list[dict]:
    """Load saved progress if it exists, otherwise return empty list."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  Resuming from checkpoint: {len(data):,} records already saved")
        return data
    return []

def save_checkpoint(records: list[dict], path: str):
    """Save current progress to checkpoint file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Prompt + API call
# ---------------------------------------------------------------------------

def build_prompt(examples: list[dict], n_to_generate: int) -> str:
    example_block = "\n".join(
        f"  English: {e['english']}\n  Luganda:  {e['luganda']}"
        for e in examples
    )
    return f"""You are a bilingual linguistic expert in English and Luganda (a Bantu language spoken in Uganda).

Below are {len(examples)} real English-Luganda sentence pairs from a curated dataset:

{example_block}

Your task: generate {n_to_generate} NEW English-Luganda sentence pairs.

Rules:
- Every sentence must be original - do NOT copy or closely paraphrase any example above.
- Vary topics (daily life, health, education, agriculture, greetings, commerce, family, etc.)
- Vary sentence length (short to medium; avoid very long sentences)
- Vary formality (conversational and formal registers both welcome)
- Luganda translations must be accurate and natural
- Do NOT include numbering, bullet points, or extra commentary

Respond ONLY with a valid JSON array, no markdown fences, like this:
[
  {{"english": "...", "luganda": "..."}},
  {{"english": "...", "luganda": "..."}}
]"""


def call_gemini(examples: list[dict], n: int) -> list[dict]:
    prompt = build_prompt(examples, n)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
            )
            text = response.text.strip()

            # Strip accidental markdown fences
            text = re.sub(r"^```[a-z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
            text = text.strip()

            pairs = json.loads(text)

            if not isinstance(pairs, list):
                raise ValueError("Response is not a JSON array")

            valid = []
            for p in pairs:
                if isinstance(p, dict) and "english" in p and "luganda" in p:
                    valid.append({
                        "english": str(p["english"]).strip(),
                        "luganda": str(p["luganda"]).strip(),
                    })
            if not valid:
                raise ValueError("No valid pairs found in response")

            return valid

        except (json.JSONDecodeError, ValueError) as e:
            print(f"  [Attempt {attempt}/{MAX_RETRIES}] Parse error: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
        except Exception as e:
            print(f"  [Attempt {attempt}/{MAX_RETRIES}] API error: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    print("  WARNING: All retries failed for this batch. Skipping.")
    return []


# ---------------------------------------------------------------------------
# Batch generation (with checkpointing)
# ---------------------------------------------------------------------------

def generate_pairs(
    seed: list[dict],
    total: int,
    id_fn,
    prefix_label: str,
    checkpoint_file: str,
    output_file: str,
    dataset_origin: str = "gemini_synthetic",
) -> list[dict]:

    # Resume from checkpoint if available
    results = load_checkpoint(checkpoint_file)
    counter = len(results) + 1
    num_batches = math.ceil(total / BATCH_SIZE)
    start_batch = math.floor(len(results) / BATCH_SIZE)

    if len(results) >= total:
        print(f"  Already complete ({len(results):,} records). Skipping generation.")
        return results

    print(f"\nGenerating {total:,} pairs in {num_batches} batches ({BATCH_SIZE}/batch) [{prefix_label}]")
    if start_batch > 0:
        print(f"  Skipping batches 1-{start_batch} (already done)")

    for batch_idx in range(start_batch, num_batches):
        remaining = total - len(results)
        n = min(BATCH_SIZE, remaining)
        if n <= 0:
            break

        examples = random.sample(seed, min(EXAMPLES_PER_CALL, len(seed)))
        seed_ids_used = [e["text_id"] for e in examples]

        print(f"  Batch {batch_idx + 1}/{num_batches}: requesting {n} pairs...", end=" ", flush=True)
        pairs = call_gemini(examples, n)
        print(f"got {len(pairs)}")

        for pair in pairs:
            record = {
                "text_id":        id_fn(counter),
                "luganda":        pair["luganda"],
                "english":        pair["english"],
                "dataset_origin": dataset_origin,
                "is_synthetic":   True,
                "derived_from":   seed_ids_used,
                "seed_group":     batch_idx + 1,
            }
            results.append(record)
            counter += 1

            if len(results) >= total:
                break

        # Save progress after every batch
        save_checkpoint(results, checkpoint_file)

        time.sleep(1)

    print(f"  Done. Generated {len(results):,} records.")

    # Deduplicate
    results = deduplicate(results, field="english")

    # Save final output
    save(results, output_file)

    # Clean up checkpoint file
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"  Removed checkpoint file {checkpoint_file}")

    return results


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate(records: list[dict], field: str = "english") -> list[dict]:
    seen = set()
    unique = []
    for r in records:
        key = r[field].lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(r)
    removed = len(records) - len(unique)
    if removed:
        print(f"  Removed {removed} duplicate(s) on '{field}'")
    return unique


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save(records: list[dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(records):,} records -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    seed = load_seed(SEED_FILE)
    total_seed = len(seed)

    # -- 1x dataset SKIPPED (already complete) --

    # -- 5x dataset --
    generate_pairs(
        seed=seed,
        total=total_seed * 5,
        id_fn=make_5x_id,
        prefix_label="5x / gem5",
        checkpoint_file=CHECKPOINT_5X,
        output_file=OUTPUT_5X,
    )

    print("\nAll done!")

if __name__ == "__main__":
    main()
