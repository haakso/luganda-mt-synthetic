# Dataset

This folder contains all Luganda–English sentence pair data for the project, from the raw combined corpus through the train/validation/test/evaluation splits.

---

## Files

| File | Records | Description |
|---|---|---|
| `Eng-Lug_Dataset_combined_deduped_shuffled.xlsx` | 45,852 | Full combined, deduplicated, and shuffled corpus — the source from which splits are drawn |
| `training.json` | 16,926 | Training split |
| `validation.json` | 2,000 | Validation split |
| `test.json` | 2,000 | Test split |
| `evaluation.json` | 2,000 | Evaluation split |
| `synthetic_1x.json` | 16,926 | Synthetically generated dataset (1:1 ratio with training data) |
| `synthetic_5x.json` | 84,630 | Synthetically generated dataset (5:1 ratio with training data) |

---

## Combined Corpus

`Eng-Lug_Dataset_combined_deduped_shuffled.xlsx` is the full deduplicated source corpus — 45,852 sentence pairs drawn from four human-curated datasets and shuffled. The train/validation/test/evaluation splits (22,926 pairs total) are a subset of this file.

**Columns:** `dataset_origin`, `text_id`, `english`, `luganda`

**Source breakdown:**

| Source | Pairs |
|---|---|
| `kimrichies2023` | 40,367 |
| `harvard2023` | 5,060 |
| `makerere2024` | 395 |
| `kambale2025` | 30 |
| **Total** | **45,852** |

---

## Splits

The four JSON split files share a common schema:

```json
{
  "text_id": "kim_s07244",
  "luganda": "Abantu batera okumanya ebintu bingi ebibeetoolodde.",
  "english": "People usually know a lot of things in their surrounding.",
  "dataset_origin": "kimrichies2023",
  "is_synthetic": false,
  "derived_from": null,
  "seed_group": null
}
```

All records across splits are human-curated (`is_synthetic: false`). Source breakdown per split:

| Source | training (16,926) | validation (2,000) | test (2,000) | evaluation (2,000) |
|---|---|---|---|---|
| `kimrichies2023` | 14,831 | 1,762 | 1,752 | 1,772 |
| `harvard2023` | 1,927 | 219 | 226 | 212 |
| `makerere2024` | 156 | 18 | 21 | 14 |
| `kambale2025` | 12 | 1 | 1 | 2 |

---

## Synthetic Data

`synthetic_1x.json` and `synthetic_5x.json` were generated via few-shot prompting with **Google Gemini** (`gemini-2.5-flash`) using the training split as the seed corpus. The model was not fine-tuned — each API call provided 15 randomly sampled seed pairs as in-context examples and instructed Gemini to generate 30 new pairs per batch.

**Schema:**

```json
{
  "text_id": "gem1_s00001",
  "luganda": "...",
  "english": "...",
  "dataset_origin": "gemini_synthetic",
  "is_synthetic": true,
  "derived_from": ["kim_s39691", "har_s2496", "..."],
  "seed_group": 1
}
```

- `derived_from` — the `text_id`s of the seed pairs shown to Gemini as few-shot examples for that batch
- `seed_group` — 1-based batch index

**`text_id` conventions:**

| Dataset | Format | Range |
|---|---|---|
| `synthetic_1x.json` | `gem1_sNNNNN` (5-digit) | `gem1_s00001` – `gem1_s16926` |
| `synthetic_5x.json` | `gem5_sNNNNNN` (6-digit) | `gem5_s000001` – `gem5_s084630` |

**Generation parameters:**

| Parameter | Value |
|---|---|
| Model | `gemini-2.5-flash` |
| Pairs generated per API call | 30 |
| Seed examples per API call | 15 (randomly sampled) |
| Total API calls (1x) | ~565 |
| Total API calls (5x) | ~2,821 |
| Approximate total cost | ~$3.00 USD |

---

## Notes

Generation was done with checkpointing enabled — a checkpoint file was saved after every batch so that interrupted runs could be resumed without data loss. Checkpoint files (`checkpoint_1x.json` / `checkpoint_5x.json`) are temporary and are deleted on completion; they are not included in this repository.
