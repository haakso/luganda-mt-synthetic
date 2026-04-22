"""
LG->EN evaluation harness.
Usage: python evaluate.py --model <model_key> [--batch_size 8] [--output results.jsonl]
"""
import argparse
import json
import os
from pathlib import Path

import mlflow
import pandas as pd
import torch
from comet import load_from_checkpoint
from tqdm import tqdm #progress bar
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_REGISTRY = {
    "llama3":  "/models/Llama-3.1-8B",
    "mistral": "/models/Mistral-7B-v0.3",
    "aya":     "/models/aya-23-8B",
    "gemma":   "/models/gemma-7b-multilingual",
}

TEST_DATA_PATH = "/data/test_initial/pairs.jsonl"
COMET_CHECKPOINT = "/opt/comet/wmt22-comet-da/checkpoints/model.ckpt"


def load_model_and_tokenizer(model_path: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        local_files_only=True,
    )
    model.eval()
    return model, tokenizer


def format_prompt(src_text: str, model_key: str) -> str:
    # Basic fallback — replace with model-specific chat templates once finalized
    return f"Translate the following from Luganda to English: {src_text}"


def run_inference(model, tokenizer, prompts: list[str], batch_size: int) -> list[str]:
    translations = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Inference"):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=256,
            )
        input_lengths = inputs["attention_mask"].sum(dim=1)
        for j, ids in enumerate(output_ids):
            new_tokens = ids[input_lengths[j]:]
            translations.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    return translations


def score_with_comet(sources: list[str], hypotheses: list[str], references: list[str]) -> dict:
    if not Path(COMET_CHECKPOINT).exists():
        raise FileNotFoundError(
            "COMET checkpoint not found. Was the image built correctly?"
        )
    comet_model = load_from_checkpoint(COMET_CHECKPOINT)
    data = [
        {"src": s, "mt": h, "ref": r}
        for s, h, r in zip(sources, hypotheses, references)
    ]
    output = comet_model.predict(data, batch_size=8, gpus=1)
    return {"comet_mean": output.system_score, "comet_scores": output.scores}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODEL_REGISTRY.keys())
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output", default=None, help="Path to write per-sentence results JSONL")
    args = parser.parse_args()

    model_path = MODEL_REGISTRY[args.model]

    pairs = [json.loads(line) for line in Path(TEST_DATA_PATH).read_text().splitlines()]
    df = pd.DataFrame(pairs)  # keys: text_id, luganda, english, dataset_origin, is_synthetic, derived_from, seed_group

    prompts = [format_prompt(row["luganda"], args.model) for _, row in df.iterrows()]

    model, tokenizer = load_model_and_tokenizer(model_path)
    hypotheses = run_inference(model, tokenizer, prompts, args.batch_size)

    comet_results = score_with_comet(df["luganda"].tolist(), hypotheses, df["english"].tolist())
    df = df.assign(hypothesis=hypotheses, comet=comet_results["comet_scores"])

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    with mlflow.start_run(run_name=args.model):
        mlflow.log_param("model", args.model)
        mlflow.log_param("model_path", model_path)
        mlflow.log_param("test_set", TEST_DATA_PATH)
        mlflow.log_param("num_examples", len(df))
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_metric("comet_mean", comet_results["comet_mean"])

        for origin, group in df.groupby("dataset_origin"):
            origin_comet = score_with_comet(
                group["luganda"].tolist(),
                group["hypothesis"].tolist(),
                group["english"].tolist(),
            )
            mlflow.log_metric(f"comet_mean_{origin}", origin_comet["comet_mean"])
            mlflow.log_param(f"num_examples_{origin}", len(group))

        if args.output:
            out_path = Path(args.output)
            df.to_json(out_path, orient="records", lines=True)
            mlflow.log_artifact(str(out_path))

    print(f"COMET (wmt22-comet-da): {comet_results['comet_mean']:.4f}")


if __name__ == "__main__":
    main()
