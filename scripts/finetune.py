"""
Sequential QLoRA fine-tuning of Aya 23 8B on two synthetic dataset sizes (1x, 5x).
Each condition trains, evaluates with COMET, and logs to a nested MLflow run
under a single parent run named "finetune_lora".

Designed to run unattended overnight: a failure in one condition does not abort
the other; partial results (hypotheses, adapter checkpoints) are persisted as
soon as they are produced.
"""
import argparse
import gc
import json
import os
import sys
from pathlib import Path

import mlflow
import pandas as pd
import torch
from comet import load_from_checkpoint
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from trl import DataCollatorForCompletionOnlyLM


COMET_CHECKPOINT = Path("/opt/comet/CHECKPOINT").read_text().strip()
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
MAX_SEQ_LEN = 2048
RESPONSE_TEMPLATE = "\nEnglish:"


def load_base_model(model_path: str):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb,
        device_map="auto",
        local_files_only=True,
    )


def load_tokenizer(model_path: str, padding_side: str):
    tok = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, padding_side=padding_side
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def format_example(record: dict, eos_token: str) -> str:
    return (
        f"Translate the following from Luganda to English: {record['luganda']}\n"
        f"English: {record['english']}{eos_token}"
    )


def build_dataset(json_path: str, tokenizer, limit: int | None = None) -> Dataset:
    records = json.loads(Path(json_path).read_text())
    if limit:
        records = records[:limit]
    texts = [format_example(r, tokenizer.eos_token) for r in records]
    enc = tokenizer(texts, truncation=True, max_length=MAX_SEQ_LEN)
    return Dataset.from_dict(
        {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
    )


def apply_lora(model, args):
    # Don't double-enable gradient checkpointing; peft owns it for QLoRA.
    model = prepare_model_for_kbit_training(
        model, gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, cfg)


def run_inference(model, tokenizer, sources, batch_size):
    prompts = [
        f"Translate the following from Luganda to English: {s}\nEnglish:"
        for s in sources
    ]
    hypotheses = []
    model.eval()
    for i in tqdm(range(0, len(prompts), batch_size), desc="Inference"):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LEN,
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, do_sample=False, max_new_tokens=256)
        input_len = inputs["input_ids"].shape[1]
        for ids in out:
            hypotheses.append(tokenizer.decode(ids[input_len:], skip_special_tokens=True))
    return hypotheses


def score_with_comet(sources, hypotheses, references):
    if not Path(COMET_CHECKPOINT).exists():
        raise FileNotFoundError("COMET checkpoint missing — was the image built correctly?")
    comet = load_from_checkpoint(COMET_CHECKPOINT)
    data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(sources, hypotheses, references)]
    out = comet.predict(data, batch_size=8, gpus=1)
    return {"comet_mean": out.system_score, "comet_scores": out.scores}


def train_and_eval(condition_name, multiplier, train_path, args, eval_records):
    summary = {
        "condition": condition_name,
        "method": "LoRA",
        "data": f"{multiplier}x",
        "comet_mean": None,
        "examples": len(eval_records),
        "status": "running",
    }
    model = None
    trainer = None

    with mlflow.start_run(nested=True, run_name=condition_name):
        # CLI args + condition metadata. HF's MLflow callback also logs
        # TrainingArguments; the only overlap is `learning_rate` and the
        # values match, so MLflow accepts the second write as a no-op.
        for k, v in vars(args).items():
            mlflow.log_param(k, v)
        mlflow.log_param("method", "lora")
        mlflow.log_param("synthetic_multiplier", multiplier)

        try:
            model = load_base_model(args.model_path)
            tokenizer = load_tokenizer(args.model_path, padding_side="right")

            train_ds = build_dataset(train_path, tokenizer, limit=args.limit)
            val_ds = build_dataset(args.val_data, tokenizer, limit=args.limit)
            mlflow.log_param("train_examples", len(train_ds))

            model = apply_lora(model, args)

            collator = DataCollatorForCompletionOnlyLM(
                response_template=RESPONSE_TEMPLATE, tokenizer=tokenizer
            )

            output_subdir = Path(args.output_dir) / condition_name
            train_args = TrainingArguments(
                output_dir=str(output_subdir),
                num_train_epochs=args.epochs,
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=args.batch_size,
                gradient_accumulation_steps=args.grad_accum,
                learning_rate=args.learning_rate,
                eval_strategy="epoch",
                save_strategy="epoch",
                save_total_limit=2,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                logging_steps=50,
                report_to=["mlflow"],
                bf16=True,
                optim="paged_adamw_8bit",
                remove_unused_columns=False,
            )

            trainer = Trainer(
                model=model,
                args=train_args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                data_collator=collator,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
            )
            trainer.train()

            # Save adapter weights only (peft model.save_pretrained does this).
            adapter_dir = Path(args.output_dir) / condition_name
            adapter_dir.mkdir(parents=True, exist_ok=True)
            trainer.model.save_pretrained(str(adapter_dir))

            # Inference uses left-padding for batched decoder-only generation.
            tokenizer.padding_side = "left"
            sources = [r["luganda"] for r in eval_records]
            references = [r["english"] for r in eval_records]
            hypotheses = run_inference(trainer.model, tokenizer, sources, args.batch_size)
            df = pd.DataFrame(eval_records).assign(hypothesis=hypotheses)

            # Persist before COMET so a scoring crash doesn't lose inference work.
            jsonl_path = Path(args.output_dir) / f"{condition_name}_eval_results.jsonl"
            df.to_json(jsonl_path, orient="records", lines=True)
            print(f"[{condition_name}] wrote inference output to {jsonl_path}")

            # Free the LLM before COMET's xlm-roberta encoder goes onto the GPU.
            del model
            try:
                del trainer.optimizer
            except AttributeError:
                pass
            del trainer
            model = None
            trainer = None
            torch.cuda.empty_cache()
            gc.collect()

            try:
                overall = score_with_comet(sources, hypotheses, references)
                df = df.assign(comet=overall["comet_scores"])
                mlflow.log_metric("comet_mean", overall["comet_mean"])
                summary["comet_mean"] = overall["comet_mean"]

                for origin, group in df.groupby("dataset_origin"):
                    grp = score_with_comet(
                        group["luganda"].tolist(),
                        group["hypothesis"].tolist(),
                        group["english"].tolist(),
                    )
                    mlflow.log_metric(f"comet_mean_{origin}", grp["comet_mean"])
                    mlflow.log_param(f"num_examples_{origin}", len(group))

                df.to_json(jsonl_path, orient="records", lines=True)
                mlflow.log_artifact(str(jsonl_path))
                summary["status"] = "complete"
            except Exception as e:
                print(f"[{condition_name}] COMET scoring failed: {e!r}", file=sys.stderr)
                mlflow.log_param("comet_error", repr(e)[:250])
                summary["status"] = "trained_eval_failed"

        except Exception as e:
            print(f"[{condition_name}] training failed: {e!r}", file=sys.stderr)
            try:
                mlflow.log_param("error", repr(e)[:250])
            except Exception:
                pass
            summary["status"] = "failed"

        finally:
            model = None
            trainer = None
            torch.cuda.empty_cache()
            gc.collect()

    return summary


def format_summary(summaries):
    header = "Condition  | Method | Data | COMET  | Examples | Status"
    sep    = "-----------|--------|------|--------|----------|--------"
    rows = []
    for s in summaries:
        comet_str = f"{s['comet_mean']:.4f}" if s["comet_mean"] is not None else "  -   "
        rows.append(
            f"{s['condition']:<10} | {s['method']:<6} | {s['data']:<4} "
            f"| {comet_str:<6} | {s['examples']:<8} | {s['status']}"
        )
    return "\n".join([header, sep, *rows])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="path to base AYA 23 8B model weights")
    parser.add_argument("--data_1x", required=True, help="path to 1x synthetic data JSON file")
    parser.add_argument("--data_5x", required=True, help="path to 5x synthetic data JSON file")
    parser.add_argument("--val_data", required=True, help="path to validation set JSON file")
    parser.add_argument("--eval_data", required=True, help="path to final eval set JSON file")
    parser.add_argument("--output_dir", required=True, help="directory to save LoRA adapter checkpoints")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap train/val/eval to first N examples (smoke test)")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    eval_records = json.loads(Path(args.eval_data).read_text())
    if args.limit:
        eval_records = eval_records[: args.limit]

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    with mlflow.start_run(run_name="finetune_lora"):
        mlflow.log_param("model_path", args.model_path)
        mlflow.log_param("eval_data", args.eval_data)
        mlflow.log_param("epoch_budget", args.epochs)

        summaries = []
        for condition_name, multiplier, train_path in [
            ("lora_1x", 1, args.data_1x),
            ("lora_5x", 5, args.data_5x),
        ]:
            summaries.append(
                train_and_eval(condition_name, multiplier, train_path, args, eval_records)
            )

        table = format_summary(summaries)
        print("\n" + table)
        mlflow.log_text(table, "summary.txt")


if __name__ == "__main__":
    main()
