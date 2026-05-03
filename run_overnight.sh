#!/bin/bash
 set -euo pipefail
 docker compose run --rm finetune \
   --model_path /models/aya-23-8B \
   --data_1x /data/synthetic_1x.json \
   --data_5x /data/synthetic_5x.json \
   --val_data /data/validation.json \
   --eval_data /data/evaluation.json \
   --output_dir /results/lora_overnight \
   2>&1 | tee results/finetune_overnight.log
