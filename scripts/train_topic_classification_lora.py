#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llama.train import LoRATrainConfig, run_lora_sft



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LoRA for topic classification with AG News-formatted data"
    )
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument(
        "--data_dir", type=str, default="data/topic_classification/ag_news"
    )
    parser.add_argument("--output_dir", type=str, default="outputs/topic-cls-lora")

    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=32.0)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_targets", type=str, default="wq,wk,wv,wo")

    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every_steps", type=int, default=0)
    return parser.parse_args()



def main():
    args = parse_args()
    config = LoRATrainConfig(
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path,
        train_data=f"{args.data_dir}/train.jsonl",
        output_dir=args.output_dir,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_targets=tuple(t.strip() for t in args.lora_targets.split(",") if t.strip()),
        precision=args.precision,
        seed=args.seed,
        save_every_steps=args.save_every_steps,
    )
    run_lora_sft(config)


if __name__ == "__main__":
    main()
