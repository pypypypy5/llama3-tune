from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class LoRATrainConfig:
    ckpt_dir: str
    tokenizer_path: str
    train_data: str
    output_dir: str

    model_parallel_size: int = 1
    max_seq_len: int = 2048
    max_batch_size: int = 8

    epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    min_lr: float = 2e-5
    max_grad_norm: float = 1.0

    lora_r: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05
    lora_targets: Tuple[str, ...] = ("wq", "wk", "wv", "wo")

    precision: str = "bf16"
    seed: int = 42
    save_every_steps: int = 0



def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LoRA SFT for Llama 3 native checkpoints")
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--model_parallel_size", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--max_batch_size", type=int, default=8)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--min_lr", type=float, default=2e-5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=32.0)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_targets",
        type=str,
        default="wq,wk,wv,wo",
        help="Comma-separated module names to inject LoRA into",
    )

    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every_steps", type=int, default=0)
    return parser



def parse_lora_sft_args(argv: Optional[list[str]] = None) -> LoRATrainConfig:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    lora_targets = tuple(t.strip() for t in args.lora_targets.split(",") if t.strip())
    if not lora_targets:
        raise ValueError("--lora_targets must contain at least one target module name")

    config = LoRATrainConfig(
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path,
        train_data=args.train_data,
        output_dir=args.output_dir,
        model_parallel_size=args.model_parallel_size,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        min_lr=args.min_lr,
        max_grad_norm=args.max_grad_norm,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_targets=lora_targets,
        precision=args.precision,
        seed=args.seed,
        save_every_steps=args.save_every_steps,
    )
    validate_train_config(config)
    return config



def validate_train_config(config: LoRATrainConfig):
    if config.model_parallel_size != 1:
        raise ValueError("This trainer currently supports model_parallel_size=1 only")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if config.max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
