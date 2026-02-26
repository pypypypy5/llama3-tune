#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from llama.lora import (
    LoRAConfig,
    apply_lora,
    count_trainable_parameters,
    freeze_non_lora_params,
    save_lora_adapter,
)
from llama.model import ModelArgs, Transformer
from llama.tokenizer import ChatFormat, Message, Tokenizer


IGNORE_INDEX = -100


class ChatSFTDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        formatter: ChatFormat,
        max_seq_len: int,
    ):
        self.samples: List[Tuple[List[int], List[int]]] = []
        self.max_seq_len = max_seq_len

        with open(data_path, "r", encoding="utf-8") as f:
            for line_no, raw in enumerate(f, 1):
                raw = raw.strip()
                if not raw:
                    continue
                example = json.loads(raw)
                messages = example.get("messages")
                if not isinstance(messages, list) or not messages:
                    raise ValueError(
                        f"{data_path}:{line_no} must contain non-empty 'messages' list"
                    )
                tokens, labels = self._encode_messages(messages, formatter)
                if all(label == IGNORE_INDEX for label in labels):
                    continue

                tokens = tokens[: self.max_seq_len]
                labels = labels[: self.max_seq_len]
                self.samples.append((tokens, labels))

        if not self.samples:
            raise ValueError(f"No valid training samples found in {data_path}")

    @staticmethod
    def _encode_messages(
        messages: Sequence[Message], formatter: ChatFormat
    ) -> Tuple[List[int], List[int]]:
        tokenizer = formatter.tokenizer
        tokens = [tokenizer.special_tokens["<|begin_of_text|>"]]
        labels = [IGNORE_INDEX]

        for message in messages:
            if "role" not in message or "content" not in message:
                raise ValueError("Each message must include 'role' and 'content'")
            encoded = formatter.encode_message(message)
            tokens.extend(encoded)
            if message["role"] == "assistant":
                labels.extend(encoded)
            else:
                labels.extend([IGNORE_INDEX] * len(encoded))

        return tokens, labels

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        return self.samples[index]


class Collator:
    def __init__(self, pad_token_id: int, max_seq_len: int):
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len

    def __call__(
        self, batch: Sequence[Tuple[List[int], List[int]]]
    ) -> Dict[str, torch.Tensor]:
        max_len = min(max(len(tokens) for tokens, _ in batch), self.max_seq_len)
        input_ids = torch.full(
            (len(batch), max_len), self.pad_token_id, dtype=torch.long
        )
        labels = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)

        for row, (tokens, sample_labels) in enumerate(batch):
            n = min(len(tokens), max_len)
            input_ids[row, :n] = torch.tensor(tokens[:n], dtype=torch.long)
            labels[row, :n] = torch.tensor(sample_labels[:n], dtype=torch.long)

        return {"input_ids": input_ids, "labels": labels}



def ensure_dist_env():
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")



def setup_distributed(model_parallel_size: int) -> int:
    ensure_dist_env()
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    if not model_parallel_is_initialized():
        initialize_model_parallel(model_parallel_size)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    world_size = dist.get_world_size()
    if world_size != model_parallel_size:
        raise ValueError(
            f"WORLD_SIZE ({world_size}) must equal model_parallel_size ({model_parallel_size})"
        )

    return local_rank



def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0



def log(msg: str):
    if is_main_process():
        print(msg, flush=True)



def load_model(
    ckpt_dir: str,
    tokenizer_path: str,
    max_seq_len: int,
    max_batch_size: int,
    device: torch.device,
) -> Tuple[Transformer, Tokenizer]:
    ckpt_files = sorted(Path(ckpt_dir).glob("*.pth"))
    if not ckpt_files:
        raise ValueError(f"No checkpoint shards found in {ckpt_dir}")

    ckpt_path = ckpt_files[get_model_parallel_rank()]
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    with open(Path(ckpt_dir) / "params.json", "r", encoding="utf-8") as f:
        params = json.load(f)

    model_args = ModelArgs(
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        **params,
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    if tokenizer.n_words != model_args.vocab_size:
        raise ValueError(
            f"Tokenizer vocab ({tokenizer.n_words}) != model vocab ({model_args.vocab_size})"
        )

    model = Transformer(model_args)
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    return model, tokenizer



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



def compute_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
    )



def save_training_manifest(args: argparse.Namespace, output_dir: Path, lora_config: LoRAConfig):
    payload = {"train_args": vars(args), "lora_config": asdict(lora_config)}
    with open(output_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)



def main():
    args = build_arg_parser().parse_args()
    local_rank = setup_distributed(args.model_parallel_size)

    if args.model_parallel_size != 1:
        raise ValueError("This script currently supports model_parallel_size=1 only")

    if args.batch_size <= 0 or args.gradient_accumulation_steps <= 0:
        raise ValueError("batch_size and gradient_accumulation_steps must be positive")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    start_time = time.time()
    model, tokenizer = load_model(
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path,
        max_seq_len=args.max_seq_len,
        max_batch_size=max(args.max_batch_size, args.batch_size),
        device=device,
    )
    formatter = ChatFormat(tokenizer)

    lora_config = LoRAConfig(
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=tuple(t.strip() for t in args.lora_targets.split(",") if t.strip()),
    )
    replaced = apply_lora(model, lora_config)
    freeze_non_lora_params(model)

    trainable, total = count_trainable_parameters(model)
    log(
        f"Loaded base model in {time.time() - start_time:.1f}s | "
        f"LoRA targets={len(replaced)} | trainable={trainable:,} / total={total:,} "
        f"({100.0 * trainable / total:.4f}%)"
    )

    dataset = ChatSFTDataset(
        data_path=args.train_data,
        formatter=formatter,
        max_seq_len=args.max_seq_len,
    )
    collator = Collator(pad_token_id=tokenizer.eos_id, max_seq_len=args.max_seq_len)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        drop_last=False,
    )

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    use_amp = device.type == "cuda" and args.precision in {"fp16", "bf16"}
    amp_dtype = torch.float16 if args.precision == "fp16" else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and args.precision == "fp16"))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_training_manifest(args, output_dir, lora_config)

    updates_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    total_updates = max(updates_per_epoch * args.epochs, 1)
    warmup_updates = int(total_updates * args.warmup_ratio)

    def lr_for_update(update_idx: int) -> float:
        if warmup_updates > 0 and update_idx < warmup_updates:
            return args.learning_rate * float(update_idx + 1) / float(warmup_updates)
        progress = (update_idx - warmup_updates) / max(total_updates - warmup_updates, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return args.min_lr + cosine * (args.learning_rate - args.min_lr)

    model.train()
    global_step = 0
    update_idx = 0

    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(dataloader, 1):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                logits = model(input_ids, start_pos=0, use_cache=False)
                loss = compute_lm_loss(logits, labels)
                loss = loss / args.gradient_accumulation_steps

            running_loss += loss.item()

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            global_step += 1
            if global_step % args.gradient_accumulation_steps != 0:
                continue

            lr = lr_for_update(update_idx)
            for group in optimizer.param_groups:
                group["lr"] = lr

            if scaler.is_enabled():
                scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=args.max_grad_norm,
            )

            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            update_idx += 1

            if is_main_process() and (update_idx % 10 == 0 or update_idx == 1):
                avg_loss = running_loss / max(step, 1)
                log(
                    f"epoch={epoch} step={step}/{len(dataloader)} "
                    f"update={update_idx}/{total_updates} lr={lr:.2e} loss={avg_loss:.4f}"
                )

            if args.save_every_steps > 0 and update_idx % args.save_every_steps == 0:
                save_lora_adapter(
                    model=model,
                    config=lora_config,
                    adapter_path=str(output_dir / f"adapter_step_{update_idx}.pt"),
                    metadata={
                        "epoch": str(epoch),
                        "update": str(update_idx),
                        "global_step": str(global_step),
                    },
                )

        epoch_loss = running_loss / max(len(dataloader), 1)
        log(f"finished epoch={epoch}/{args.epochs} avg_loss={epoch_loss:.4f}")

        save_lora_adapter(
            model=model,
            config=lora_config,
            adapter_path=str(output_dir / f"adapter_epoch_{epoch}.pt"),
            metadata={
                "epoch": str(epoch),
                "update": str(update_idx),
                "global_step": str(global_step),
            },
        )

    save_lora_adapter(
        model=model,
        config=lora_config,
        adapter_path=str(output_dir / "adapter_final.pt"),
        metadata={
            "epoch": str(args.epochs),
            "update": str(update_idx),
            "global_step": str(global_step),
        },
    )
    log(f"training complete | adapters saved under {output_dir}")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
