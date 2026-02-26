from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Tuple

import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import get_model_parallel_rank
from torch.optim import AdamW

from llama.lora import (
    LoRAConfig,
    apply_lora,
    count_trainable_parameters,
    freeze_non_lora_params,
    save_lora_adapter,
)
from llama.model import ModelArgs, Transformer
from llama.tokenizer import ChatFormat, Tokenizer

from .config import LoRATrainConfig, validate_train_config
from .distributed import cleanup_distributed, log, setup_distributed
from data.sft_dataset import IGNORE_INDEX, build_sft_dataloader



def compute_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
    )



def load_model_and_tokenizer(
    config: LoRATrainConfig,
    device: torch.device,
) -> Tuple[Transformer, Tokenizer]:
    ckpt_files = sorted(Path(config.ckpt_dir).glob("*.pth"))
    if not ckpt_files:
        raise ValueError(f"No checkpoint shards found in {config.ckpt_dir}")

    ckpt_path = ckpt_files[get_model_parallel_rank()]
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    with open(Path(config.ckpt_dir) / "params.json", "r", encoding="utf-8") as f:
        params = json.load(f)

    model_args = ModelArgs(
        max_seq_len=config.max_seq_len,
        max_batch_size=max(config.max_batch_size, config.batch_size),
        **params,
    )
    tokenizer = Tokenizer(model_path=config.tokenizer_path)
    if tokenizer.n_words != model_args.vocab_size:
        raise ValueError(
            f"Tokenizer vocab ({tokenizer.n_words}) != model vocab ({model_args.vocab_size})"
        )

    model = Transformer(model_args)
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    return model, tokenizer



def _make_lr_schedule(config: LoRATrainConfig, total_updates: int) -> Callable[[int], float]:
    warmup_updates = int(total_updates * config.warmup_ratio)

    def lr_for_update(update_idx: int) -> float:
        if warmup_updates > 0 and update_idx < warmup_updates:
            return config.learning_rate * float(update_idx + 1) / float(warmup_updates)
        progress = (update_idx - warmup_updates) / max(total_updates - warmup_updates, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return config.min_lr + cosine * (config.learning_rate - config.min_lr)

    return lr_for_update



def _save_training_manifest(config: LoRATrainConfig, output_dir: Path, lora_config: LoRAConfig):
    payload = {"train_args": asdict(config), "lora_config": asdict(lora_config)}
    with open(output_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)



class LoRASFTTrainer:
    def __init__(self, config: LoRATrainConfig):
        validate_train_config(config)
        self.config = config

    def run(self):
        local_rank = setup_distributed(self.config.model_parallel_size)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

        device = (
            torch.device("cuda", local_rank)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        try:
            self._train(device)
        finally:
            cleanup_distributed()

    def _train(self, device: torch.device):
        config = self.config
        start_time = time.time()

        model, tokenizer = load_model_and_tokenizer(config=config, device=device)
        formatter = ChatFormat(tokenizer)

        lora_config = LoRAConfig(
            r=config.lora_r,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
            target_modules=config.lora_targets,
        )
        replaced = apply_lora(model, lora_config)
        freeze_non_lora_params(model)
        trainable_params = [p for p in model.parameters() if p.requires_grad]

        trainable, total = count_trainable_parameters(model)
        log(
            f"Loaded base model in {time.time() - start_time:.1f}s | "
            f"LoRA targets={len(replaced)} | trainable={trainable:,} / total={total:,} "
            f"({100.0 * trainable / total:.4f}%)"
        )

        dataloader = build_sft_dataloader(
            data_path=config.train_data,
            formatter=formatter,
            max_seq_len=config.max_seq_len,
            batch_size=config.batch_size,
            shuffle=True,
        )

        optimizer = AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        use_amp = device.type == "cuda" and config.precision in {"fp16", "bf16"}
        amp_dtype = torch.float16 if config.precision == "fp16" else torch.bfloat16
        scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and config.precision == "fp16"))

        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _save_training_manifest(config, output_dir, lora_config)

        updates_per_epoch = math.ceil(len(dataloader) / config.gradient_accumulation_steps)
        total_updates = max(updates_per_epoch * config.epochs, 1)
        lr_for_update = _make_lr_schedule(config, total_updates)

        model.train()
        global_step = 0
        update_idx = 0

        for epoch in range(1, config.epochs + 1):
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
                    loss = loss / config.gradient_accumulation_steps

                running_loss += loss.item()

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                global_step += 1
                if global_step % config.gradient_accumulation_steps != 0:
                    continue

                lr = lr_for_update(update_idx)
                for group in optimizer.param_groups:
                    group["lr"] = lr

                if scaler.is_enabled():
                    scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    trainable_params,
                    max_norm=config.max_grad_norm,
                )

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                update_idx += 1

                if update_idx % 10 == 0 or update_idx == 1:
                    avg_loss = running_loss / max(step, 1)
                    log(
                        f"epoch={epoch} step={step}/{len(dataloader)} "
                        f"update={update_idx}/{total_updates} lr={lr:.2e} loss={avg_loss:.4f}"
                    )

                if (
                    config.save_every_steps > 0
                    and update_idx % config.save_every_steps == 0
                ):
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
            log(f"finished epoch={epoch}/{config.epochs} avg_loss={epoch_loss:.4f}")

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
                "epoch": str(config.epochs),
                "update": str(update_idx),
                "global_step": str(global_step),
            },
        )
        log(f"training complete | adapters saved under {output_dir}")



def run_lora_sft(config: LoRATrainConfig):
    LoRASFTTrainer(config).run()
