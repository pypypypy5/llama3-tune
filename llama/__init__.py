# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from typing import TYPE_CHECKING

__all__ = [
    "Dialog",
    "Llama",
    "LoRAConfig",
    "ModelArgs",
    "Tokenizer",
    "Transformer",
    "apply_lora",
    "count_trainable_parameters",
    "load_lora_adapter",
]

if TYPE_CHECKING:
    from .generation import Llama
    from .lora import (
        LoRAConfig,
        apply_lora,
        count_trainable_parameters,
        load_lora_adapter,
    )
    from .model import ModelArgs, Transformer
    from .tokenizer import Dialog, Tokenizer


def __getattr__(name):
    if name == "Llama":
        from .generation import Llama

        return Llama
    if name in {"LoRAConfig", "apply_lora", "count_trainable_parameters", "load_lora_adapter"}:
        from .lora import (
            LoRAConfig,
            apply_lora,
            count_trainable_parameters,
            load_lora_adapter,
        )

        return {
            "LoRAConfig": LoRAConfig,
            "apply_lora": apply_lora,
            "count_trainable_parameters": count_trainable_parameters,
            "load_lora_adapter": load_lora_adapter,
        }[name]
    if name in {"ModelArgs", "Transformer"}:
        from .model import ModelArgs, Transformer

        return {"ModelArgs": ModelArgs, "Transformer": Transformer}[name]
    if name in {"Dialog", "Tokenizer"}:
        from .tokenizer import Dialog, Tokenizer

        return {"Dialog": Dialog, "Tokenizer": Tokenizer}[name]
    raise AttributeError(f"module 'llama' has no attribute '{name}'")
