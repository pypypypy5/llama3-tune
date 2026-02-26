from .config import LoRATrainConfig, build_arg_parser, parse_lora_sft_args


def run_lora_sft(config: LoRATrainConfig):
    from .trainer import run_lora_sft as _run_lora_sft

    return _run_lora_sft(config)


def build_trainer(config: LoRATrainConfig):
    from .trainer import LoRASFTTrainer

    return LoRASFTTrainer(config)


def load_model_and_tokenizer(config: LoRATrainConfig, device):
    from .trainer import load_model_and_tokenizer as _load_model_and_tokenizer

    return _load_model_and_tokenizer(config, device)


def build_sft_dataloader(*args, **kwargs):
    from .data import build_sft_dataloader as _build_sft_dataloader

    return _build_sft_dataloader(*args, **kwargs)

__all__ = [
    "LoRATrainConfig",
    "build_sft_dataloader",
    "build_trainer",
    "build_arg_parser",
    "load_model_and_tokenizer",
    "parse_lora_sft_args",
    "run_lora_sft",
]
