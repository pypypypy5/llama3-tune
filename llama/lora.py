# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class LoRAConfig:
    r: int = 16
    alpha: float = 32.0
    dropout: float = 0.05
    target_modules: Tuple[str, ...] = ("wq", "wk", "wv", "wo")


class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Module, r: int, alpha: float, dropout: float):
        super().__init__()
        if not hasattr(base_layer, "weight"):
            raise ValueError(f"{base_layer.__class__.__name__} has no weight parameter")
        if r <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {r}")

        self.base_layer = base_layer
        self.r = r
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        out_features, in_features = self.base_layer.weight.shape
        device = self.base_layer.weight.device
        param_dtype = (
            torch.float32
            if self.base_layer.weight.dtype in (torch.float16, torch.bfloat16)
            else self.base_layer.weight.dtype
        )

        self.lora_A = nn.Parameter(
            torch.empty(r, in_features, device=device, dtype=param_dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, r, device=device, dtype=param_dtype)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)

        lora_out = F.linear(
            F.linear(self.dropout(x).to(self.lora_A.dtype), self.lora_A),
            self.lora_B,
        )
        return base_out + lora_out.to(base_out.dtype) * self.scaling



def _resolve_parent_module(root: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]



def apply_lora(model: nn.Module, config: LoRAConfig) -> List[str]:
    target_set = set(config.target_modules)
    replaced: List[str] = []
    already_wrapped = 0

    for module_name, module in list(model.named_modules()):
        leaf_name = module_name.split(".")[-1]
        if leaf_name not in target_set:
            continue
        if isinstance(module, LoRALinear):
            already_wrapped += 1
            continue
        if not hasattr(module, "weight"):
            continue

        parent, attr_name = _resolve_parent_module(model, module_name)
        setattr(
            parent,
            attr_name,
            LoRALinear(module, r=config.r, alpha=config.alpha, dropout=config.dropout),
        )
        replaced.append(module_name)

    if not replaced and already_wrapped == 0:
        raise ValueError(f"No modules matched target_modules={sorted(target_set)}")

    return replaced



def freeze_non_lora_params(model: nn.Module):
    for name, param in model.named_parameters():
        param.requires_grad = "lora_A" in name or "lora_B" in name



def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu()
        for name, param in model.state_dict().items()
        if "lora_A" in name or "lora_B" in name
    }



def save_lora_adapter(
    model: nn.Module,
    config: LoRAConfig,
    adapter_path: str,
    metadata: Optional[Dict[str, str]] = None,
):
    payload = {
        "lora_config": asdict(config),
        "state_dict": lora_state_dict(model),
        "metadata": metadata or {},
    }
    adapter_file = Path(adapter_path)
    adapter_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, adapter_file)



def load_lora_adapter(
    model: nn.Module,
    adapter_path: str,
    device: str = "cpu",
) -> LoRAConfig:
    payload = torch.load(adapter_path, map_location=device)
    if "lora_config" not in payload or "state_dict" not in payload:
        raise ValueError("Adapter checkpoint must contain 'lora_config' and 'state_dict'")

    config = LoRAConfig(**payload["lora_config"])
    apply_lora(model, config)
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)

    unexpected_non_lora = [name for name in unexpected if "lora_" not in name]
    if unexpected_non_lora:
        raise ValueError(f"Unexpected non-LoRA keys in adapter: {unexpected_non_lora}")

    if missing:
        missing_lora = [name for name in missing if "lora_" in name]
        if missing_lora:
            raise ValueError(f"Missing LoRA keys in adapter: {missing_lora}")

    return config



def count_trainable_parameters(model: nn.Module) -> Tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
