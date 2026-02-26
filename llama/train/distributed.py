from __future__ import annotations

import os

import torch
import torch.distributed as dist
from fairscale.nn.model_parallel.initialize import (
    initialize_model_parallel,
    model_parallel_is_initialized,
)



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



def cleanup_distributed():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()



def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0



def log(msg: str):
    if is_main_process():
        print(msg, flush=True)
