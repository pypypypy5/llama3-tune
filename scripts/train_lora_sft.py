#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for p in (str(SRC_ROOT), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from train import parse_lora_sft_args, run_lora_sft



def main():
    config = parse_lora_sft_args()
    run_lora_sft(config)


if __name__ == "__main__":
    main()
