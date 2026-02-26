#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.topic_classification import (  # noqa: E402
    DEFAULT_TEST_URL,
    DEFAULT_TRAIN_URL,
    TopicDataPrepConfig,
    prepare_topic_classification_data,
)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare AG News topic classification SFT data (JSONL)."
    )
    parser.add_argument("--output_dir", type=str, default="data/topic_classification/ag_news")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--train_csv", type=str, default="")
    parser.add_argument("--test_csv", type=str, default="")
    parser.add_argument("--download_raw", action="store_true")
    parser.add_argument("--train_url", type=str, default=DEFAULT_TRAIN_URL)
    parser.add_argument("--test_url", type=str, default=DEFAULT_TEST_URL)

    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_val_samples", type=int, default=0)
    parser.add_argument("--max_test_samples", type=int, default=0)
    return parser.parse_args()



def main():
    args = parse_args()
    config = TopicDataPrepConfig(
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        download_raw=args.download_raw,
        train_url=args.train_url,
        test_url=args.test_url,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
    )
    manifest = prepare_topic_classification_data(config)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
