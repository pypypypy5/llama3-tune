#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llama.tasks.topic_classification import build_messages, label_id_to_name


DEFAULT_TRAIN_URL = (
    "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/"
    "master/data/ag_news_csv/train.csv"
)
DEFAULT_TEST_URL = (
    "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/"
    "master/data/ag_news_csv/test.csv"
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



def download_if_needed(path: Path, url: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    print(f"Downloading {url} -> {path}")
    urllib.request.urlretrieve(url, path)



def load_ag_news_csv(path: Path, split_name: str) -> List[Dict]:
    samples: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if len(row) < 3:
                continue
            raw_label = int(row[0])
            label_id = raw_label - 1
            label_name = label_id_to_name(label_id)
            title = row[1].strip()
            description = row[2].strip()
            text = (title + "\n" + description).strip()
            samples.append(
                {
                    "id": f"{split_name}-raw-{i}",
                    "task": "topic_classification",
                    "source_dataset": "ag_news",
                    "text": text,
                    "label_id": label_id,
                    "label": label_name,
                    "messages": build_messages(text=text, label_id=label_id),
                }
            )
    if not samples:
        raise ValueError(f"No rows loaded from {path}")
    return samples



def stratified_split_train_val(
    samples: List[Dict],
    val_ratio: float,
    seed: int,
) -> tuple[List[Dict], List[Dict]]:
    by_label: Dict[int, List[Dict]] = defaultdict(list)
    for sample in samples:
        by_label[int(sample["label_id"])].append(sample)

    rng = random.Random(seed)
    train, val = [], []

    for label_id, rows in by_label.items():
        rng.shuffle(rows)
        n_val = int(len(rows) * val_ratio)
        if val_ratio > 0 and n_val == 0:
            n_val = 1
        val_rows = rows[:n_val]
        train_rows = rows[n_val:]
        train.extend(train_rows)
        val.extend(val_rows)

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val



def maybe_limit(samples: List[Dict], max_samples: int, seed: int) -> List[Dict]:
    if max_samples <= 0 or len(samples) <= max_samples:
        return samples
    rng = random.Random(seed)
    out = list(samples)
    rng.shuffle(out)
    return out[:max_samples]



def rewrite_ids(samples: List[Dict], split_name: str):
    for i, sample in enumerate(samples):
        sample["id"] = f"{split_name}-{i}"



def write_jsonl(path: Path, records: Iterable[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")



def label_distribution(records: List[Dict]) -> Dict[str, int]:
    counter = Counter(rec["label"] for rec in records)
    return {k: counter[k] for k in sorted(counter.keys())}



def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"

    if args.download_raw:
        train_csv = raw_dir / "train.csv"
        test_csv = raw_dir / "test.csv"
        download_if_needed(train_csv, args.train_url)
        download_if_needed(test_csv, args.test_url)
    else:
        if not args.train_csv or not args.test_csv:
            raise ValueError(
                "Set --download_raw, or provide both --train_csv and --test_csv"
            )
        train_csv = Path(args.train_csv)
        test_csv = Path(args.test_csv)

    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError("train/test CSV not found")

    train_all = load_ag_news_csv(train_csv, "train")
    test_records = load_ag_news_csv(test_csv, "test")
    train_records, val_records = stratified_split_train_val(
        train_all,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train_records = maybe_limit(train_records, args.max_train_samples, args.seed)
    val_records = maybe_limit(val_records, args.max_val_samples, args.seed)
    test_records = maybe_limit(test_records, args.max_test_samples, args.seed)

    rewrite_ids(train_records, "train")
    rewrite_ids(val_records, "val")
    rewrite_ids(test_records, "test")

    write_jsonl(output_dir / "train.jsonl", train_records)
    write_jsonl(output_dir / "val.jsonl", val_records)
    write_jsonl(output_dir / "test.jsonl", test_records)

    manifest = {
        "dataset": "ag_news",
        "seed": args.seed,
        "raw_train_csv": str(train_csv),
        "raw_test_csv": str(test_csv),
        "splits": {
            "train": {
                "num_samples": len(train_records),
                "label_distribution": label_distribution(train_records),
                "path": str(output_dir / "train.jsonl"),
            },
            "val": {
                "num_samples": len(val_records),
                "label_distribution": label_distribution(val_records),
                "path": str(output_dir / "val.jsonl"),
            },
            "test": {
                "num_samples": len(test_records),
                "label_distribution": label_distribution(test_records),
                "path": str(output_dir / "test.jsonl"),
            },
        },
    }

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
