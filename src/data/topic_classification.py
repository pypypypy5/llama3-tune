from __future__ import annotations

import csv
import json
import random
import urllib.request
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from tasks.topic_classification import build_messages, label_id_to_name

DEFAULT_TRAIN_URL = (
    "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/"
    "master/data/ag_news_csv/train.csv"
)
DEFAULT_TEST_URL = (
    "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/"
    "master/data/ag_news_csv/test.csv"
)


@dataclass
class TopicDataPrepConfig:
    output_dir: str = "data/topic_classification/ag_news"
    val_ratio: float = 0.1
    seed: int = 42

    train_csv: str = ""
    test_csv: str = ""
    download_raw: bool = False
    train_url: str = DEFAULT_TRAIN_URL
    test_url: str = DEFAULT_TEST_URL

    max_train_samples: int = 0
    max_val_samples: int = 0
    max_test_samples: int = 0



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
) -> Tuple[List[Dict], List[Dict]]:
    by_label: Dict[int, List[Dict]] = defaultdict(list)
    for sample in samples:
        by_label[int(sample["label_id"])].append(sample)

    rng = random.Random(seed)
    train, val = [], []

    for _, rows in by_label.items():
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



def prepare_topic_classification_data(config: TopicDataPrepConfig) -> Dict:
    output_dir = Path(config.output_dir)
    raw_dir = output_dir / "raw"

    if config.download_raw:
        train_csv = raw_dir / "train.csv"
        test_csv = raw_dir / "test.csv"
        download_if_needed(train_csv, config.train_url)
        download_if_needed(test_csv, config.test_url)
    else:
        if not config.train_csv or not config.test_csv:
            raise ValueError(
                "Set download_raw=True, or provide both train_csv and test_csv"
            )
        train_csv = Path(config.train_csv)
        test_csv = Path(config.test_csv)

    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError("train/test CSV not found")

    train_all = load_ag_news_csv(train_csv, "train")
    test_records = load_ag_news_csv(test_csv, "test")
    train_records, val_records = stratified_split_train_val(
        train_all,
        val_ratio=config.val_ratio,
        seed=config.seed,
    )

    train_records = maybe_limit(train_records, config.max_train_samples, config.seed)
    val_records = maybe_limit(val_records, config.max_val_samples, config.seed)
    test_records = maybe_limit(test_records, config.max_test_samples, config.seed)

    rewrite_ids(train_records, "train")
    rewrite_ids(val_records, "val")
    rewrite_ids(test_records, "test")

    write_jsonl(output_dir / "train.jsonl", train_records)
    write_jsonl(output_dir / "val.jsonl", val_records)
    write_jsonl(output_dir / "test.jsonl", test_records)

    manifest = {
        "dataset": "ag_news",
        "config": asdict(config),
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

    return manifest
