from __future__ import annotations

import json
from typing import Dict, List, Sequence

from llama.tasks.topic_classification import ALLOWED_LABELS, normalize_label

Dialog = List[Dict[str, str]]



def load_eval_samples(path: str, max_samples: int = 0) -> List[Dict]:
    samples: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            ex = json.loads(raw)
            if "messages" not in ex or "label" not in ex:
                raise ValueError(f"{path}:{line_no} missing messages/label")
            if not isinstance(ex["messages"], list) or len(ex["messages"]) < 2:
                raise ValueError(f"{path}:{line_no} has invalid messages")

            prompt_messages = [m for m in ex["messages"] if m["role"] != "assistant"]
            if not prompt_messages:
                raise ValueError(f"{path}:{line_no} has no non-assistant prompt messages")

            samples.append(
                {
                    "id": ex.get("id", f"sample-{line_no}"),
                    "label": ex["label"],
                    "messages": prompt_messages,
                }
            )
            if max_samples > 0 and len(samples) >= max_samples:
                break

    if not samples:
        raise ValueError(f"No samples loaded from {path}")
    return samples



def safe_div(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return a / b



def compute_metrics(rows: List[Dict]) -> Dict:
    labels = list(ALLOWED_LABELS)
    confusion: Dict[str, Dict[str, int]] = {
        gold: {pred: 0 for pred in labels + ["unknown"]} for gold in labels
    }

    for row in rows:
        gold = row["gold"]
        pred = row["pred"] if row["pred"] is not None else "unknown"
        if gold not in confusion:
            continue
        confusion[gold][pred] += 1

    total = sum(sum(v.values()) for v in confusion.values())
    correct = sum(confusion[label].get(label, 0) for label in labels)
    accuracy = safe_div(correct, total)

    per_label = {}
    f1_sum = 0.0
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[g][label] for g in labels if g != label)
        fn = sum(confusion[label][p] for p in confusion[label] if p != label)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        f1_sum += f1
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(confusion[label].values()),
        }

    macro_f1 = safe_div(f1_sum, len(labels))
    unknown_predictions = sum(1 for row in rows if row["pred"] is None)

    return {
        "num_samples": len(rows),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "unknown_prediction_rate": safe_div(unknown_predictions, len(rows)),
        "per_label": per_label,
        "confusion": confusion,
    }



def batched(items: Sequence[Dict], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]



def evaluate_topic_classifier(
    generator,
    samples: Sequence[Dict],
    max_batch_size: int,
    temperature: float,
    top_p: float,
    max_gen_len: int,
) -> List[Dict]:
    rows: List[Dict] = []

    for batch in batched(list(samples), max_batch_size):
        dialogs: List[Dialog] = [sample["messages"] for sample in batch]
        outputs = generator.chat_completion(
            dialogs=dialogs,
            temperature=temperature,
            top_p=top_p,
            max_gen_len=max_gen_len,
        )

        for sample, output in zip(batch, outputs):
            pred_raw = output["generation"]["content"].strip()
            pred = normalize_label(pred_raw)
            gold = normalize_label(sample["label"])
            rows.append(
                {
                    "id": sample["id"],
                    "gold": gold,
                    "pred": pred,
                    "pred_raw": pred_raw,
                    "correct": gold == pred,
                }
            )

    return rows
