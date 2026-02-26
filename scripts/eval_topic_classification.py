#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for p in (str(SRC_ROOT), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval import compute_metrics, evaluate_topic_classifier, load_eval_samples



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate topic classification accuracy")
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--eval_data", type=str, required=True)
    parser.add_argument("--lora_adapter_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--max_batch_size", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_gen_len", type=int, default=8)
    return parser.parse_args()



def main():
    args = parse_args()
    samples = load_eval_samples(args.eval_data, max_samples=args.max_samples)
    from llama import Llama

    generator = Llama.build(
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size,
        lora_adapter_path=args.lora_adapter_path,
    )

    rows = evaluate_topic_classifier(
        generator=generator,
        samples=samples,
        max_batch_size=args.max_batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
        max_gen_len=args.max_gen_len,
    )

    metrics = compute_metrics(rows)
    result = {
        "eval_data": args.eval_data,
        "lora_adapter_path": args.lora_adapter_path,
        "metrics": metrics,
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.output_path:
        out_path = Path(args.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        pred_path = out_path.with_suffix(".predictions.jsonl")
        with open(pred_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
