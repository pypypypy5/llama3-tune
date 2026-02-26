import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.topic_classification import TopicDataPrepConfig, prepare_topic_classification_data
from eval.topic_classification import compute_metrics
from tasks.topic_classification import build_messages, normalize_label
from train import parse_lora_sft_args


class TopicPipelineTests(unittest.TestCase):
    def test_parse_train_config_args(self):
        cfg = parse_lora_sft_args(
            [
                "--ckpt_dir",
                "ckpt",
                "--tokenizer_path",
                "tokenizer.model",
                "--train_data",
                "data/train.jsonl",
                "--output_dir",
                "outputs/run1",
                "--lora_targets",
                "wq,wv",
            ]
        )
        self.assertEqual(cfg.ckpt_dir, "ckpt")
        self.assertEqual(cfg.lora_targets, ("wq", "wv"))

    def test_task_label_normalization_and_messages(self):
        msgs = build_messages("example article", label_id=3)
        self.assertEqual(msgs[-1]["content"], "sci_tech")
        self.assertEqual(normalize_label("Science & Technology"), "sci_tech")
        self.assertEqual(normalize_label("label: business"), "business")
        self.assertIsNone(normalize_label("unknown-label"))

    def test_compute_metrics(self):
        rows = [
            {"gold": "world", "pred": "world"},
            {"gold": "sports", "pred": "business"},
            {"gold": "business", "pred": "business"},
            {"gold": "sci_tech", "pred": None},
        ]
        metrics = compute_metrics(rows)
        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertIn("macro_f1", metrics)
        self.assertEqual(metrics["num_samples"], 4)

    def test_prepare_topic_data_from_local_csv(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            train_csv = tdp / "train.csv"
            test_csv = tdp / "test.csv"
            out_dir = tdp / "out"

            # AG News CSV format: label,title,description (label is 1..4)
            train_rows = [
                '1,"W1","World news one"\n',
                '1,"W2","World news two"\n',
                '2,"S1","Sports news one"\n',
                '2,"S2","Sports news two"\n',
                '3,"B1","Business news one"\n',
                '3,"B2","Business news two"\n',
                '4,"T1","Tech news one"\n',
                '4,"T2","Tech news two"\n',
            ]
            test_rows = [
                '1,"TW","World test"\n',
                '2,"TS","Sports test"\n',
                '3,"TB","Business test"\n',
                '4,"TT","Tech test"\n',
            ]
            train_csv.write_text("".join(train_rows), encoding="utf-8")
            test_csv.write_text("".join(test_rows), encoding="utf-8")

            cfg = TopicDataPrepConfig(
                output_dir=str(out_dir),
                val_ratio=0.5,
                seed=7,
                train_csv=str(train_csv),
                test_csv=str(test_csv),
                download_raw=False,
            )
            manifest = prepare_topic_classification_data(cfg)

            self.assertEqual(manifest["dataset"], "ag_news")
            self.assertTrue((out_dir / "train.jsonl").exists())
            self.assertTrue((out_dir / "val.jsonl").exists())
            self.assertTrue((out_dir / "test.jsonl").exists())

            train_lines = (out_dir / "train.jsonl").read_text(encoding="utf-8").strip().splitlines()
            val_lines = (out_dir / "val.jsonl").read_text(encoding="utf-8").strip().splitlines()
            test_lines = (out_dir / "test.jsonl").read_text(encoding="utf-8").strip().splitlines()

            self.assertEqual(len(train_lines), 4)
            self.assertEqual(len(val_lines), 4)
            self.assertEqual(len(test_lines), 4)

            sample = json.loads(train_lines[0])
            self.assertIn("messages", sample)
            self.assertEqual(sample["messages"][0]["role"], "system")


if __name__ == "__main__":
    unittest.main()
