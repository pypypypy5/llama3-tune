from __future__ import annotations

import json
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from llama.tokenizer import ChatFormat, Message


IGNORE_INDEX = -100


class ChatSFTDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        formatter: ChatFormat,
        max_seq_len: int,
    ):
        self.samples: List[Tuple[List[int], List[int]]] = []
        self.max_seq_len = max_seq_len

        with open(data_path, "r", encoding="utf-8") as f:
            for line_no, raw in enumerate(f, 1):
                raw = raw.strip()
                if not raw:
                    continue
                example = json.loads(raw)
                messages = example.get("messages")
                if not isinstance(messages, list) or not messages:
                    raise ValueError(
                        f"{data_path}:{line_no} must contain non-empty 'messages' list"
                    )
                tokens, labels = self._encode_messages(messages, formatter)
                if all(label == IGNORE_INDEX for label in labels):
                    continue

                tokens = tokens[: self.max_seq_len]
                labels = labels[: self.max_seq_len]
                self.samples.append((tokens, labels))

        if not self.samples:
            raise ValueError(f"No valid training samples found in {data_path}")

    @staticmethod
    def _encode_messages(
        messages: Sequence[Message], formatter: ChatFormat
    ) -> Tuple[List[int], List[int]]:
        tokenizer = formatter.tokenizer
        tokens = [tokenizer.special_tokens["<|begin_of_text|>"]]
        labels = [IGNORE_INDEX]

        for message in messages:
            if "role" not in message or "content" not in message:
                raise ValueError("Each message must include 'role' and 'content'")
            encoded = formatter.encode_message(message)
            tokens.extend(encoded)
            if message["role"] == "assistant":
                labels.extend(encoded)
            else:
                labels.extend([IGNORE_INDEX] * len(encoded))

        return tokens, labels

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        return self.samples[index]


class ChatSFTCollator:
    def __init__(self, pad_token_id: int, max_seq_len: int):
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len

    def __call__(
        self, batch: Sequence[Tuple[List[int], List[int]]]
    ) -> Dict[str, torch.Tensor]:
        max_len = min(max(len(tokens) for tokens, _ in batch), self.max_seq_len)
        input_ids = torch.full(
            (len(batch), max_len), self.pad_token_id, dtype=torch.long
        )
        labels = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)

        for row, (tokens, sample_labels) in enumerate(batch):
            n = min(len(tokens), max_len)
            input_ids[row, :n] = torch.tensor(tokens[:n], dtype=torch.long)
            labels[row, :n] = torch.tensor(sample_labels[:n], dtype=torch.long)

        return {"input_ids": input_ids, "labels": labels}


def build_sft_dataloader(
    data_path: str,
    formatter: ChatFormat,
    max_seq_len: int,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    dataset = ChatSFTDataset(
        data_path=data_path,
        formatter=formatter,
        max_seq_len=max_seq_len,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=ChatSFTCollator(
            pad_token_id=formatter.tokenizer.eos_id,
            max_seq_len=max_seq_len,
        ),
        drop_last=False,
    )
