from .topic_classification import TopicDataPrepConfig, prepare_topic_classification_data

__all__ = [
    "IGNORE_INDEX",
    "ChatSFTCollator",
    "ChatSFTDataset",
    "TopicDataPrepConfig",
    "build_sft_dataloader",
    "prepare_topic_classification_data",
]


def __getattr__(name):
    if name in {"IGNORE_INDEX", "ChatSFTCollator", "ChatSFTDataset", "build_sft_dataloader"}:
        from .sft_dataset import (
            IGNORE_INDEX,
            ChatSFTCollator,
            ChatSFTDataset,
            build_sft_dataloader,
        )

        return {
            "IGNORE_INDEX": IGNORE_INDEX,
            "ChatSFTCollator": ChatSFTCollator,
            "ChatSFTDataset": ChatSFTDataset,
            "build_sft_dataloader": build_sft_dataloader,
        }[name]
    if name in {"TopicDataPrepConfig", "prepare_topic_classification_data"}:
        return {
            "TopicDataPrepConfig": TopicDataPrepConfig,
            "prepare_topic_classification_data": prepare_topic_classification_data,
        }[name]
    raise AttributeError(f"module 'data' has no attribute '{name}'")
