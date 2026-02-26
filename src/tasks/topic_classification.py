from __future__ import annotations

import re
from typing import Dict, List, Optional


LABEL_ID_TO_NAME: Dict[int, str] = {
    0: "world",
    1: "sports",
    2: "business",
    3: "sci_tech",
}

LABEL_NAME_TO_ID: Dict[str, int] = {v: k for k, v in LABEL_ID_TO_NAME.items()}
ALLOWED_LABELS = tuple(LABEL_NAME_TO_ID.keys())

SYSTEM_PROMPT = (
    "You are a news topic classifier. "
    "Return exactly one label and nothing else: "
    "world, sports, business, or sci_tech."
)



def label_id_to_name(label_id: int) -> str:
    if label_id not in LABEL_ID_TO_NAME:
        raise ValueError(f"Unknown label id: {label_id}")
    return LABEL_ID_TO_NAME[label_id]



def label_name_to_id(label_name: str) -> int:
    key = normalize_label(label_name)
    if key is None:
        raise ValueError(f"Unknown label name: {label_name}")
    return LABEL_NAME_TO_ID[key]



def build_user_prompt(text: str) -> str:
    clean_text = text.strip()
    return (
        "Classify the topic of the following news article into one label: "
        "world, sports, business, or sci_tech.\n\n"
        f"Article:\n{clean_text}\n\n"
        "Label:"
    )



def build_messages(text: str, label_id: Optional[int] = None) -> List[dict]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(text)},
    ]
    if label_id is not None:
        messages.append({"role": "assistant", "content": label_id_to_name(label_id)})
    return messages



def normalize_label(raw: str) -> Optional[str]:
    if not raw:
        return None

    value = raw.strip().lower()
    value = value.replace("science and technology", "sci_tech")
    value = value.replace("science/technology", "sci_tech")
    value = value.replace("science & technology", "sci_tech")
    value = value.replace("science and tech", "sci_tech")
    value = value.replace("sci-tech", "sci_tech")
    value = value.replace("sci/tech", "sci_tech")
    value = value.replace("scitech", "sci_tech")
    value = value.replace(" ", "_")

    if value in LABEL_NAME_TO_ID:
        return value

    # Handle cases like "label: business" or "business\n"
    match = re.search(r"\b(world|sports|business|sci_tech)\b", value)
    if match:
        return match.group(1)

    # Also handle tokenization artifacts such as "label:_business".
    relaxed = re.sub(r"[^a-z_]+", " ", value)
    match = re.search(r"(world|sports|business|sci_tech)", relaxed)
    if match:
        return match.group(1)

    # Very short fallback for likely science/tech predictions.
    if "science" in value or "tech" in value or "technology" in value:
        return "sci_tech"

    return None
