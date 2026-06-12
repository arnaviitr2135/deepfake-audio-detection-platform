from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path


LABELS = {"real": 0, "fake": 1}
LABEL_NAMES = {0: "Genuine (Human)", 1: "Deepfake (AI-Generated)"}


@dataclass(frozen=True)
class AudioExample:
    split: str
    label_name: str
    label: int
    path: str


def find_dataset_zip(root: Path) -> Path:
    candidates = sorted(root.glob("*.zip"))
    if not candidates:
        raise FileNotFoundError("No dataset .zip file found in the project root.")
    for candidate in candidates:
        if "for" in candidate.name.lower() or "fake" in candidate.name.lower():
            return candidate
    return candidates[0]


def list_examples(zip_path: Path) -> list[AudioExample]:
    examples: list[AudioExample] = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir() or not info.filename.lower().endswith(".wav"):
                continue
            parts = info.filename.replace("\\", "/").split("/")
            if len(parts) < 4:
                continue
            split, label_name = parts[1].lower(), parts[2].lower()
            if split not in {"training", "validation", "testing"} or label_name not in LABELS:
                continue
            examples.append(AudioExample(split, label_name, LABELS[label_name], info.filename))
    return examples


def read_zip_bytes(zip_path: Path, member_path: str) -> bytes:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member_path) as fh:
            return fh.read()


def read_zip_buffer(zip_path: Path, member_path: str) -> io.BytesIO:
    return io.BytesIO(read_zip_bytes(zip_path, member_path))
