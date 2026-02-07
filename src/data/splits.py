"""
Utilities for VOC–COCO T1/T2 split handling.
Provide filenames and helpers to load class prompts.
"""

from pathlib import Path
from typing import Tuple, List


def split_files(split: str) -> Tuple[Path, Path]:
    """
    Returns (train_list, val_list) paths for a given split name.
    """
    base = Path("data") / "splits"
    if split == "voc_coco_t1":
        return base / "voc_coco_t1_train.txt", base / "voc_coco_t1_val.txt"
    if split == "voc_coco_t2":
        return base / "voc_coco_t2_train.txt", base / "voc_coco_t2_val.txt"
    if split == "voc_sanity":
        # Re-use VOC trainval for a tiny sanity run (caller may subset)
        return base / "voc_sanity_train.txt", base / "voc_sanity_val.txt"
    raise ValueError(f"Unknown split: {split}")


def prompt_file(split: str) -> Path:
    base = Path("data") / "prompts"
    if split in ("voc_coco_t1", "voc_coco_t2"):
        candidate = base / f"{split}.txt"
        if candidate.exists():
            return candidate
        # fallback to combined full list if split-specific prompt missing
        return base / "voc_coco_full.txt"
    if split == "voc_sanity":
        return base / "voc_sanity.txt"
    raise ValueError(f"No prompt file configured for split: {split}")


def load_split_ids(path: Path) -> list[str]:
    with path.open() as f:
        return [line.strip() for line in f if line.strip()]

