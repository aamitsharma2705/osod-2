from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import ConcatDataset

from src.data.voc import VOCDataset
from src.data.coco import COCODataset


def build_voc_coco_dataset(
    data_root: str | Path,
    split: Literal["voc_coco_t1", "voc_coco_t2"],
    max_samples: int = 0,
):
    """
    Build a composite dataset per paper T1/T2 protocol:
    - Uses VOC2007/2012 trainval
    - Uses COCO train2017
    Note: assumes split files define which IDs belong to known/unknown; currently using full sets.
    """
    root = Path(data_root)

    voc_root = root / "VOC"
    coco_root = root / "COCO"

    voc2007 = VOCDataset(voc_root, year="2007", image_set="trainval", max_samples=max_samples)
    voc2012 = VOCDataset(voc_root, year="2012", image_set="trainval", max_samples=max_samples)
    coco_train = COCODataset(coco_root, image_set="train2017", max_samples=max_samples)

    return ConcatDataset([voc2007, voc2012, coco_train])

