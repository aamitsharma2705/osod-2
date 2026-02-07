from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import ConcatDataset

from src.data.voc import VOCDataset, VOC_CLASSES
from src.data.coco import COCODataset
from src.data.splits import split_files, load_split_ids


def subset_dataset(dataset, keep_ids: list[str], id_key: str = "image_id"):
    """
    Subset a dataset by string image ids. Assumes targets carry image ids matching split lists.
    """
    id_set = set(keep_ids)

    def _filter_sample(idx):
        _, tgt = dataset[idx]
        img_id = tgt[id_key]
        if torch.is_tensor(img_id):
            img_id = int(img_id.item())
        # check both raw and zero-padded forms
        as_str = str(img_id)
        as_pad6 = as_str.zfill(6)
        as_pad12 = as_str.zfill(12)
        return (as_str in id_set) or (as_pad6 in id_set) or (as_pad12 in id_set)

    kept = [i for i in range(len(dataset)) if _filter_sample(i)]
    return torch.utils.data.Subset(dataset, kept)


def build_voc_coco_dataset(
    data_root: str | Path,
    split: Literal["voc_coco_t1", "voc_coco_t2"],
    max_samples: int = 0,
):
    """
    Build a composite dataset per paper T1/T2 protocol:
    - Uses VOC2007/2012 trainval
    - Uses COCO train2017
    - Applies split ID filters if split files exist; otherwise falls back to full sets.
    """
    root = Path(data_root)

    voc_root = root / "VOC"
    coco_root = root / "COCO"

    voc2007 = VOCDataset(voc_root, year="2007", image_set="trainval", max_samples=max_samples)
    voc2012 = VOCDataset(voc_root, year="2012", image_set="trainval", max_samples=max_samples)
    coco_train = COCODataset(
        coco_root,
        image_set="train2017",
        max_samples=max_samples,
        base_label_offset=len(VOC_CLASSES),
    )

    try:
        train_ids_path, _ = split_files(split)
        if train_ids_path.exists():
            keep_ids = load_split_ids(train_ids_path)
            # VOC ids are shorter (6 chars), COCO are longer; filter both.
            voc_ids = [x for x in keep_ids if len(x) <= 6]
            coco_ids = [x for x in keep_ids if len(x) > 6]
            if voc_ids:
                voc2007 = subset_dataset(voc2007, voc_ids, id_key="image_id")
                voc2012 = subset_dataset(voc2012, voc_ids, id_key="image_id")
            if coco_ids:
                coco_train = subset_dataset(coco_train, coco_ids, id_key="image_id")
    except Exception:
        # Fallback: use full datasets
        pass

    return ConcatDataset([voc2007, voc2012, coco_train])

