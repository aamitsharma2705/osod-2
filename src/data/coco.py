from pathlib import Path

import torch
from pycocotools.coco import COCO
from PIL import Image
from torch.utils.data import Dataset


class COCODataset(Dataset):
    """
    Minimal COCO detection dataset loader.
    """

    def __init__(
        self,
        root: str | Path,
        image_set: str = "train2017",
        max_samples: int = 0,
        base_label_offset: int = 0,
    ):
        self.root = Path(root)
        self.image_set = image_set
        ann_file = self.root / "annotations" / f"instances_{image_set}.json"
        if not ann_file.exists():
            raise FileNotFoundError(f"Missing COCO annotations: {ann_file}")

        self.coco = COCO(str(ann_file))
        self.ids = list(sorted(self.coco.imgs.keys()))
        if max_samples and max_samples > 0:
            self.ids = self.ids[:max_samples]

        self.img_folder = self.root / image_set

        # Build contiguous label mapping starting after base_label_offset
        cat_ids = sorted(self.coco.getCatIds())
        self.cat_id_to_contig = {
            cid: base_label_offset + 1 + i for i, cid in enumerate(cat_ids)
        }

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        img_info = self.coco.loadImgs(img_id)[0]
        path = img_info["file_name"]

        image = Image.open(self.img_folder / path).convert("RGB")

        boxes = []
        labels = []
        for ann in anns:
            # skip crowd
            if ann.get("iscrowd", 0):
                continue
            bbox = ann["bbox"]  # [x, y, w, h]
            x1, y1, w, h = bbox
            x2 = x1 + w
            y2 = y1 + h
            boxes.append([x1, y1, x2, y2])
            labels.append(self.cat_id_to_contig[ann["category_id"]])

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
        }
        return image, target

