import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


class VOCDataset(Dataset):
    """
    Pascal VOC dataset loader (2007/2012).
    """

    def __init__(self, root: str | Path, year: str, image_set: str, max_samples: int = 0):
        self.root = Path(root).expanduser()
        self.year = year
        self.image_set = image_set
        self.max_samples = max_samples

        voc_root = (
            self.root / "VOCdevkit" / f"VOC{year}"
            if (self.root / "VOCdevkit").exists()
            else self.root / f"VOC{year}"
        )

        # Some datasets only include class-specific split files; allow fallback.
        split_file = voc_root / "ImageSets" / "Main" / f"{image_set}.txt"
        if not split_file.exists():
            alt = voc_root / "ImageSets" / "Main" / f"{image_set}_ids.txt"
            if alt.exists():
                split_file = alt
            else:
                raise FileNotFoundError(f"Missing split file: {split_file}")

        with split_file.open() as f:
            ids = [x.strip() for x in f.readlines()]

        if self.max_samples and self.max_samples > 0:
            ids = ids[: self.max_samples]

        self.ids = ids
        self.img_dir = voc_root / "JPEGImages"
        self.ann_dir = voc_root / "Annotations"

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        img_id = self.ids[idx]

        img_path = self.img_dir / f"{img_id}.jpg"
        ann_path = self.ann_dir / f"{img_id}.xml"

        image = Image.open(img_path).convert("RGB")
        boxes, labels = self._parse_annotation(ann_path)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
        }

        return image, target

    def _parse_annotation(self, path: Path):
        tree = ET.parse(path)
        root = tree.getroot()

        boxes: list[list[float]] = []
        labels: list[int] = []

        for obj in root.iter("object"):
            cls = obj.find("name").text
            if cls not in VOC_CLASSES:
                continue

            cls_id = VOC_CLASSES.index(cls)

            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(cls_id)

        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

