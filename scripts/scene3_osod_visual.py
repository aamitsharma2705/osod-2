"""
Scene 3: OSOD visualization (our checkpoint) in the same style as scenes 1/2.
Uses our trained model to show known vs unknown predictions on provided images.

Usage:
  poetry run python scripts/scene3_osod_visual.py \
    --image data/demo/zebra.jpeg \
    --checkpoint checkpoints/latest.pth \
    --split voc_coco_t1 \
    --score-thresh 0.3 \
    --unknown-thresh 0.5
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from PIL import Image
from torchvision import transforms

from src.config import DEFAULT_CONFIG
from src.data.splits import prompt_file
from src.models.build import build_model


def load_class_names(split: str):
    p = prompt_file(split)
    if not p.exists():
        raise FileNotFoundError(f"Prompt file not found for split {split}: {p}")
    with p.open() as f:
        return [line.strip() for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser(description="Scene 3: OSOD visualization with our checkpoint")
    ap.add_argument("--image", type=Path, required=True, help="Image path (jpg/png/jpeg)")
    ap.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint path (.pth)")
    ap.add_argument("--split", type=str, default="voc_coco_t1", help="Split for prompts")
    ap.add_argument("--score-thresh", type=float, default=0.3, help="Score threshold")
    ap.add_argument("--unknown-thresh", type=float, default=0.5, help="Unknown threshold on score (low => unknown)")
    args = ap.parse_args()

    device = torch.device(DEFAULT_CONFIG.device if torch.cuda.is_available() else "cpu")
    class_names = load_class_names(args.split)

    ckpt = torch.load(args.checkpoint, map_location=device)
    cls_w = ckpt["model"]["detector.roi_heads.box_predictor.cls_score.weight"]
    num_classes = cls_w.shape[0]

    model = build_model(
        num_classes=num_classes,
        embed_dim=DEFAULT_CONFIG.clip_embed_dim,
        temperature=DEFAULT_CONFIG.semantic_temperature,
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    tfm = transforms.Compose([transforms.ToTensor()])
    img = Image.open(args.image).convert("RGB")
    img_t = tfm(img).to(device)

    with torch.no_grad():
        out = model([img_t])[0]

    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(img_t.cpu().permute(1, 2, 0))
    ax.axis("off")

    if out["scores"].numel() == 0:
        plt.title("Scene 3: OSOD (no detections)")
        plt.show()
        return

    for box, label, score in zip(out["boxes"], out["labels"], out["scores"]):
        if score < args.score_thresh:
            continue
        lbl_id = int(label)
        name = class_names[lbl_id - 1] if 0 < lbl_id <= len(class_names) else f"id{lbl_id}"
        is_unknown = score < args.unknown_thresh
        color = "red" if is_unknown else "green"

        x1, y1, x2, y2 = box.tolist()
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x1,
            y1 - 5,
            f"{'unknown' if is_unknown else name} ({score:.2f})",
            fontsize=10,
            color="black",
            backgroundcolor=color,
        )

    plt.title("Scene 3: OSOD (ours) - known vs unknown")
    plt.show()


if __name__ == "__main__":
    main()
