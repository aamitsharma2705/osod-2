"""
Quick demo script: run a saved checkpoint on example images and visualize
known vs unknown predictions. Intended for presentation (no full training).

Usage:
  poetry run python scripts/demo_visualize.py \
    --images data/demo/zebra.jpeg data/demo/cat_dog.jpeg \
    --checkpoint checkpoints/latest.pth \
    --split voc_coco_t1 \
    --out-dir demo_out \
    --score-thresh 0.5 \
    --unknown-thresh 0.5
"""

import argparse
from pathlib import Path
from typing import List

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import draw_bounding_boxes

from src.config import DEFAULT_CONFIG
from src.data.splits import prompt_file
from src.models.build import build_model


def load_class_names(split: str) -> List[str]:
    p = prompt_file(split)
    if not p.exists():
        raise FileNotFoundError(f"Prompt file not found for split {split}: {p}")
    with p.open() as f:
        return [line.strip() for line in f if line.strip()]


def annotate_image(
    img: torch.Tensor,
    boxes,
    labels,
    scores,
    class_names,
    unknown_mask,
    out_path: Path,
    show: bool = False,
):
    # Build label strings
    H, W = img.shape[1], img.shape[2]
    boxes = boxes.clone()
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, W - 1)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, H - 1)
    boxes = boxes.round().long()

    text = []
    for lbl, sc, unk in zip(labels, scores, unknown_mask):
        if unk:
            text.append(f"unknown {sc*100:.0f}%")
        else:
            name = class_names[lbl - 1] if 0 < lbl <= len(class_names) else f"id{lbl}"
            text.append(f"{name} {sc*100:.0f}%")
    colors = ["red" if unk else "green" for unk in unknown_mask]
    vis = draw_bounding_boxes(
        (img * 255).byte(), boxes, labels=text, colors=colors, width=3, font_size=16
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(vis.permute(1, 2, 0).cpu().numpy()).save(out_path)
    print(f"Saved {out_path}")
    if show:
        Image.open(out_path).show()


def run_demo(args):
    device = torch.device(DEFAULT_CONFIG.device if torch.cuda.is_available() else "cpu")

    class_names = load_class_names(args.split)
    ckpt = torch.load(args.checkpoint, map_location=device)
    # Infer num_classes from checkpoint head to avoid shape mismatch
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
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in args.images:
        img = Image.open(img_path).convert("RGB")
        img_t = tfm(img).to(device)
        with torch.no_grad():
            outputs = model([img_t])[0]

        out_path = Path(args.out_dir) / (Path(img_path).stem + "_demo.jpg")

        # If no detections at all, save the raw image
        if outputs["scores"].numel() == 0:
            Image.fromarray((img_t.cpu() * 255).byte().permute(1, 2, 0).numpy()).save(out_path)
            print(f"No detections for {img_path}; saved raw image to {out_path}")
            if args.show:
                Image.open(out_path).show()
            continue

        # Apply score threshold; if nothing survives, take top-k and mark all unknown
        scores_all = outputs["scores"]
        keep = scores_all >= args.score_thresh
        if keep.sum() == 0:
            topk = min(5, scores_all.numel())
            scores = scores_all[:topk].cpu()
            boxes = outputs["boxes"][:topk].cpu()
            labels = outputs["labels"][:topk].cpu()
            unknown_mask = torch.ones_like(scores, dtype=torch.bool)
        else:
            boxes = outputs["boxes"][keep].cpu()
            labels = outputs["labels"][keep].cpu()
            scores = scores_all[keep].cpu()
            unknown_mask = scores < args.unknown_thresh

        annotate_image(
            img_t.cpu(),
            boxes,
            labels,
            scores,
            class_names,
            unknown_mask,
            out_path,
            show=args.show,
        )


def main():
    ap = argparse.ArgumentParser(description="OSOD demo visualize")
    ap.add_argument("--images", nargs="+", required=True, help="List of image paths")
    ap.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint (.pth)")
    ap.add_argument("--split", type=str, default="voc_coco_t1", help="Split name (for prompts)")
    ap.add_argument("--out-dir", type=Path, default=Path("demo_out"), help="Output directory for visualizations")
    ap.add_argument("--score-thresh", type=float, default=0.5, help="Detection score threshold")
    ap.add_argument("--unknown-thresh", type=float, default=0.5, help="Score threshold to flag unknown")
    ap.add_argument("--show", action="store_true", help="Open saved images after writing")
    args = ap.parse_args()

    run_demo(args)


if __name__ == "__main__":
    main()
