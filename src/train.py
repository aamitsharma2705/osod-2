import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F

from src.clip_text import build_class_embeddings
from src.config import DEFAULT_CONFIG
from src.data.voc import VOCDataset, VOC_CLASSES
from src.data.splits import prompt_file
from src.data.composite import build_voc_coco_dataset
from src.models.build import build_model
from src.models.losses.semantic_alignment import semantic_alignment_loss
from src.models.losses.class_decorrelation import class_decorrelation_loss
from src.eval import evaluate_entropy


def collate_fn(batch):
    return tuple(zip(*batch))


def parse_args():
    parser = argparse.ArgumentParser(description="Train OSOD-impl scaffold")
    parser.add_argument("--data-root", type=str, default="data/VOC")
    parser.add_argument("--split", type=str, default="voc_sanity")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG.batch_size)
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG.lr)
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG.epochs)
    parser.add_argument("--num-iters", type=int, default=0, help="Override for short runs")
    parser.add_argument("--max-samples", type=int, default=DEFAULT_CONFIG.max_samples)
    parser.add_argument("--entropy-thresh", type=float, default=DEFAULT_CONFIG.entropy_thresh)
    parser.add_argument("--eval-every", type=int, default=0, help="Run eval every N epochs; 0 = skip")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"), help="Directory to save checkpoints")
    parser.add_argument("--resume", type=Path, default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()


def build_dataloader(data_root: str, max_samples: int, batch_size: int, split: str):
    root = Path(data_root)

    if split in ("voc_coco_t1", "voc_coco_t2"):
        dataset = build_voc_coco_dataset(root, split=split, max_samples=max_samples)
    else:
        # Default: VOC 2007+2012 trainval as a single dataset
        if (root / "VOC2007").exists() and (root / "VOC2012").exists():
            voc_root = root
        elif (root / "VOC" / "VOC2007").exists():
            voc_root = root / "VOC"
        else:
            voc_root = root
        ds2007 = VOCDataset(voc_root, year="2007", image_set="trainval", max_samples=max_samples)
        ds2012 = VOCDataset(voc_root, year="2012", image_set="trainval", max_samples=max_samples)
        dataset = torch.utils.data.ConcatDataset([ds2007, ds2012])

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    return loader


def main():
    args = parse_args()
    device = torch.device(DEFAULT_CONFIG.device if torch.cuda.is_available() else "cpu")

    num_classes = len(VOC_CLASSES) + 1  # background
    model = build_model(
        num_classes=num_classes,
        embed_dim=DEFAULT_CONFIG.clip_embed_dim,
    )
    model.to(device)
    model.train()

    # Load class embeddings if prompt file exists
    class_embeds = None
    prompt_path = prompt_file(args.split)
    if prompt_path.exists():
        class_embeds, class_names = build_class_embeddings(prompt_path)
        print(f"Loaded {len(class_names)} prompts from {prompt_path}")
        class_embeds = class_embeds.to(device)
    else:
        print(f"Prompt file not found for split {args.split}; skipping CLIP embeddings.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    start_epoch = 0
    iteration = 0
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if args.resume and args.resume.exists():
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state.get("epoch", 0)
        iteration = state.get("iteration", 0)
        print(f"Resumed from {args.resume} at epoch {start_epoch}, iteration {iteration}")
    dataloader = build_dataloader(args.data_root, args.max_samples, args.batch_size, args.split)

    max_iters = args.num_iters if args.num_iters > 0 else args.epochs * len(dataloader)
    iteration = 0

    for epoch in range(start_epoch, args.epochs):
        for images, targets in dataloader:
            images = [F.to_tensor(img).to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            losses, sem = model(images, targets)
            total_loss = sum(loss for loss in losses.values())

            # Semantic alignment/decorrelation if embeddings available
            if class_embeds is not None and sem is not None:
                roi_feats = sem["roi_feats"]
                roi_labels = sem["labels"]
                # Filter out background (label 0); shift to 0-based for classes
                fg_mask = roi_labels > 0
                if fg_mask.any():
                    roi_feats_fg = roi_feats[fg_mask]
                    roi_labels_fg = roi_labels[fg_mask] - 1  # background -> -1 removed
                    # Project ROI feats into CLIP space and get logits
                    logits, proj = model.semantic_head(roi_feats_fg, class_embeds)
                    # Semantic alignment (Eq. 2)
                    sem_loss = semantic_alignment_loss(logits, roi_labels_fg)
                    # De-correlation (Eq. 5) on projected feats
                    decor_loss = class_decorrelation_loss(proj)
                    total_loss = (
                        total_loss
                        + DEFAULT_CONFIG.semantic_loss_weight * sem_loss
                        + DEFAULT_CONFIG.decorrelation_weight * decor_loss
                    )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            iteration += 1
            if iteration % 10 == 0 or iteration <= 5:
                print(f"Iter {iteration} | Total Loss: {total_loss.item():.4f}")

            if iteration >= max_iters:
                break
        # Optional eval at epoch end
        if args.eval_every and (epoch + 1) % args.eval_every == 0:
            with torch.no_grad():
                metrics = evaluate_entropy(
                    model, dataloader, entropy_thresh=args.entropy_thresh, device=device
                )
            print(
                f"Epoch {epoch+1} eval | PK: {metrics['PK']:.4f} | "
                f"PU: {metrics['PU']:.4f} | HMP: {metrics['HMP']:.4f}"
            )

        # Save checkpoint
        ckpt_path = args.checkpoint_dir / "latest.pth"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "iteration": iteration,
                "config": DEFAULT_CONFIG,
            },
            ckpt_path,
        )

        # Stop early if max_iters reached after saving
        if iteration >= max_iters:
            break

    print("Training complete.")


if __name__ == "__main__":
    main()

