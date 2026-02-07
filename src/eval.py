"""
Evaluation scaffold with entropy thresholding and HMP.
GT matching included for known precision; unknown remains placeholder without GT unknown labels.
"""

import torch
from torchvision.transforms import functional as TF
from torchvision.ops import box_iou

from src.models.losses.entropy_threshold import entropy_threshold_mask


def hmp(pk: float, pu: float) -> float:
    if pk + pu == 0:
        return 0.0
    return 2 * pk * pu / (pk + pu)


def evaluate_entropy(model, dataloader, entropy_thresh: float = 0.85, device: str = "cpu"):
    """
    Eval with entropy threshold:
    - Low entropy => known; high entropy => unknown (placeholder).
    - Computes PK via IoU + label match. PU uses unknown GT boxes if provided (key: unknown_boxes).
    """
    model.eval()
    known_tp = 0
    known_fp = 0
    unknown_tp = 0
    unknown_fp = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = [TF.to_tensor(img).to(device) if not isinstance(img, torch.Tensor) else img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)
            for det, tgt in zip(outputs, targets):
                if "scores" not in det or det["boxes"].numel() == 0:
                    continue
                logits = det["scores"].unsqueeze(0)
                mask_known = entropy_threshold_mask(logits, entropy_thresh).squeeze(0)

                pred_boxes = det["boxes"][mask_known]
                pred_labels = det["labels"][mask_known]
                gt_boxes = tgt["boxes"]
                gt_labels = tgt["labels"]

                if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
                    known_fp += int(mask_known.sum().item())
                    continue

                ious = box_iou(pred_boxes, gt_boxes)
                # greedy match
                matched_gt = ious.argmax(dim=1)
                max_iou = ious.max(dim=1).values
                iou_mask = max_iou >= 0.5
                label_mask = pred_labels == gt_labels[matched_gt]
                tp_mask = iou_mask & label_mask

                known_tp += int(tp_mask.sum().item())
                known_fp += int(mask_known.sum().item() - tp_mask.sum().item())

                # unknown accounting: use unknown_boxes if present
                unknown_mask = ~mask_known
                if unknown_mask.any():
                    if "unknown_boxes" in tgt and tgt["unknown_boxes"].numel() > 0:
                        unk_boxes = tgt["unknown_boxes"]
                        pred_unk_boxes = det["boxes"][unknown_mask]
                        ious_unk = box_iou(pred_unk_boxes, unk_boxes)
                        max_iou_unk = ious_unk.max(dim=1).values if ious_unk.numel() else torch.tensor([])
                        if max_iou_unk.numel() > 0:
                            tp_unk = (max_iou_unk >= 0.5).sum().item()
                            unknown_tp += tp_unk
                            unknown_fp += int(unknown_mask.sum().item() - tp_unk)
                        else:
                            unknown_fp += int(unknown_mask.sum().item())
                    else:
                        unknown_fp += int(unknown_mask.sum().item())

    pk = known_tp / max(1, known_tp + known_fp)
    pu = unknown_tp / max(1, unknown_tp + unknown_fp)
    return {"PK": pk, "PU": pu, "HMP": hmp(pk, pu)}

