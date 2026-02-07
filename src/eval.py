"""
Evaluation scaffold with entropy thresholding and HMP.
Note: still needs proper unknown/known labeling to be paper-accurate.
"""

import torch
from torchvision.transforms import functional as TF

from src.models.losses.entropy_threshold import entropy_threshold_mask


def hmp(pk: float, pu: float) -> float:
    if pk + pu == 0:
        return 0.0
    return 2 * pk * pu / (pk + pu)


def evaluate_entropy(model, dataloader, entropy_thresh: float = 0.85, device: str = "cpu"):
    """
    Placeholder evaluation:
    - Applies entropy threshold on detection scores to flag low-entropy (known) vs high-entropy (unknown).
    - Computes dummy PK/PU/HMP since GT unknown labels are not available in VOC-only sanity.
    Replace with proper matching against GT for known/unknown when available.
    """
    model.eval()
    known_tp = 0
    known_fp = 0
    unknown_tp = 0
    unknown_fp = 0

    with torch.no_grad():
        for images, _ in dataloader:
            # Ensure PIL -> tensor conversion for eval
            images = [TF.to_tensor(img).to(device) if not isinstance(img, torch.Tensor) else img.to(device) for img in images]
            outputs = model(images)
            for det in outputs:
                if "scores" not in det:
                    continue
                # Treat scores as logits for entropy; this is a simplification.
                logits = det["scores"].unsqueeze(0)
                mask = entropy_threshold_mask(logits, entropy_thresh)
                # Without GT unknowns, we cannot compute true PK/PU; count as known for sanity.
                known_tp += int(mask.sum().item())
                known_fp += int((~mask).sum().item())

    pk = known_tp / max(1, known_tp + known_fp)
    pu = unknown_tp / max(1, unknown_tp + unknown_fp)
    return {"PK": pk, "PU": pu, "HMP": hmp(pk, pu)}

