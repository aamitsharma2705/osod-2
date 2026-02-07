import torch
import torch.nn.functional as F


def entropy_threshold_mask(logits: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Returns a boolean mask where prediction entropy is below threshold.
    """
    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)
    return entropy < threshold

