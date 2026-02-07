import torch
import torch.nn.functional as F


def semantic_alignment_loss(logits, labels):
    """
    Semantic alignment (Eq. 2): cross-entropy on cosine/temperature logits.
    Logits should already be computed via normalized ROI projections vs class embeds.
    """
    return F.cross_entropy(logits, labels)

