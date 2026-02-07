import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticClusterHead(nn.Module):
    """
    Projects ROI-pooled features into CLIP text-embedding space (1024-D),
    produces cosine logits against class prototypes, and exposes intermediate
    features for decorrelation regularization.
    """

    def __init__(self, in_dim: int, embed_dim: int = 512, temperature: float = 0.07):
        super().__init__()
        self.proj = nn.Linear(in_dim, embed_dim)
        self.temperature = temperature

    def forward(self, roi_feats: torch.Tensor, class_embeds: torch.Tensor):
        """
        roi_feats: (N, C)
        class_embeds: (K, D) normalized CLIP text embeddings
        """
        proj = self.proj(roi_feats)  # (N, D_out)
        proj = F.normalize(proj, dim=-1)
        class_embeds = F.normalize(class_embeds, dim=-1)

        logits = (proj @ class_embeds.t()) / self.temperature
        return logits, proj

