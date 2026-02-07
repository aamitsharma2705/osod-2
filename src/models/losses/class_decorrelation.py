import torch


def class_decorrelation_loss(features: torch.Tensor) -> torch.Tensor:
    """
    Class decorrelation (Eq. 5 intent):
    - Normalize features per dimension
    - Compute correlation matrix
    - Penalize off-diagonal terms
    """
    if features.numel() == 0 or features.shape[0] < 2:
        return torch.tensor(0.0, device=features.device)

    feats = features - features.mean(dim=0, keepdim=True)
    std = feats.std(dim=0, keepdim=True)
    std = torch.clamp(std, min=1e-3)  # avoid div-by-zero / tiny std
    feats = feats / std

    corr = (feats.T @ feats) / max(1, feats.shape[0])
    off_diag = corr - torch.diag(torch.diag(corr))
    return (off_diag**2).mean()

