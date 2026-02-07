"""
Default hyperparameters and constants for OSOD-impl.
"""

from dataclasses import dataclass


@dataclass
class TrainConfig:
    device: str = "cuda"
    batch_size: int = 2
    lr: float = 1e-4
    weight_decay: float = 0.0001
    num_iters: int = 1000
    epochs: int = 12
    entropy_thresh: float = 0.85
    semantic_loss_weight: float = 1.0  # Eq. (2)
    decorrelation_weight: float = 0.1  # Eq. (5)
    clip_embed_dim: int = 512          # CLIP ViT-B/32 default
    semantic_temperature: float = 0.07 # temperature for Eq. (2) logits
    object_focus_weight: float = 1.0   # Eq. (7)
    max_samples: int = 0  # 0 = use full dataset


DEFAULT_CONFIG = TrainConfig()
