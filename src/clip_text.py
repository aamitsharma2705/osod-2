"""
Frozen CLIP text encoder utilities (text branch only).
"""

from pathlib import Path
from typing import List, Tuple

import torch
import open_clip


class CLIPTextEncoder:
    """
    Loads a frozen CLIP text encoder and returns normalized text embeddings.
    """

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k"):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_prompts(self, prompts: List[str]) -> torch.Tensor:
        """
        Returns L2-normalized text embeddings for a list of prompts.
        """
        tokens = self.tokenizer(prompts)
        text_features = self.model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features


def load_prompts(path: str | Path) -> List[str]:
    with Path(path).open() as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    return lines


def build_class_embeddings(prompt_path: str | Path) -> Tuple[torch.Tensor, List[str]]:
    prompts = load_prompts(prompt_path)
    encoder = CLIPTextEncoder()
    embeddings = encoder.encode_prompts(prompts)
    return embeddings, prompts

