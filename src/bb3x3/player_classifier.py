"""12-way player classifier (squad colour + jersey number + whatever else identifies
the person) over padded detection crops. Trained only on official training crops
(scripts/train_player_classifier.py), from an ImageNet-pretrained timm backbone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from bb3x3.identity import SQUADS

PLAYERS: list[str] = sorted(f"{squad}_{n}" for squad, numbers in SQUADS.items() for n in numbers)
INPUT_HW = (256, 128)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def build_model(backbone: str, pretrained: bool = True) -> nn.Module:
    import timm

    return timm.create_model(backbone, pretrained=pretrained, num_classes=len(PLAYERS))


def to_tensor(crops_bgr: np.ndarray) -> torch.Tensor:
    """(N, h, w, 3) uint8 BGR crops -> normalised (N, 3, H, W) float tensor at INPUT_HW."""
    x = torch.from_numpy(crops_bgr[..., ::-1].copy()).permute(0, 3, 1, 2).float() / 255.0
    x = nn.functional.interpolate(x, size=INPUT_HW, mode="bilinear", align_corners=False)
    return (x - MEAN) / STD


class PlayerClassifier:
    def __init__(self, checkpoint: str | Path, device: str = "cuda"):
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model = build_model(state["backbone"], pretrained=False)
        self.model.load_state_dict(state["model"])
        self.model.eval().to(device)
        self.device = device

    @torch.no_grad()
    def log_probs(self, crops_bgr: np.ndarray, batch: int = 256) -> np.ndarray:
        out = []
        for i in range(0, len(crops_bgr), batch):
            x = to_tensor(crops_bgr[i : i + batch]).to(self.device)
            with torch.autocast("cuda", dtype=torch.float16):
                out.append(torch.log_softmax(self.model(x).float(), dim=1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(PLAYERS)), np.float32)
