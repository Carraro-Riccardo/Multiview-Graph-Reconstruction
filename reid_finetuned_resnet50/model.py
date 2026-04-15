"""Fine-tunable ResNet50 with projection head for ReID.

Architecture:
  ResNet50 backbone (pretrained, layer4 + head unfrozen)
  → Linear(2048, 512) → BatchNorm1d → ReLU → Linear(512, 128)
  → L2 normalize → 128-dim embedding

Usage:
    from reid_finetuned_resnet50.model import ReIDResNet50
    model = ReIDResNet50()
    emb = model(tensor)          # (B, 128) L2-normalized
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as M
import torchvision.transforms as T


EMBED_DIM = 128


class ReIDResNet50(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM, pretrained: bool = True):
        super().__init__()
        weights = M.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = M.resnet50(weights=weights)

        # Remove classification head — keep up to avgpool
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # → (B, 2048, 1, 1)

        # Freeze everything except layer4
        for name, param in self.backbone.named_parameters():
            if not name.startswith("6."):   # "6" = layer4 in Sequential
                param.requires_grad_(False)

        # Projection head
        self.head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x).flatten(1)   # (B, 2048)
        emb  = self.head(feat)               # (B, embed_dim)
        return nn.functional.normalize(emb, dim=1)

    @staticmethod
    def get_transform() -> T.Compose:
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
