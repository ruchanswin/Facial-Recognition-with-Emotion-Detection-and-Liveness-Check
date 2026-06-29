import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class FaceModel(nn.Module):
    """
    Shared ResNet-18 backbone with two interchangeable heads:

      - 'classifier' mode: backbone → FC(512, num_classes)
        Used during classification training (Approach A).

      - 'embedding' mode: backbone → FC(512, embedding_dim) → L2-normalise
        Used for verification at inference and during triplet training (Approach B).

    Switch modes by calling model.set_mode('classifier') or model.set_mode('embedding').
    """

    def __init__(self, num_classes: int = 1018, embedding_dim: int = 128):
        super().__init__()

        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = backbone.fc.in_features   # 512 for ResNet-18
        backbone.fc = nn.Identity()             # strip original FC
        self.backbone = backbone

        self.classifier_head = nn.Linear(in_features, num_classes)
        self.embedding_head = nn.Linear(in_features, embedding_dim)

        self._mode = "classifier"

    def set_mode(self, mode: str):
        assert mode in ("classifier", "embedding"), f"Unknown mode: {mode}"
        self._mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)                     # (B, 512)

        if self._mode == "classifier":
            return self.classifier_head(features)       # (B, num_classes)

        emb = self.embedding_head(features)             # (B, embedding_dim)
        return F.normalize(emb, p=2, dim=1)             # L2-normalised

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Always returns L2-normalised embeddings regardless of current mode."""
        features = self.backbone(x)
        return F.normalize(self.embedding_head(features), p=2, dim=1)

    def get_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised backbone features (512-dim). Use this for Approach A identification."""
        features = self.backbone(x)
        return F.normalize(features, p=2, dim=1)
