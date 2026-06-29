import os

import torch
import torch.nn as nn
from torchvision import models

from emotion.dataset import EMOTION_CLASSES

# Default checkpoint filenames per training dataset
EMOTION_CHECKPOINT_BY_DATASET = {
    "fer2013": "fer2013_emotion_model.pth",
    "fane_split": "fane_emotion_model.pth",
}


def emotion_checkpoint_name(dataset: str) -> str:
    return EMOTION_CHECKPOINT_BY_DATASET.get(dataset, f"{dataset}_emotion_model.pth")


def default_emotion_model_path(dataset: str, save_dir: str = "saved_models") -> str:
    return os.path.join(save_dir, emotion_checkpoint_name(dataset))


def default_confusion_matrix_path(dataset: str, save_dir: str = "saved_models") -> str:
    stem = emotion_checkpoint_name(dataset).removesuffix(".pth")
    return os.path.join(save_dir, f"{stem}_confusion_matrix.png")


def default_evaluation_csv_path(dataset: str, save_dir: str = "saved_models") -> str:
    stem = emotion_checkpoint_name(dataset).removesuffix(".pth")
    return os.path.join(save_dir, f"{stem}_evaluation_results.csv")


class EmotionModel(nn.Module):
    """ResNet-34 classifier fine-tuned for emotion classes (e.g. 7 FER2013 or 9 FANE)."""

    def __init__(self, num_classes: int = len(EMOTION_CLASSES), pretrained: bool = True):
        super().__init__()
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet34(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def load_emotion_model(model_path: str, device: torch.device) -> tuple[EmotionModel, list[str]]:
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint.get("class_names", list(EMOTION_CLASSES))
    model = EmotionModel(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_names
