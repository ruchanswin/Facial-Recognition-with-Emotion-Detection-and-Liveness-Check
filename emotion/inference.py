import torch

from emotion.model import load_emotion_model


EMOTION_ICONS = {
    "angry": ":(",
    "disgust": ":/",
    "fear": ":o",
    "happy": ":)",
    "sad": ":'(",
    "surprise": ":O",
    "neutral": ":|",
    "confused": ":-?",
    "shy": ":-$",
}


class EmotionPredictor:
    """Small inference wrapper used by the webcam app."""

    def __init__(self, model_path: str, device: torch.device | None = None):
        self.device = device or self._select_device()
        self.model, self.class_names = load_emotion_model(model_path, self.device)

    @staticmethod
    def _select_device() -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @torch.no_grad()
    def predict_tensor(self, face_tensor: torch.Tensor) -> tuple[str, float, str]:
        # face_tensor is already ImageNet-normalized; just resize to the training resolution
        batch = torch.nn.functional.interpolate(
            face_tensor.unsqueeze(0), size=(64, 64), mode="bilinear", align_corners=False
        ).to(self.device)
        logits = self.model(batch)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        confidence, index = probs.max(dim=0)
        label = self.class_names[index.item()]
        icon = EMOTION_ICONS.get(label.lower(), "")
        return label, confidence.item(), icon
