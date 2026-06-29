"""
Individual Innovation Feature -- Quality-Aware Multi-Frame Face Registration
Author: Anh Phan  |  COS30082

Problem with standard registration:
    The naive approach captures N frames and averages their embeddings with
    equal weight. A blurry or poorly lit frame contributes as much as a
    sharp, well-lit one -- degrading the registered embedding and causing
    false rejections at verification time.

This module:
    1. Scores each captured frame on two quality axes:
         - Sharpness  : Laplacian variance (high variance = sharp edges)
         - Brightness : Distance from ideal brightness (penalises dark/overexposed)
    2. Discards frames below a minimum quality threshold.
    3. Aggregates remaining embeddings with softmax-weighted averaging --
       sharper, better-lit frames contribute proportionally more.

Comparison:
    Standard:       embedding = mean(e1, e2, ..., eN)          equal weight
    Quality-aware:  embedding = sum(softmax(q_i * T) * e_i)    quality weight

Usage:
    from anh_innovation import QualityAwareRegistration
    reg = QualityAwareRegistration(model, device)
    embedding, scores = reg.register("Alice", face_crops_bgr)
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms


# ---------------------------------------------------------------------------
# Quality scorer
# ---------------------------------------------------------------------------

class FaceQualityScorer:
    """
    Scores a BGR face crop on sharpness + brightness in [0, 1].

    Sharpness  : Laplacian variance, capped at SHARPNESS_CAP -> normalised to [0,1].
    Brightness : 1 - |mean_pixel - 128| / 128  (penalises dark & overexposed).
    Combined   : 0.7 * sharpness + 0.3 * brightness
    """

    SHARPNESS_CAP = 500.0

    def sharpness_score(self, face_bgr: np.ndarray) -> float:
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        return min(float(cv2.Laplacian(gray, cv2.CV_64F).var()) / self.SHARPNESS_CAP, 1.0)

    def brightness_score(self, face_bgr: np.ndarray) -> float:
        return 1.0 - abs(float(face_bgr.mean()) - 128.0) / 128.0

    def score(self, face_bgr: np.ndarray) -> float:
        return 0.7 * self.sharpness_score(face_bgr) + 0.3 * self.brightness_score(face_bgr)

    def score_batch(self, faces: list) -> np.ndarray:
        return np.array([self.score(f) for f in faces], dtype=np.float32)


# ---------------------------------------------------------------------------
# Quality-aware registration
# ---------------------------------------------------------------------------

class QualityAwareRegistration:
    """
    Quality-weighted multi-frame face registration and identification.

    Parameters
    ----------
    model      : FaceModel (ResNet-18) or ConvNeXt backbone
    device     : torch.device
    arch       : 'resnet18' uses model.backbone(); 'convnext_tiny' calls model() directly
    min_quality: frames below this score are discarded (default 0.15)
    softmax_temp: sharpens softmax weight distribution (default 5.0)
    """

    def __init__(
        self,
        model,
        device: torch.device,
        arch: str = "resnet18",
        min_quality: float = 0.15,
        softmax_temp: float = 5.0,
    ):
        self.model        = model
        self.device       = device
        self.arch         = arch
        self.min_quality  = min_quality
        self.softmax_temp = softmax_temp
        self.scorer       = FaceQualityScorer()
        self.registered: dict[str, np.ndarray] = {}

        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(112),
            transforms.CenterCrop(112),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def _embed(self, face_crops_bgr: list) -> torch.Tensor:
        tensors = [self._transform(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in face_crops_bgr]
        batch = torch.stack(tensors).to(self.device)
        self.model.eval()
        features = self.model(batch) if self.arch == "convnext_tiny" else self.model.backbone(batch)
        return F.normalize(features, p=2, dim=1).cpu()

    def register(self, name: str, face_crops_bgr: list, verbose: bool = True) -> tuple:
        """
        Register identity from N BGR face crops using quality-weighted averaging.
        Returns (embedding np.ndarray, quality_scores np.ndarray).
        """
        if not face_crops_bgr:
            raise ValueError("No face crops provided.")

        quality_scores = self.scorer.score_batch(face_crops_bgr)

        good_mask = quality_scores >= self.min_quality
        if not good_mask.any():
            good_mask = np.ones(len(face_crops_bgr), dtype=bool)
            if verbose:
                print("  Warning: all frames below threshold -- keeping all.")

        good_crops  = [c for c, m in zip(face_crops_bgr, good_mask) if m]
        good_scores = quality_scores[good_mask]

        if verbose:
            print(f"\n  Registering '{name}'  ({len(face_crops_bgr)} frames, "
                  f"{len(good_crops)} kept after quality filter)")
            print(f"  {'Frame':>5}  {'Quality':>8}  {'Sharp':>7}  {'Bright':>7}  Status")
            good_idx = 0
            weights_display = torch.softmax(
                torch.tensor(good_scores) * self.softmax_temp, dim=0
            ).numpy()
            for i, (score, crop) in enumerate(zip(quality_scores, face_crops_bgr)):
                kept = score >= self.min_quality
                s = self.scorer.sharpness_score(crop)
                b = self.scorer.brightness_score(crop)
                w = f"{weights_display[good_idx]*100:.1f}%" if kept else "--"
                if kept:
                    good_idx += 1
                print(f"  {i+1:>5}  {score:>8.3f}  {s:>7.3f}  {b:>7.3f}  "
                      f"{'keep ' + w if kept else 'discard'}")

        embeddings = self._embed(good_crops)

        weights = torch.softmax(
            torch.tensor(good_scores) * self.softmax_temp, dim=0
        ).unsqueeze(1)

        final = F.normalize(
            (embeddings * weights).sum(dim=0, keepdim=True), p=2, dim=1
        ).squeeze(0)

        self.registered[name] = final.numpy()
        return final.numpy(), quality_scores

    def register_equal_weight(self, name: str, face_crops_bgr: list) -> np.ndarray:
        """Baseline: standard equal-weight average (no quality filtering)."""
        embeddings = self._embed(face_crops_bgr)
        final = F.normalize(embeddings.mean(dim=0, keepdim=True), p=2, dim=1).squeeze(0)
        return final.numpy()

    def identify(self, face_crop_bgr: np.ndarray, threshold: float = 0.45) -> tuple:
        if not self.registered:
            return "Unknown", 0.0
        live_emb = self._embed([face_crop_bgr])[0].numpy()
        best_name, best_score = "Unknown", -1.0
        for name, ref in self.registered.items():
            sim = float(np.dot(live_emb, ref))
            if sim > best_score:
                best_score = sim
                best_name  = name if sim >= threshold else "Unknown"
        return best_name, best_score

    def save(self, path: str = "registered_faces.npz") -> None:
        np.savez(path, **self.registered)
        print(f"Saved {len(self.registered)} identities -> {path}")

    def load(self, path: str = "registered_faces.npz") -> None:
        if not os.path.exists(path):
            return
        data = np.load(path)
        self.registered = {k: data[k] for k in data.files}
        print(f"Loaded {len(self.registered)} identities from {path}")

    def list_registered(self) -> list:
        return list(self.registered.keys())
