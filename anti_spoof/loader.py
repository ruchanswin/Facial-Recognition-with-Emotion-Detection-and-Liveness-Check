"""Load antispoof .h5 checkpoints across TensorFlow / Keras versions."""

import tensorflow as tf
from tensorflow.keras.models import load_model

from anti_spoof.model import build_liveness_model

_RENORM_KEYS = ("renorm", "renorm_clipping", "renorm_momentum")


class _CompatibleBatchNormalization(tf.keras.layers.BatchNormalization):
    def __init__(self, *args, **kwargs):
        for key in _RENORM_KEYS:
            kwargs.pop(key, None)
        super().__init__(*args, **kwargs)


class _CompatibleDense(tf.keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop("quantization_config", None)
        super().__init__(*args, **kwargs)


def load_liveness_model(model_path: str) -> tf.keras.Model:
    """
    Load a MobileNetV2 liveness .h5 file.

    Prefer rebuilding the architecture and loading weights only, which avoids
    Keras 3 rejecting older serialized configs (BatchNormalization renorm,
    Dense quantization_config). Fall back to full-model load with shims.
    """
    model = build_liveness_model()
    try:
        model.load_weights(model_path)
        return model
    except (ValueError, OSError):
        pass

    return load_model(
        model_path,
        compile=False,
        custom_objects={
            "BatchNormalization": _CompatibleBatchNormalization,
            "Dense": _CompatibleDense,
        },
    )
