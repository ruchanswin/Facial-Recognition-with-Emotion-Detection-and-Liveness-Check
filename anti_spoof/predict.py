# import cv2
# import numpy as np

# from anti_spoof.loader import load_liveness_model


# class LivenessDetector:
#     def __init__(self, model_path):
#         self.model = load_liveness_model(model_path)

#     def is_real(self, face_image, threshold: float = 0.5):
#         """
#         Returns (is_live, prob_real).

#         Model output is sigmoid P(spoof), label 0=live / 1=spoof in training data.
#         prob_real = 1 - P(spoof).
#         """
#         face = cv2.resize(face_image, (224, 224), interpolation=cv2.INTER_LINEAR)
#         face = face.astype("float32") / 255.0
#         face = np.expand_dims(face, axis=0)

#         p_spoof = float(self.model.predict(face, verbose=0)[0][0])
#         prob_real = 1.0 - p_spoof
#         return prob_real >= threshold, prob_real

"""
predict.py — public interface for the anti-spoof module.
 
LivenessDetector is now the full behavioural-fusion detector from dominic_innovation/liveness.py
pipeline.py imports from here, so no changes are needed in pipeline.py.
 
The is_real() signature is backwards-compatible:
    is_real(face_image)                    → CNN-only  (same as before)
    is_real(face_image, full_frame=frame)  → CNN + behavioural fusion
"""


from dominic_innovation.liveness import LivenessDetector