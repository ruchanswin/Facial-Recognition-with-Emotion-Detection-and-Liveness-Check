# ◈ FaceAttend

**AI-Powered Face Recognition & Attendance System**
_COS30082 Applied Machine Learning - Unified Gradio Console_

---

## Overview

FaceAttend is an end-to-end, real-time attendance management system that combines three deep learning disciplines - face recognition, emotion detection, and liveness/anti-spoof detection - into a single unified Gradio web interface. The system is structured as a group project with individual innovation contributions from each team member.

**Key capabilities:**

- Real-time face identification via live webcam stream
- Three distinct face recognition approaches compared and benchmarked
- Emotion detection with temporal smoothing (`justin_innovation`) to reduce frame-to-frame jitter
- Anti-spoofing (liveness) detection fusing a CNN model with behavioural signals
- Automatic SQLite attendance logging with analytics dashboard
- Quality-aware face registration that filters blurry or poorly-lit enrolment frames
- MLOps drift monitoring with statistical alerting

---

## Architecture

```
FaceAttend/
├── app.py                      # ASGI entry point for Vercel and other ASGI hosts
├── gradio_app.py               # Unified single-page Gradio UI (local entry point)
│
├── core/
│   ├── pipeline.py             # Main inference pipeline: detect → liveness → ID → emotion
│   └── registry.py             # Face enrolment & identity lookup (pickle store)
│
├── face_recognition/
│   ├── model.py                # FaceModel: shared ResNet-18 backbone, classifier/embedding heads
│   ├── dataset.py              # Train/val DataLoader helpers
│   ├── train_classification.py # Approach A - 4000-class softmax classifier
│   ├── train_triplet.py        # Approach B - semi-hard triplet loss with P×K sampling
│   ├── train_arcface.py        # Approach C - ArcFace angular margin loss (best AUC)
│   ├── evaluate.py             # Per-model AUC evaluation + ROC curve
│   └── evaluate_all.py         # Comparison ROC across all three approaches
│
├── emotion/
│   ├── model.py                # EmotionModel: ResNet-34 fine-tuned for 7 FER2013 classes
│   ├── dataset.py              # FER2013 + FANE dataset loaders with stratified split
│   ├── train.py                # Training loop with class-imbalance strategies
│   ├── evaluate.py             # Confusion matrix + per-class metrics
│   ├── inference.py            # EmotionPredictor inference wrapper
│   └── prepare_fane_splits.py  # Splits raw FANE dataset into train/val/test
│
├── anti_spoof/
│   ├── model.py                # MobileNetV2 binary classifier (live vs spoof)
│   ├── dataset.py              # HuggingFace AxonData streaming dataset pipeline
│   ├── train.py                # Training script → antispoof_v3.h5
│   ├── predict.py              # Re-exports LivenessDetector from dominic_innovation
│   └── loader.py               # Cross-Keras-version checkpoint loader with shims
│
├── dominic_innovation/
│   ├── liveness.py             # LivenessDetector: CNN + multi-factor behavioural fusion (individual innovation)
│   └── __init__.py
│
├── anh_innovation/
│   ├── quality_registration.py # Quality-aware multi-frame registration (individual innovation)
│   ├── benchmark.py            # Contamination benchmark: QW vs equal-weight
│   ├── demo.py                 # 3-part demo: synthetic scoring, real data, contamination
│   └── __init__.py
│
├── nathan_innovation/
│   ├── db.py                   # SQLite attendance store (log, query, init)
│   ├── stats.py                # Aggregation helpers: summary, emotion dist., daily counts
│   ├── dashboard.py            # Standalone Matplotlib attendance dashboard
│   ├── demo_seed.py            # Seeds 7 days of realistic fake attendance data
│   └── __init__.py
│
├── justin_innovation/
│   ├── emotion_smoother.py     # Rolling-window majority-vote emotion smoother (individual innovation)
│   ├── session_log.py          # CSV logger, quality gates, export_summary()
│   ├── eval_window.py          # Flicker-rate evaluation across window sizes
│   ├── demo.py                 # Standalone innovation demo (4 parts)
│   └── __init__.py
│
├── theja_innovation/
│   ├── drift_monitoring.py     # MLOps drift monitor: KS test, brightness skew, confidence drop
│   └── __init__.py
│
├── utils/
│   └── face_detector.py        # MTCNN face detection → normalised tensor + raw crop
│
├── saved_models/               # Pre-trained checkpoints (not in repo - download separately)
│   ├── face_classif_model.pth
│   ├── face_triplet_model.pth
│   ├── face_arcface_model.pth
│   ├── fer2013_emotion_model.pth
│   ├── fane_emotion_model.pth
│   ├── antispoof_v3.h5
│   └── liveness_model.h5
│
└── logs/
    ├── attendance_session.csv  # Per-session CSV log (justin_innovation.SessionLogger)
    └── mlops_drift_alerts.log  # Drift and skew alerts from MLOpsDriftMonitor
```

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA GPU (recommended) or Apple Silicon (MPS) or CPU fallback

### Steps

```bash
# 1. Clone the repository
git clone <repo-url>
cd Facial-Recognition-with-Emotion-and-Liveness

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download saved model checkpoints and place them in saved_models/
#    (see Model Training section to train from scratch)

# 4. Launch the app
python gradio_app.py
```

Open your browser at `http://localhost:7860`.

For Vercel or another ASGI host, `app.py` exposes the Gradio interface as a FastAPI
application. The local `python gradio_app.py` launch command remains unchanged.

---

## Usage

The Gradio interface is a **single scrollable page** with four sections:

### 1. Live Attendance

- Click **▶ Start Camera** to begin the webcam stream.
- Select a face recognition model from the dropdown.
- Click **⚡ Load AI Models** to load all three models (face, emotion, liveness).
- The live feed displays overlaid FPS, identity name, similarity score, and liveness status.
- The results panel on the right shows identity, liveness score with a progress bar, and raw vs. smoothed emotion.
- Attendance is automatically logged to SQLite and CSV once a known face is confirmed (10-second per-person cooldown).

### 2. Register Face

- Enter an employee's full name.
- Click **▶ Start Camera** to open the live register camera preview.
- Set the frame count (3–10) and click **📸 Auto-Capture** - the system counts down 3-2-1 for each frame and captures automatically.
- Alternatively, upload a photo via the upload box and click **➕ Add Upload to Queue**.
- The queue counter updates in real time. 3+ frames are recommended for quality scoring.
- Click **✅ Register** to enrol - a per-frame quality report is displayed showing sharpness scores and softmax weights for each kept frame.
- Click **■ Stop Camera** when finished capturing.
- The registered identities panel updates immediately after enrolment.
- Use **🗑 Clear All Registrations** to wipe the registry.

### 3. Dashboard

- Select a date (defaults to today) and click **🔄 Refresh Dashboard**.
- KPI cards show: present today, this week's total, average confidence, and spoof attempts.
- The attendance log table lists each person's first-seen time, emotion, liveness result, and confidence.
- Four Plotly charts: 7-day attendance bar chart, emotion distribution donut, arrivals-by-hour bar chart, and per-person sentiment valence timeline.
- **⬇ Export CSV** downloads all attendance events as a flat CSV file.

### 4. Settings

- Adjust similarity threshold (0.30–0.95), liveness enforcement toggle, and liveness threshold.
- Switch the active face recognition model and reload it.
- Test a camera index with a single-frame preview.
- **🌱 Seed Demo Data** inserts 7 days of synthetic attendance for dashboard testing.
- **🗑 Clear Attendance DB** wipes the SQLite database.

---

## Face Recognition - Three Approaches

All three approaches share the **same ResNet-18 backbone** from `face_recognition/model.py`, enabling fair comparison.

| Approach           | Method                                                    | Cosine AUC |
| ------------------ | --------------------------------------------------------- | ---------- |
| A - Classification | 4000-class softmax; backbone features used at inference   | 0.8924     |
| B - Triplet Loss   | Semi-hard mining with P×K batches; 128-dim embedding head | 0.9025     |
| C - ArcFace ★ Best | Angular margin loss with margin=0.5, scale=64             | 0.9250     |

### Training

```bash
# Approach A
python -m face_recognition.train_classification --data_root data/...

# Approach B (warm-start from A)
python -m face_recognition.train_triplet --init_from saved_models/face_classif_model.pth

# Approach C (ArcFace; ConvNeXt-Tiny backbone, warm-start optional)
python -m face_recognition.train_arcface --arch convnext_tiny

# Evaluate all three and produce comparison ROC
python -m face_recognition.evaluate_all
```

### Dataset

The CMU Multi-PIE / 11-785 face dataset is expected at:

```
data/11-785-fall-20-homework-2-part-2/
    classification_data/train_data/<identity_id>/*.jpg
    classification_data/val_data/<identity_id>/*.jpg
    verification_pairs_val.txt
```
FER2013 dataset: https://www.kaggle.com/datasets/msambare/fer2013
---
FANE dataset: https://www.kaggle.com/datasets/furcifer/fane-facial-expressions-and-emotion-dataset

## Emotion Recognition

A **ResNet-34** classifier fine-tuned for 7 emotion classes: `angry, disgust, fear, happy, sad, surprise, neutral`.

Supports two datasets:

- **FER2013** - stratified train/val split from the `train/` folder.
- **FANE** - separate `train/`, `val/`, `test/` folders prepared by `emotion/prepare_fane_splits.py`.

Class imbalance is handled via weighted random sampling, class-weighted loss, or their combination.

```bash
# FER2013
python -m emotion.train --dataset fer2013 --data_root data/fer2013 --epochs 50

# FANE (prepare splits first)
python -m emotion.prepare_fane_splits --source data/fane_raw --output data/fane_split
python -m emotion.train --dataset fane_split --data_root data/fane_split

# Evaluate
python -m emotion.evaluate --dataset fer2013 --data_root data/fer2013
```

Real-time smoothing is applied by `justin_innovation.EmotionSmoother` (rolling majority vote, window=15 frames), reducing label flicker.

---

## Anti-Spoofing / Liveness Detection

The liveness pipeline fuses two sources of evidence:

**CNN Texture Classifier** (`anti_spoof/model.py`)

- MobileNetV2 backbone with a custom binary head.
- Trained on the `nguyenkhoa/antispoofing-3` HuggingFace dataset.
- Outputs P(spoof) → `prob_real = 1 - P(spoof)`.

**Behavioural Fusion** (`dominic_innovation/liveness.py`)

- Eye-blink detection (Haar cascade, absence of eyes → blink)
- Head-turn detection (frontal vs. profile face, left/right eye offset)
- Mouth-action detection (mouth aspect ratio thresholding)
- Multi-user tracking via centroid displacement

Final liveness score: `CNN_confidence + behavioral_count × 7.5`, clamped to [0, 100].

`LivenessDetector` is defined in `dominic_innovation/liveness.py` and re-exported through `anti_spoof/predict.py` so the rest of the system requires no import changes. The full frame is passed from `core/pipeline.py` into `is_real()` on every liveness check, enabling behavioural signals to run alongside the CNN classifier in the live attendance loop.

```bash
python -m anti_spoof.train      # trains antispoof_v3.h5
```

---

## Individual Innovations

### Dominic - Behavioural-Fusion Liveness Detection

The baseline anti-spoof system (`anti_spoof/predict.py`) classified liveness from a single CNN pass on a pre-cropped face, with no awareness of user behaviour across frames. A static photo or a screen replay could pass if the texture looked convincing enough.

**Solution:** `dominic_innovation/liveness.py`

The `LivenessDetector` class fuses three independent behavioural signals with the CNN texture score across the full video frame:

1. **Blink detection** - a Haar eye cascade monitors whether both eyes are visible. Consecutive frames without detected eyes register as a blink event, timestamped and held for 15 seconds.
2. **Head-turn detection** - profile cascades (left and right, via frame mirroring) and eye-centre offset within a frontal detection together determine whether the user has turned their head. Events are held for 35 seconds.
3. **Mouth-action detection** - the mouth aspect ratio (height / width) is thresholded to detect open/close events. Held for 25 seconds.
4. **Multi-user centroid tracking** - each face is assigned a persistent ID across frames using Euclidean centroid displacement, so behavioural state is correctly accumulated per person even when multiple faces are present.

Final score: `CNN_confidence + active_signal_count × 10`, clamped to [0, 100]. A face is accepted as live only when a blink has been observed within its timeout window and the total score clears the threshold - making replay attacks significantly harder to sustain.

The detector exposes two modes with a single class:

- **Pipeline mode** - `is_real(face_image, full_frame=frame)` returns `(bool, float)`, compatible with `core/pipeline.py`'s existing `_safe_is_real()` helper. No drawing is performed.
- **Standalone / debug mode** - `process_frame(frame, show_debug=True)` annotates the frame in-place with per-user HUD panels, bounding boxes, and signal timers.

`anti_spoof/predict.py` was updated to re-export `LivenessDetector` from this module, and `core/pipeline.py` was updated to pass `full_frame=frame_bgr` into every liveness call - the only change needed in the pipeline.

```bash
# Run the standalone webcam demo with debug HUD
python -m dominic_innovation.liveness
```

### Anh - Quality-Aware Multi-Frame Face Registration

Standard enrolment averages all registration frames equally. Low-quality frames (blurry, dark, overexposed) degrade the registered embedding and increase false rejections.

**Solution:** `anh_innovation/quality_registration.py`

1. Each frame is scored on two axes: sharpness (Laplacian variance) and brightness (deviation from 128).
2. Frames below a minimum quality threshold (0.15) are discarded.
3. Remaining embeddings are aggregated using **softmax-weighted averaging** - sharper, better-lit frames contribute proportionally more.

```
Standard:       embedding = mean(e1, e2, ..., eN)
Quality-aware:  embedding = sum(softmax(q_i × T) × e_i)
```

This innovation is directly integrated into `core/registry.py` and the Gradio registration flow. After enrolment, the UI displays a full per-frame quality report showing each frame's sharpness score, whether it was kept or discarded, and its softmax weight - making the weighting decision transparent and auditable.

**Multi-frame queue workflow** (implemented in `gradio_app.py`):

- The Register section streams a live camera preview with Start/Stop controls.
- **Auto-Capture** counts down 3-2-1 per frame and captures N frames automatically from the live feed.
- Upload photos can be added to the queue individually via the upload box.
- Frames accumulate in a module-level buffer; enrolment processes all queued frames at once through the quality scorer.

```bash
# Run the 3-part demo (no model required for Part 1)
python -m anh_innovation.demo

# Run the contamination benchmark (QW vs equal-weight under increasing bad-frame contamination)
python -m anh_innovation.benchmark --model_path saved_models/face_arcface_model.pth
```

### Justin - Emotion Smoothing & Session Attendance Log

Raw emotion classifiers flicker frame-to-frame; attendance demos also need a durable audit trail beyond on-screen overlays.

**Solution:** `justin_innovation/`

| Module | Role |
| ------ | ---- |
| `emotion_smoother.py` | `EmotionSmoother` - rolling window (default 15 frames), majority vote, confidence tie-break |
| `session_log.py` | `SessionLogger` - appends to `logs/attendance_session.csv` with per-person cooldown (default 10 s) |

**CSV columns:** `timestamp`, `person_id`, `similarity`, `liveness_passed`, `emotion_raw`, `emotion_conf_raw`, `emotion_smooth`, `emotion_conf_smooth`

Integrated in `gradio_app.py`: the live feed shows raw vs. smoothed emotion; when a known face passes **quality gates**, CSV rows are written and `nathan_innovation.log_attendance` updates SQLite for the Dashboard.

| Feature | Module | Description |
| ------- | ------ | ----------- |
| Quality-aware logging | `session_log.py` | `maybe_log` only when similarity ≥ τ, emotion conf ≥ τ, optional liveness pass |
| Session summary | `session_log.py` | `export_summary()` → `logs/session_YYYYMMDD_HHMMSS_summary.md` (auto on **Stop Camera**) |
| Window evaluation | `eval_window.py` | Compare flicker rate (label changes/min) for windows {5, 15, 30} |

```python
from justin_innovation import EmotionSmoother, SessionLogger

smoother = EmotionSmoother(window_size=15)
logger = SessionLogger(
    cooldown_sec=10.0,
    gate_enabled=True,
    min_similarity=0.55,
    min_emotion_conf=0.20,
    require_liveness=False,
)
logger.export_summary()  # markdown report + gate stats
```

```bash
# Interactive demo (no camera required)
py -m justin_innovation.demo

# Optional: include your real session log in PART 4
py -m justin_innovation.demo --csv logs/attendance_session.csv

# Compare smoothing windows on a recorded session CSV
py -m justin_innovation.eval_window --csv logs/attendance_session.csv --windows 5 15 30
```

### Dhruv - Valence-Based Emotion Analysis

`dhruv_innovation/` provides tools for reducing frame-to-frame emotion label jitter and for session-level analysis of attendance logs:

- `eval_smoother.py` - script to evaluate different smoothing strategies (rolling majority vote, exponential smoothing) on an attendance CSV and write per-session reports.
- `visualize_session.py` - lightweight plots for emotion timelines and confidence traces per attendee.
- `scripts/` - small helper scripts for converting `logs/attendance_session.csv` into standardized evaluation CSVs.

Key features:

- Rolling-window majority-vote smoothing with configurable window size and label confidence threshold.
- Per-session emotion summary (mode, entropy, average confidence) exported as CSV.
- Integration with `nathan_innovation` dashboard formats for quick inspection.

Example usage:

```bash
# Demo
python -m dhruv_innovation.demo

# Evaluate smoothing strategies on an attendance CSV
python -m dhruv_innovation.eval_smoother --csv logs/attendance_session.csv --out_dir dhruv_innovation/

```

This module is useful for improving UX in the live demo by stabilizing emotion overlays and producing cleaner analytics for the report.

### Nathan - Attendance Database & Analytics Dashboard

`nathan_innovation/` provides the attendance persistence layer:

- `db.py` - SQLite CRUD (`init_db`, `log_attendance`, `get_events`)
- `stats.py` - aggregation functions: daily summaries, emotion distribution, arrivals by hour, average confidence
- `dashboard.py` - standalone Matplotlib dashboard (alternative to Gradio)
- `demo_seed.py` - seeds 7 days of realistic synthetic data for testing

### Theja - MLOps Drift Monitoring

theja_innovation/drift_monitoring.py implements an active production-grade telemetry monitoring pipeline integrated directly into the live camera feed thread loop using Kolmogorov-Smirnov (KS) two-sample distribution testing:
* **Initialization:** Requires a 45-frame camera exposure stabilization period, followed by an accelerated 50-frame accumulation pool to establish reference brightness and confidence distribution profiles.
* **Data Skew Detection:** Continuously tracks ambient lighting environment shifts by comparing live brightness states against the reference baseline. Alerts are logged to disk when the KS test p-value drops below `0.05` and the mean brightness shift deviates past `brightness_shift_threshold` (12 units).
* **Model Drift Detection:** Evaluates classifier degradation or physical obstructions by running a KS test on the rolling joint confidence scores ($Similarity \times Liveness$). Logs an explicit critical alert when the computed p-value drops below `0.05` and confidence slips past the `confidence_drop_threshold` (0.15) relative to the baseline profile.
* **Asymmetric Recovery Gate:** Features a hardcoded single-frame fast recovery bypass inside `app.py`. If a frame yields pristine metrics (confidence within threshold limits and brightness shift fully stabilized), it instantly overrides lagging rolling statistics and clears the dashboard alerts back to a green "Normal" state.

* Alerts are piped directly to `logs/mlops_drift_alerts.log` in real time.
---

## Configuration & Thresholds

| Parameter            | Default    | Description                                            |
| -------------------- | ---------- | ------------------------------------------------------ |
| Similarity threshold | 0.55       | Minimum cosine similarity to claim identity            |
| Liveness threshold   | 0.50       | Minimum `prob_real` to pass liveness                   |
| Enforce liveness     | False      | If True, blocks identity + emotion when spoof detected |
| Emotion window       | 15 frames  | Rolling window for smoothing                           |
| Session cooldown     | 10 seconds | Minimum gap between attendance logs per person         |
| Log gate enabled     | True       | Require min similarity + emotion conf before CSV log   |
| Min emotion conf log | 0.20       | Minimum smoothed emotion confidence to log             |
| Require liveness log | False      | Only log when liveness passes                          |
| Quality min score    | 0.15       | Frames below this are discarded during enrolment       |
| Softmax temperature  | 5.0        | Controls sharpness of quality-weighted averaging       |

All thresholds can be adjusted live via the **Settings** panel without restarting the app.

---

## Saved Model Checkpoints

| File                        | Description                                      |
| --------------------------- | ------------------------------------------------ |
| `face_classif_model.pth`    | Approach A - ResNet-18 classification checkpoint |
| `face_triplet_model.pth`    | Approach B - ResNet-18 triplet loss checkpoint   |
| `face_arcface_model.pth`    | Approach C - ArcFace checkpoint (backbone only)  |
| `fer2013_emotion_model.pth` | ResNet-34 emotion model trained on FER2013       |
| `fane_emotion_model.pth`    | ResNet-34 emotion model trained on FANE          |
| `antispoof_v3.h5`           | MobileNetV2 liveness classifier (primary)        |
| `liveness_model.h5`         | Alternate liveness model checkpoint              |

All checkpoints belong in `saved_models/`. The directory is tracked with `.gitkeep` but the binary files are not committed to the repository.

---

## Data Flow

```
Webcam frame (BGR)
        │
        ▼
MTCNN face detection
  ├── No face → skip
  └── Face detected (112×112 crop)
          │
          ├──► LivenessDetector (every 5 frames)
          │       CNN texture + behavioural signals
          │       → is_live, liveness_score
          │
          ├──► FaceModel backbone (every 5 frames)
          │       512-dim L2-normalised embedding
          │       → cosine similarity vs. registry
          │       → name, similarity
          │
          └──► EmotionPredictor (every 3 frames)
                  ResNet-34 → 7-class softmax
                  justin_innovation.EmotionSmoother (window=15)
                  → label, confidence
                          │
                          ▼
                  justin_innovation.SessionLogger (10s cooldown)
                  → logs/attendance_session.csv
                          │
                          ▼
                  nathan_innovation.db
                  → nathan_innovation/attendance.db
```

---

## Known Limitations

- **Liveness detection** relies on Haar cascade cascades for eye/mouth detection, which can be unreliable under poor lighting or extreme head poses.
- **Emotion model confidence** is often low (0.2–0.5 range); this is a known characteristic of FER2013-trained models due to ambiguous training labels.
- **Registry persistence** uses a plain pickle file (`registered_faces.pkl`). For production use, this should be replaced with a proper database.
- **Multi-face support** is not implemented - the pipeline processes only the highest-confidence face per frame.
- The **MLOps drift monitor** requires a warm-up period to establish a baseline before alerts can fire.

---

## License

This project was developed for academic purposes as part of COS30082 Applied Machine Learning. All dataset usage is subject to the original dataset licences.
