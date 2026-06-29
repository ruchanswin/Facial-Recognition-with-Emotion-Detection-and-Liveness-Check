"""
FaceAttend - Unified Gradio UI  (single page, no tabs)

Sections rendered top-to-bottom on one scrollable page:
  1. Live Attendance - camera stream + detection results
  2. Register Face - enrol employees into the face registry
  3. Dashboard - attendance analytics, charts, export
  4. Settings - model, thresholds, camera, data management

All sections share the same SQLite database and face registry on disk.
"""

from __future__ import annotations

import gc
import pickle
import platform
import random
import sqlite3
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import gradio as gr

#  Optional heavy deps 
try:
    import cv2
    import numpy as np
    CV2_OK = True
except ImportError:
    CV2_OK = False
    np = None

try:
    import plotly.graph_objects as go
    import plotly.io as pio
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False

try:
    import pandas as pd
    PD_OK = True
except ImportError:
    PD_OK = False

# TensorFlow GPU guard
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for _gpu in gpus:
            tf.config.experimental.set_memory_growth(_gpu, True)
except Exception:
    pass

#  ML project imports
MODELS_OK    = False
_import_error = ""
try:
    from core.pipeline import load_models, run_pipeline
    from core.registry import enroll, load_registry, save_registry
    from theja_innovation.drift_monitoring import MLOpsDriftMonitor
    from justin_innovation import EmotionSmoother, SessionLogger
    from nathan_innovation.db import log_attendance, get_events, init_db
    from nathan_innovation import stats as _stats_mod
    from dhruv_innovation import ValenceSmoother         
    from dhruv_innovation import team_sentiment_summary, person_valence_timeline              
    MODELS_OK = True
except Exception as _e:
    _import_error = str(_e)

#  Stubs when ML libs are unavailable 
if not MODELS_OK:
    def load_registry():
        p = Path("registered_faces.pkl")
        return pickle.loads(p.read_bytes()) if p.exists() else {}

    def save_registry(db):
        Path("registered_faces.pkl").write_bytes(pickle.dumps(db))

    def enroll(name, frames_bgr, face_model, device):
        raise RuntimeError("ML libraries not available.")

    _DB_PATH = Path("nathan_innovation/attendance.db")

    def _connect():
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db():
        with _connect() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS attendance_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL, person_id TEXT NOT NULL,
                emotion TEXT, liveness_passed INTEGER, confidence REAL NOT NULL)""")

    def get_events(date_str=None):
        init_db()
        with _connect() as conn:
            if date_str is None:
                rows = conn.execute(
                    "SELECT * FROM attendance_events ORDER BY timestamp").fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM attendance_events WHERE timestamp LIKE ? ORDER BY timestamp",
                    (f"{date_str}%",)).fetchall()
        return [dict(r) for r in rows]

    def log_attendance(person_id, emotion=None, liveness_passed=None, confidence=0.0):
        init_db()
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        with _connect() as conn:
            conn.execute(
                "INSERT INTO attendance_events "
                "(timestamp,person_id,emotion,liveness_passed,confidence) VALUES (?,?,?,?,?)",
                (ts, person_id, emotion,
                 None if liveness_passed is None else int(liveness_passed), confidence))

    class EmotionSmoother:
        def __init__(self, ws=15): self._buf = deque(maxlen=ws)
        def update(self, lbl, conf): self._buf.append((lbl, conf)); return lbl
        def reset(self): self._buf.clear()

    class SessionLogger:
        def __init__(self, *a, **kw): pass
        def maybe_log(self, *a, **kw): return False
        def configure_gates(self, **kw): pass
        def export_summary(self, *a, **kw): return Path("logs/session_summary.md")
        def print_summary(self): pass

    class _stats_mod:
        @staticmethod
        def attendance_summary(d=None):
            evs = get_events(d or date.today().isoformat())
            seen: dict = {}
            for e in evs:
                if e["person_id"] not in seen:
                    seen[e["person_id"]] = e["timestamp"]
            return sorted(seen)

        @staticmethod
        def emotion_distribution(d=None):
            evs = get_events(d or date.today().isoformat())
            c: dict = {}
            for e in evs:
                c[e["emotion"] or "unknown"] = c.get(e["emotion"] or "unknown", 0) + 1
            return c

        @staticmethod
        def daily_counts(n=7):
            res = []
            today = date.today()
            for off in range(n - 1, -1, -1):
                day = (today - timedelta(days=off)).isoformat()
                res.append((day, len({e["person_id"] for e in get_events(day)})))
            return res

        @staticmethod
        def arrivals_by_hour(d=None):
            evs = get_events(d or date.today().isoformat())
            seen: dict = {}
            for e in evs:
                if e["person_id"] not in seen:
                    seen[e["person_id"]] = datetime.fromisoformat(e["timestamp"]).hour
            c: dict = {}
            for h in seen.values():
                c[h] = c.get(h, 0) + 1
            return c

        @staticmethod
        def avg_confidence(d=None):
            evs = get_events(d or date.today().isoformat())
            return sum(e["confidence"] for e in evs) / len(evs) if evs else 0.0

init_db()

#  Design tokens 
ACCENT  = "#00E5FF"; ACCENT2 = "#7B61FF"; SUCCESS = "#00E676"
DANGER  = "#FF1744"; WARN    = "#FFB300"; MUTED   = "#546E7A"
BG      = "#0A0D14"; SURFACE = "#111520"; CARD    = "#161C2D"; BORDER  = "#1E2740"

EMOTION_COLORS = {
    "happy": "#FFD600", "sad": "#40C4FF", "angry": "#FF1744",
    "surprise": "#FF6D00", "fear": "#AA00FF", "disgust": "#00E676",
    "neutral": "#90A4AE", "confused": "#7B61FF", "shy": "#FF80AB", "unknown": "#546E7A",
}
EMOTION_ICONS = {
    "happy": "😊", "sad": "😢", "angry": "😠", "surprise": "😲", "fear": "😨",
    "disgust": "🤢", "neutral": "😐", "confused": "😕", "shy": "😳", "unknown": "❓",
}
MODEL_CHOICES = {
    "Approach C - ArcFace  ★ Best  (AUC 0.9250)": "saved_models/face_arcface_model.pth",
    "Approach B - Triplet Loss  (AUC 0.9025)":    "saved_models/face_triplet_model.pth",
    "Approach A - Classification  (AUC 0.8924)":  "saved_models/face_classif_model.pth",
}
EMPLOYEES = ["Alice Chen", "Bob Nguyen", "Claire Smith", "David Park", "Emma Wilson"]
EMOTIONS  = ["happy", "neutral", "sad", "angry", "surprise", "fear", "disgust"]
EMOTION_W = [0.35, 0.30, 0.12, 0.08, 0.07, 0.05, 0.03]

#  Global app state 
@dataclass
class AppState:
    lock: threading.Lock       = field(default_factory=threading.Lock)
    cam_running: bool          = False
    cam_thread: Any            = None
    cam_index: int             = 0
    latest_frame: Any          = None
    latest_result: dict        = field(default_factory=dict)
    cam_fps: float             = 0.0
    cam_error: str             = ""
    models: Any                = None
    models_loading: bool       = False
    model_load_error: str      = ""
    face_model_path: str       = list(MODEL_CHOICES.values())[0]
    smoother: Any              = None
    valence_smoother: Any      = None
    logger: Any                = None
    threshold: float           = 0.55
    enforce_liveness: bool     = False
    liveness_threshold: float  = 0.40
    drift_monitor: Any         = None
    baseline_frames: list      = field(default_factory=list)
    frame_skip_counter: int    = 0
    mlops_status_text: str     = "MLOps Engine Status: Initializing baseline"
    mlops_status_style: str    = "neutral"
    log_gate_enabled: bool     = True
    min_emotion_conf_log: float = 0.20
    require_liveness_log: bool = False

_state = AppState()
if MODELS_OK:
    _state.drift_monitor = MLOpsDriftMonitor(buffer_size=60)

_reg_buffer: list = []  # queued numpy RGB images for multi-frame enrollment



def _sync_logger_gates() -> None:
    with _state.lock:
        logger = _state.logger
        thr = _state.threshold
        gate = _state.log_gate_enabled
        emo_thr = _state.min_emotion_conf_log
        req_live = _state.require_liveness_log
    if logger is None or not hasattr(logger, "configure_gates"):
        return
    logger.configure_gates(
        gate_enabled=gate,
        min_similarity=thr if gate else None,
        min_emotion_conf=emo_thr if gate else None,
        require_liveness=req_live,
    )

#  Camera thread 
def _camera_loop() -> None:
    if not CV2_OK:
        with _state.lock:
            _state.cam_error   = "OpenCV not installed."
            _state.cam_running = False
        return

    with _state.lock:
        cam_index = _state.cam_index

    backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(cam_index, backend)

    deadline = time.time() + 3.0
    while not cap.isOpened() and time.time() < deadline:
        time.sleep(0.1)

    if not cap.isOpened():
        with _state.lock:
            _state.cam_error   = f"Cannot open camera {cam_index}."
            _state.cam_running = False
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          15)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    for _ in range(3): cap.read()

    frame_idx  = 0
    _fps_times: deque = deque(maxlen=30)

    while True:
        with _state.lock:
            if not _state.cam_running:
                break
        
        for _ in range(4):
            cap.grab()

        ret, bgr = cap.retrieve()
        if not ret:
            time.sleep(0.02)
            continue

        frame_idx += 1
        _fps_times.append(time.time())
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if frame_idx % 50 == 0:
            gc.collect()

        result: dict = {}
        if frame_idx % 5 == 0:
            with _state.lock:
                models_snap   = _state.models
                smoother_snap = _state.smoother
                valence_smoother_snap = _state.valence_smoother
                logger_snap   = _state.logger
                thr_snap      = _state.threshold
                enf_snap      = _state.enforce_liveness
                lthr_snap     = _state.liveness_threshold
                log_gate_snap = _state.log_gate_enabled
                emo_thr_snap  = _state.min_emotion_conf_log
                req_live_snap = _state.require_liveness_log

            if models_snap is not None:
                try:
                    result = run_pipeline(
                        bgr, models_snap,
                        threshold=thr_snap,
                        enforce_liveness=enf_snap,
                        liveness_threshold=lthr_snap,
                    )
                    if smoother_snap and result.get("emotion"):
                        raw = result["emotion"].split()[-1].lower()
                        result["emotion_smooth"] = smoother_snap.update(
                            raw, result.get("emotion_conf", 0)
                        )
                    if valence_smoother_snap and result.get("emotion"):
                        raw  = result["emotion"].split()[-1].lower()
                        conf = result.get("emotion_conf", 0.0)
                        vs   = valence_smoother_snap.update(raw, conf)
                        result["valence"]   = vs["valence"]
                        result["sentiment"] = vs["sentiment"]
                        result["vs_window"] = vs["window"]

                    if logger_snap and result.get("name"):
                        raw_e  = (result.get("emotion") or "").split()[-1].lower()
                        sm_e   = result.get("emotion_smooth") or raw_e
                        conf_r = result.get("emotion_conf", 0.0)
                        conf_s = (
                            smoother_snap.mean_confidence(sm_e)
                            if smoother_snap and hasattr(smoother_snap, "mean_confidence")
                            else conf_r
                        )
                        if hasattr(logger_snap, "configure_gates"):
                            logger_snap.configure_gates(
                                gate_enabled=log_gate_snap,
                                min_similarity=thr_snap if log_gate_snap else None,
                                min_emotion_conf=emo_thr_snap if log_gate_snap else None,
                                require_liveness=req_live_snap,
                            )
                        logged = logger_snap.maybe_log(
                            result["name"], result.get("similarity", 0.0),
                            raw_e, conf_r, sm_e, conf_s,
                            liveness_passed=result.get("is_live"),
                        )
                        if logged:
                            log_attendance(
                                result["name"], emotion=sm_e,
                                liveness_passed=result.get("is_live"),
                                confidence=result.get("similarity", 0.0),
                            )

                    # MLOps Analytics Section
                    sim = result.get("similarity", 0.0)
                    live_score = result.get("liveness_score", 0.0)

                    monitor = _state.drift_monitor

                    # SECURITY GATE
                    if live_score < 0.20:  # If liveness drops below 20%, it's a spoof attempt
                        _state.mlops_status_style = "critical"
                        _state.mlops_status_text = "SECURITY ALERT: SPOOF ATTACK DETECTED"

                        monitor.current_brightness.clear()
                        monitor.current_confidences.clear()

                    else:
                        confidence_metric = float(sim * live_score)
                        confidence_metric = max(confidence_metric, 0.05)
                    
                        if not monitor.baseline_established:
                            # Skip the first 45 frames to allow camera exposure sensor stabilization
                            if _state.frame_skip_counter < 45:
                                _state.frame_skip_counter += 1
                                _state.mlops_status_text = "Warming up camera sensor"
                                _state.mlops_status_style = "warm_up"
                            
                            # Accumulate 50 frames for baseline profile registration
                            elif len(_state.baseline_frames) < 50:
                                _state.baseline_frames.append(bgr) # Monitor expects BGR arrays
                                _state.mlops_status_text = f"Registering baseline profile ({len(_state.baseline_frames)}/50)"
                                _state.mlops_status_style = "profiling"
                            
                            # Finalize pipeline and lock in distributions
                            else:
                                monitor.establish_baseline_reference(_state.baseline_frames, target_confidence=0.75)
                                _state.mlops_status_text = "Engine Status: Normal (Baseline registered)"
                                _state.mlops_status_style = "normal"
                        
                        else:
                            mlops_report = monitor.collect_live_telemetry(bgr, confidence_metric)
                            
                            current_brightness = monitor.compute_frame_brightness(bgr)
                            brightness_shift = abs(current_brightness - monitor.reference_brightness_mean)
                            
                            is_confidence_stable = confidence_metric >= (monitor.reference_confidence_mean - monitor.confidence_drop_threshold)
                            is_lighting_stable = brightness_shift < monitor.brightness_shift_threshold
                            
                            if is_confidence_stable and is_lighting_stable:
                                _state.mlops_status_text = "Engine Status: Normal (Data stable)"
                                _state.mlops_status_style = "normal"
                            
                            # fall back to the rolling statistical distribution reports
                            elif mlops_report:
                                if mlops_report["model_drift"]:
                                    _state.mlops_status_text = f"Alert: {mlops_report['msg']}"
                                    _state.mlops_status_style = "critical"
                                elif mlops_report["data_skew"]:
                                    _state.mlops_status_text = f"Warning: {mlops_report['msg']}"
                                    _state.mlops_status_style = "warning"
                                else:
                                    _state.mlops_status_text = "Engine Status: Normal (Data stable)"
                                    _state.mlops_status_style = "normal"

                except Exception:
                    pass

        fps_val = 0.0
        if len(_fps_times) >= 2:
            elapsed = _fps_times[-1] - _fps_times[0]
            fps_val = (len(_fps_times) - 1) / elapsed if elapsed > 0 else 0.0

        with _state.lock:
            _state.latest_frame  = rgb
            _state.cam_fps       = round(fps_val, 1)
            _state.cam_error     = ""
            if result:
                _state.latest_result = result

        time.sleep(0.05)

    cap.release()
    with _state.lock:
        _state.latest_frame = None


def _start_camera() -> str:
    with _state.lock:
        if _state.cam_running:
            return "already_running"
        old = _state.cam_thread
    if old is not None and old.is_alive():
        return "already_running"
    with _state.lock:
        _state.cam_running   = True
        _state.cam_error     = ""
        _state.latest_result = {}
    t = threading.Thread(target=_camera_loop, daemon=True, name="CameraWorker")
    with _state.lock:
        _state.cam_thread = t
    t.start()
    return "started"


def _stop_camera() -> None:
    with _state.lock:
        _state.cam_running   = False
        _state.cam_thread    = None
        _state.latest_frame  = None
        _state.latest_result = {}


def _load_models_safe(face_model_path: str) -> str:
    if not MODELS_OK:
        return f"❌ ML libraries not available: {_import_error[:120]}"
    with _state.lock:
        _state.models_loading   = True
        _state.model_load_error = ""
        _state.face_model_path  = face_model_path
    try:
        m        = load_models(face_model_path,
                               "saved_models/fer2013_emotion_model.pth",
                               "saved_models/antispoof_v3.h5")
        smoother = EmotionSmoother(ws=15)
        valence_smoother = ValenceSmoother(base_window=15)
        with _state.lock:
            gate = _state.log_gate_enabled
            thr = _state.threshold
            emo_thr = _state.min_emotion_conf_log
            req_live = _state.require_liveness_log
        logger = SessionLogger(
            gate_enabled=gate,
            min_similarity=thr if gate else None,
            min_emotion_conf=emo_thr if gate else None,
            require_liveness=req_live,
        )
        with _state.lock:
            _state.models         = m
            _state.smoother       = smoother
            _state.valence_smoother = valence_smoother 
            _state.logger         = logger
            _state.models_loading = False
        return "✅ Models loaded successfully!"
    except Exception as e:
        msg = str(e)
        with _state.lock:
            _state.models_loading   = False
            _state.model_load_error = msg
        return f"❌ {msg}"

#  Display helpers 
def _emotion_badge_html(emotion: str) -> str:
    clean = emotion.split()[-1].lower() if emotion else "unknown"
    icon  = EMOTION_ICONS.get(clean, "❓")
    color = EMOTION_COLORS.get(clean, MUTED)
    return (f'<span style="display:inline-block;padding:3px 11px;border-radius:20px;'
            f'font-size:12px;font-weight:700;background:rgba(0,0,0,0.35);'
            f'color:{color};border:1px solid {color};">{icon} {clean}</span>')


def _liveness_badge_html(is_live) -> str:
    if is_live is True:
        return (f'<span style="display:inline-block;padding:3px 11px;border-radius:20px;'
                f'font-size:12px;font-weight:700;background:rgba(0,230,118,.12);'
                f'color:{SUCCESS};border:1px solid {SUCCESS};">✓ LIVE</span>')
    if is_live is False:
        return (f'<span style="display:inline-block;padding:3px 11px;border-radius:20px;'
                f'font-size:12px;font-weight:700;background:rgba(255,23,68,.12);'
                f'color:{DANGER};border:1px solid {DANGER};">✗ SPOOF</span>')
    return (f'<span style="display:inline-block;padding:3px 11px;border-radius:20px;'
            f'font-size:12px;font-weight:700;background:rgba(84,110,122,.12);'
            f'color:{MUTED};border:1px solid {MUTED};">- n/a</span>')


def _results_html(result: dict) -> str:
    name       = result.get("name") or "Unknown"
    sim        = result.get("similarity",     0.0)
    is_live    = result.get("is_live")
    live_score = result.get("liveness_score", 0.0)
    emotion    = result.get("emotion")    or ""
    emo_smooth = result.get("emotion_smooth") or ""
    emo_conf   = result.get("emotion_conf",   0.0)
    valence   = result.get("valence")
    sentiment = result.get("sentiment", "")
    vs_window = result.get("vs_window", 15)

    name_color  = ACCENT if name not in ("Unknown", None, "") else MUTED
    score_color = SUCCESS if live_score >= 0.4 else DANGER
    sim_pct     = int(min(sim,        1.0) * 100)
    live_pct    = int(min(live_score, 1.0) * 100)

    if valence is not None:
        # Map valence [-1,+1] → bar width [0,100]%
        bar_pct = int((valence + 1.0) / 2.0 * 100)
        bar_pct = max(2, min(98, bar_pct))   # keep bar visible at extremes
 
        sent_cfg = {
            "positive": ("#00E676", "▲ Positive atmosphere"),
            "neutral":  ("#FFB300", "◆ Neutral"),
            "stressed": ("#FF1744", "▼ Stress detected"),
        }
        sent_color, sent_label = sent_cfg.get(sentiment, ("#546E7A", "—"))

        sentiment_card_html = f"""
        <div style="background:{CARD};border:1px solid {BORDER};border-radius:12px;
                    padding:14px 18px;margin-bottom:10px;">
            <div style="color:{MUTED};font-size:10px;font-weight:700;
                        text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">
            Sentiment · Adaptive Window ({vs_window} frames)
            </div>
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">
            <span style="color:{sent_color};font-size:13px;font-weight:700;">{sent_label}</span>
            <span style="color:{MUTED};font-size:11px;">valence {valence:+.2f}</span>
            </div>
            <div style="background:{SURFACE};border-radius:4px;height:8px;overflow:hidden;position:relative;">
            <!-- track background: red → amber → green -->
            <div style="position:absolute;inset:0;
                        background:linear-gradient(90deg,#FF1744 0%,#FFB300 50%,#00E676 100%);
                        opacity:0.25;"></div>
            <!-- needle: thin vertical line at valence position -->
            <div style="position:absolute;top:0;bottom:0;width:3px;border-radius:2px;
                        background:{sent_color};left:{bar_pct}%;transform:translateX(-50%);"></div>
            </div>
            <div style="display:flex;justify-content:space-between;
                        margin-top:3px;font-size:9px;color:{MUTED};">
            <span>Stressed</span><span>Neutral</span><span>Positive</span>
            </div>
        </div>"""
    else:
        sentiment_card_html = ""

    card  = (f"background:{CARD};border:1px solid {BORDER};border-radius:12px;"
             f"padding:14px 18px;margin-bottom:10px;")
    lbl   = (f"color:{MUTED};font-size:10px;font-weight:700;text-transform:uppercase;"
             f"letter-spacing:1px;margin-bottom:6px;")
    track = f"background:{SURFACE};border-radius:4px;height:6px;margin-top:6px;overflow:hidden;"
    bar_id   = f"background:linear-gradient(90deg,{ACCENT2},{ACCENT});width:{sim_pct}%;height:100%;border-radius:4px;"
    bar_live_col = f"linear-gradient(90deg,{DANGER},{SUCCESS})" if live_score >= 0.4 else DANGER
    bar_live = f"background:{bar_live_col};width:{live_pct}%;height:100%;border-radius:4px;"

    emo_html  = _emotion_badge_html(emotion)    if emotion    else f'<span style="color:{MUTED};">-</span>'
    sm_html   = _emotion_badge_html(emo_smooth) if emo_smooth else f'<span style="color:{MUTED};">-</span>'
    live_html = _liveness_badge_html(is_live)

    empty_note = (
        f'<div style="color:{MUTED};font-size:12px;padding:8px;">'
        f'Start camera and load models to see results.</div>'
        if not result else ""
    )

    style_mapping = {
        "neutral":   {"bg": SURFACE, "border": BORDER, "text": MUTED},
        "warm_up":   {"bg": SURFACE, "border": ACCENT2, "text": ACCENT2},
        "profiling": {"bg": SURFACE, "border": ACCENT, "text": ACCENT},
        "normal":    {"bg": "#14532d", "border": "#22c55e", "text": "#dcfce7"},
        "warning":   {"bg": "#7c2d12", "border": "#f97316", "text": "#ffedd5"},
        "critical":  {"bg": "#7f1d1d", "border": "#ef4444", "text": "#fecaca"}
    }
    
    cfg = style_mapping.get(_state.mlops_status_style, style_mapping["neutral"])
    
    mlops_card_html = f"""
    <div style="background:{cfg['bg']}; border: 1px solid {cfg['border']}; color: {cfg['text']};
                border-radius:12px; padding:12px 18px; margin-bottom:10px; 
                text-align:center; font-size:11px; font-weight:700; letter-spacing:0.5px;">
        ⚙️ {_state.mlops_status_text}
    </div>
    """

    return (
        f'<div style="font-family:\'DM Sans\',sans-serif;">'
        f'<div style="{card}">'
        f'<div style="{lbl}">Identity</div>'
        f'<div style="font-size:22px;font-weight:800;color:{name_color};">{name}</div>'
        f'<div style="color:{MUTED};font-size:12px;margin-top:4px;">'
        f'Similarity: <b style="color:#E8EAF6;">{sim:.1%}</b>'
        f'</div>'
        f'<div style="{track}"><div style="{bar_id}"></div></div>'
        f'</div>'
        f'<div style="{card}">'
        f'<div style="{lbl}">Liveness</div>'
        f'<div style="display:flex;align-items:center;gap:10px;">'
        f'{live_html}'
        f'<span style="color:{score_color};font-size:18px;font-weight:800;">{live_score:.1%}</span>'
        f'</div>'
        f'<div style="{track}"><div style="{bar_live}"></div></div>'
        f'</div>'
        f'<div style="{card}">'
        f'<div style="{lbl}">Emotion</div>'
        f'<div style="margin-bottom:6px;">'
        f'<span style="color:{MUTED};font-size:11px;">Raw </span>{emo_html}'
        f'<span style="color:{MUTED};font-size:10px;margin-left:6px;">({emo_conf:.0%})</span>'
        f'</div>'
        f'<div><span style="color:{MUTED};font-size:11px;">Smoothed </span>{sm_html}</div>'
        f'</div>'
        f'{sentiment_card_html}'
        f'{empty_note}'
        f'{mlops_card_html}'
        f'</div>'
    )


#  Camera stream generator 
def _camera_stream():
    """Gradio streaming generator - yields (frame_array, results_html)."""
    blank      = np.zeros((480, 640, 3), dtype=np.uint8) if CV2_OK else None
    _last_yield = 0.0
    _MIN_GAP    = 0.05

    while True:
        now = time.time()
        if now - _last_yield < _MIN_GAP:
            time.sleep(0.01)
            continue

        with _state.lock:
            running = _state.cam_running
            frame   = _state.latest_frame
            result  = dict(_state.latest_result)
            fps     = _state.cam_fps
            error   = _state.cam_error

        if not CV2_OK:
            _last_yield = time.time()
            yield None, _results_html({})
            time.sleep(0.5)
            continue

        if not running:
            out = blank.copy()
            cv2.putText(out, "Camera Off  -  press  Start Camera", (70, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (84, 110, 122), 2)
            _last_yield = time.time()
            yield out, _results_html({})
            time.sleep(0.1)
            continue

        if error:
            out = blank.copy()
            cv2.putText(out, error[:55], (20, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 23, 68), 2)
            _last_yield = time.time()
            yield out, _results_html({})
            time.sleep(0.2)
            continue

        if frame is None:
            out = blank.copy()
            cv2.putText(out, "Starting camera...", (170, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (84, 110, 122), 2)
            _last_yield = time.time()
            yield out, _results_html({})
            time.sleep(0.05)
            continue

        display = frame.copy()
        cv2.rectangle(display, (8, 8), (112, 32), (22, 28, 45), -1)
        cv2.putText(display, f"FPS {fps:.0f}", (14, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 229, 255), 1, cv2.LINE_AA)

        if result:
            name = result.get("name") or "Unknown"
            sim  = result.get("similarity", 0.0)
            label_color = (0, 230, 118) if name not in ("Unknown", None, "") else (84, 110, 122)
            h = display.shape[0]
            cv2.rectangle(display, (8, h - 50), (310, h - 8), (22, 28, 45), -1)
            cv2.putText(display, f"{name}  {sim:.0%}", (14, h - 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, label_color, 2, cv2.LINE_AA)
            is_live   = result.get("is_live")
            live_text = "LIVE" if is_live else ("SPOOF" if is_live is False else "")
            if live_text:
                live_color = (0, 230, 118) if is_live else (255, 23, 68)
                w = display.shape[1]
                cv2.rectangle(display, (w - 120, 8), (w - 8, 32), (22, 28, 45), -1)
                cv2.putText(display, live_text, (w - 110, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, live_color, 1, cv2.LINE_AA)

        _last_yield = time.time()
        yield display, _results_html(result)


def _reg_preview_stream():
    """Streaming generator for the Register Face camera preview."""
    blank = np.zeros((360, 480, 3), dtype=np.uint8) if CV2_OK else None
    while True:
        with _state.lock:
            running = _state.cam_running
            frame   = _state.latest_frame
            error   = _state.cam_error

        if not CV2_OK:
            yield None
            time.sleep(0.5)
            continue

        if not running:
            out = blank.copy()
            cv2.putText(out, "Press  Start Camera  to begin", (45, 185),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (84, 110, 122), 1, cv2.LINE_AA)
            yield out
            time.sleep(0.1)
            continue

        if error:
            out = blank.copy()
            cv2.putText(out, error[:55], (20, 185),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 23, 68), 1, cv2.LINE_AA)
            yield out
            time.sleep(0.2)
            continue

        if frame is None:
            out = blank.copy()
            cv2.putText(out, "Starting camera...", (120, 185),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (84, 110, 122), 1, cv2.LINE_AA)
            yield out
            time.sleep(0.05)
            continue

        yield frame.copy()
        time.sleep(0.05)


#  Plotly chart helpers
_CL = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter,sans-serif", color="#E8EAF6", size=12),
    margin=dict(l=10, r=10, t=28, b=10),
)


def _bar_chart(daily):
    if not PLOTLY_OK or not daily:
        return None
    days   = [d[5:] for d, _ in daily]
    counts = [c for _, c in daily]
    fig = go.Figure(go.Bar(
        x=days, y=counts,
        marker=dict(color=counts, colorscale=[[0, ACCENT2], [1, ACCENT]], line=dict(width=0)),
        text=counts, textposition="outside", textfont=dict(size=11, color="#E8EAF6"),
    ))
    fig.update_layout(**_CL, height=220,
        xaxis=dict(showgrid=False, tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor=BORDER, zeroline=False))
    return fig


def _donut_chart(emotions):
    if not PLOTLY_OK or not emotions:
        return None
    lbs  = list(emotions.keys()); vals = list(emotions.values())
    cols = [EMOTION_COLORS.get(l, MUTED) for l in lbs]
    dlbs = [f"{EMOTION_ICONS.get(l,'❓')} {l}" for l in lbs]
    fig = go.Figure(go.Pie(labels=dlbs, values=vals, hole=0.55,
        marker=dict(colors=cols, line=dict(color=BG, width=2)),
        textinfo="label+percent", textfont=dict(size=11)))
    fig.update_layout(**_CL, height=280, showlegend=False)
    return fig


def _arrivals_chart(hourly):
    if not PLOTLY_OK or not hourly:
        return None
    hs = sorted(hourly); cs = [hourly[h] for h in hs]
    fig = go.Figure(go.Bar(
        x=[f"{h:02d}:00" for h in hs], y=cs, marker_color=ACCENT2,
        text=cs, textposition="outside", textfont=dict(size=10, color="#E8EAF6")))
    fig.update_layout(**_CL, height=220,
        xaxis=dict(showgrid=False, tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor=BORDER, zeroline=False))
    return fig

def _valence_timeline_chart(attendees: list, date_str: str):
    """Per-person emotion valence timeline — innovation chart."""
    if not PLOTLY_OK or not attendees:
        return None
 
    import plotly.graph_objects as go
    from dhruv_innovation.valence_stats import person_valence_timeline
 
    fig = go.Figure()
    colors = [ACCENT, ACCENT2, SUCCESS, WARN, DANGER, "#FF6D00", "#AA00FF"]
 
    for i, person in enumerate(sorted(attendees)):
        timeline = person_valence_timeline(person, date_str)
        if not timeline:
            continue
        times    = [t["time"]    for t in timeline]
        valences = [t["valence"] for t in timeline]
        color    = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=times, y=valences,
            mode="lines+markers",
            name=person,
            line=dict(color=color, width=2),
            marker=dict(size=6, color=color),
        ))
 
    # Reference lines
    fig.add_hline(y= 0.25, line_dash="dot", line_color=SUCCESS,
                  annotation_text="positive", annotation_position="right",
                  line_width=1)
    fig.add_hline(y=-0.25, line_dash="dot", line_color=DANGER,
                  annotation_text="stressed", annotation_position="right",
                  line_width=1)
    fig.add_hline(y=0,     line_dash="dot", line_color=MUTED,
                  line_width=0.5)
 
    fig.update_layout(
        **_CL, height=280,
        yaxis=dict(range=[-1.1, 1.1], showgrid=True, gridcolor=BORDER,
                   zeroline=False, title="Valence"),
        xaxis=dict(showgrid=False, tickfont=dict(size=10)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0),
    )
    return fig

#  Section divider HTML 
def _section_header(icon: str, title: str, sub: str = "") -> str:
    sub_html = f'<div style="color:{MUTED};font-size:12px;margin-top:2px;">{sub}</div>' if sub else ""
    return f"""
<div style="padding:18px 0 10px;">
  <div style="font-size:20px;font-weight:800;letter-spacing:-0.3px;color:#E8EAF6;">
    {icon} {title}
  </div>
  {sub_html}
</div>"""


def _divider() -> str:
    return f'<hr style="border:none;border-top:1px solid {BORDER};margin:12px 0 20px;">'


#  Gradio callbacks 

# - Live Attendance -
def cb_start_camera():
    res = _start_camera()
    if res == "already_running":
        return "⚠️ Camera is already running."
    time.sleep(0.3)
    with _state.lock:
        err = _state.cam_error
    return f"❌ {err}" if err else "🎥 Camera started."


def cb_stop_camera():
    _stop_camera()
    msg = "⬛ Camera stopped."
    with _state.lock:
        logger = _state.logger
    if logger and hasattr(logger, "export_summary"):
        try:
            logger.print_summary()
            summary_path = logger.export_summary()
            msg += f" Summary → {summary_path}"
        except Exception as e:
            msg += f" (summary export failed: {e})"
    return msg


def cb_load_models(model_choice: str):
    with _state.lock:
        if _state.models_loading:
            return "⏳ Already loading…"
    path = MODEL_CHOICES.get(model_choice, list(MODEL_CHOICES.values())[0])
    return _load_models_safe(path)


def cb_cam_status():
    with _state.lock:
        running = _state.cam_running
        fps     = _state.cam_fps
        loading = _state.models_loading
        models  = _state.models
        err     = _state.model_load_error

    parts = []
    if running:
        parts.append(f'<span style="color:{SUCCESS};font-weight:700;">🎥 Live · {fps:.0f} fps</span>')
    else:
        parts.append(f'<span style="color:{MUTED};">⬛ Camera off</span>')

    if models is not None:
        parts.append(f'<span style="color:{SUCCESS};margin-left:12px;">✓ Models ready</span>')
    elif loading:
        parts.append(f'<span style="color:{WARN};margin-left:12px;">⏳ Loading models…</span>')
    elif err:
        parts.append(f'<span style="color:{DANGER};margin-left:12px;">✗ {err[:60]}</span>')
    else:
        parts.append(f'<span style="color:{MUTED};margin-left:12px;">○ Models not loaded</span>')

    return "  ".join(parts)


# - Register Face -
def _any_to_bgr(item):
    """
    Convert whatever Gradio gives us into a BGR numpy array.
    Handles: str path, pathlib.Path, dict with 'name'/'path' key (Gradio File),
             object with .name, numpy ndarray (RGB), PIL Image.
    Returns None on failure.
    """
    if item is None:
        return None

    # Resolve to filepath string first
    path = None
    if isinstance(item, dict):
        path = item.get("path") or item.get("name") or item.get("tmp_path")
    elif hasattr(item, "path") and isinstance(getattr(item, "path"), str):
        path = item.path
    elif hasattr(item, "name") and isinstance(getattr(item, "name"), str):
        path = item.name
    elif isinstance(item, str):
        path = item
    elif hasattr(item, "__fspath__"):  # pathlib.Path
        path = str(item)

    if path:
        if not CV2_OK:
            return None
        bgr = cv2.imread(str(path))
        return bgr  # BGR, or None if unreadable

    # numpy array - assumed RGB from gr.Image type='numpy'
    if CV2_OK and isinstance(item, np.ndarray):
        if item.ndim == 2:
            return cv2.cvtColor(item, cv2.COLOR_GRAY2BGR)
        if item.shape[-1] == 4:
            return cv2.cvtColor(item, cv2.COLOR_RGBA2BGR)
        return cv2.cvtColor(item, cv2.COLOR_RGB2BGR)

    # PIL Image fallback
    try:
        arr = np.array(item.convert("RGB"))
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if CV2_OK else arr
    except Exception:
        return None


def cb_enroll(name: str):
    """
    Enroll using all images in _reg_buffer.
    Returns (reg_msg, registry_html, queue_label_html).
    """
    _q_empty = f'<span style="color:{MUTED};font-size:12px;">Queue: 0 images</span>'

    if not name.strip():
        return "❌ Please enter an employee name.", _registry_html(), gr.update()
    if not MODELS_OK:
        return f"❌ ML libraries not available: {_import_error[:80]}", _registry_html(), gr.update()
    with _state.lock:
        models_snap = _state.models
    if models_snap is None:
        return "❌ Load AI models first (use ⚡ Load AI Models above).", _registry_html(), gr.update()
    if not CV2_OK:
        return "❌ OpenCV not installed.", _registry_html(), gr.update()

    frames = list(_reg_buffer)
    if not frames:
        return "❌ Queue is empty — use 📸 Auto-Capture or ➕ Add Upload to Queue first.", _registry_html(), gr.update()

    bgr_list = []
    for f in frames:
        b = cv2.cvtColor(f, cv2.COLOR_RGB2BGR) if isinstance(f, np.ndarray) else _any_to_bgr(f)
        if b is not None:
            bgr_list.append(b)

    if not bgr_list:
        return "❌ Could not read any of the queued images.", _registry_html(), gr.update()

    try:
        ok, report = enroll(name.strip(), bgr_list, models_snap["face"], models_snap["device"])
        if not ok:
            reason = report if report else "No face detected — try better lighting or a clearer photo."
            return f"❌ {reason}", _registry_html(), gr.update()
        _reg_buffer.clear()
        msg = f"✅ {name.strip()} registered!\n\n{report}"
        return msg, _registry_html(), _q_empty
    except Exception as e:
        return f"❌ Enrolment failed: {e}", _registry_html(), gr.update()


def _registry_html() -> str:
    db = load_registry()
    if not db:
        return f'<div style="color:{MUTED};padding:14px 0;font-size:13px;">No faces registered yet.</div>'
    rows = ""
    for n, embs in db.items():
        count = len(embs) if isinstance(embs, list) else 1
        rows += f"""
<div style="display:flex;align-items:center;gap:12px;background:{SURFACE};
     border-radius:8px;padding:9px 14px;margin-bottom:5px;font-size:13px;">
  <span style="flex:1;color:#E8EAF6;font-weight:600;">{n}</span>
  <span style="color:{MUTED};font-size:11px;">{count} embedding(s)</span>
  <span style="display:inline-block;padding:2px 10px;border-radius:20px;font-size:11px;
        font-weight:700;background:rgba(0,230,118,.12);color:{SUCCESS};
        border:1px solid {SUCCESS};">✓</span>
</div>"""
    return rows


def cb_get_registry():
    return _registry_html()


def cb_clear_registry():
    save_registry({})
    return "✅ Registry cleared.", _registry_html()


def cb_auto_capture(n_frames: int):
    """Generator: 3-2-1 countdown per frame, captures from the register camera stream."""
    n_frames = int(n_frames)

    def _q_html(n):
        color = SUCCESS if n >= 3 else WARN
        tip   = "" if n >= 3 else " (3+ recommended for quality scoring)"
        return f'<span style="color:{color};font-size:12px;">✓ Queue: {n} image(s){tip}</span>'

    if not CV2_OK:
        yield f'<span style="color:{DANGER};font-size:13px;">❌ OpenCV not available</span>', gr.update()
        return

    with _state.lock:
        running = _state.cam_running
    if not running:
        yield (f'<span style="color:{DANGER};font-size:13px;">'
               f'❌ Camera not running — click ▶ Start Camera first</span>'), gr.update()
        return

    for i in range(n_frames):
        for c in [3, 2, 1]:
            yield (
                f'<span style="color:{ACCENT};font-size:20px;font-weight:800;">'
                f'📸 Frame {i+1}/{n_frames} &nbsp; {c}...</span>',
                gr.update(),
            )
            time.sleep(1.0)

        with _state.lock:
            frame = _state.latest_frame

        if frame is not None:
            _reg_buffer.append(frame.copy())
            n = len(_reg_buffer)
            yield (
                f'<span style="color:{SUCCESS};font-size:15px;font-weight:700;">'
                f'✓ Frame {i+1}/{n_frames} captured!</span>',
                _q_html(n),
            )
        else:
            yield (
                f'<span style="color:{WARN};font-size:13px;">'
                f'⚠ Frame {i+1} missed — camera not ready</span>',
                gr.update(),
            )

        if i < n_frames - 1:
            time.sleep(0.8)

    n = len(_reg_buffer)
    yield (
        f'<span style="color:{SUCCESS};font-size:14px;font-weight:700;">'
        f'✅ Done! {n} frame(s) in queue — click ✅ Register to enrol.</span>',
        _q_html(n),
    )



# - Dashboard -
def cb_dashboard(sel_date_str: str):
    today = sel_date_str or date.today().isoformat()

    attendees    = _stats_mod.attendance_summary(today)
    emotions     = _stats_mod.emotion_distribution(today)
    daily        = _stats_mod.daily_counts(7)
    hourly       = _stats_mod.arrivals_by_hour(today)
    avg_conf     = _stats_mod.avg_confidence(today)
    events_today = get_events(today)
    total_week   = sum(c for _, c in daily)
    spoof_count  = sum(1 for e in events_today if not e.get("liveness_passed"))

    sent         = team_sentiment_summary(today)
    sent_color   = sent["color"]
    sent_label   = sent["label"]
    sent_valence = sent["mean_valence"]

    # KPI HTML
    kpi = f"""
<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:16px;">
  <div style="background:{CARD};border:1px solid {BORDER};border-radius:12px;padding:16px 20px;">
    <div style="color:{MUTED};font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.8px;">👥 Present</div>
    <div style="color:{ACCENT};font-size:28px;font-weight:800;">{len(attendees)}</div>
  </div>
  <div style="background:{CARD};border:1px solid {BORDER};border-radius:12px;padding:16px 20px;">
    <div style="color:{MUTED};font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.8px;">📅 This Week</div>
    <div style="color:{ACCENT};font-size:28px;font-weight:800;">{total_week}</div>
  </div>
  <div style="background:{CARD};border:1px solid {BORDER};border-radius:12px;padding:16px 20px;">
    <div style="color:{MUTED};font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.8px;">📈 Avg Confidence</div>
    <div style="color:{ACCENT};font-size:28px;font-weight:800;">{avg_conf:.0%}</div>
  </div>
  <div style="background:{CARD};border:1px solid {BORDER};border-radius:12px;padding:16px 20px;">
    <div style="color:{MUTED};font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.8px;">🚨 Spoof Attempts</div>
    <div style="color:{DANGER};font-size:28px;font-weight:800;">{spoof_count}</div>
  </div>
    <div style="background:{CARD};border:1px solid {BORDER};border-radius:12px;padding:16px 20px;">
    <div style="color:{MUTED};font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.8px;">💬 Team Sentiment</div>
    <div style="color:{sent_color};font-size:20px;font-weight:800;">{sent_label}</div>
    <div style="color:{MUTED};font-size:11px;margin-top:2px;">valence {sent_valence:+.2f}</div>
  </div>
</div>

"""

    # Attendance log HTML
    first_seen: dict = {}
    for ev in events_today:
        if ev["person_id"] not in first_seen:
            first_seen[ev["person_id"]] = ev

    if first_seen:
        header = f"""
<div style="display:grid;grid-template-columns:3fr 2fr 2fr 2fr 2fr;gap:8px;
     padding:6px 14px;margin-bottom:4px;">
  {''.join(f'<span style="color:{MUTED};font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.8px;">{h}</span>'
           for h in ['Name','Arrived','Emotion','Liveness','Confidence'])}
</div>"""
        body = ""
        for n in sorted(first_seen):
            ev   = first_seen[n]
            ts   = datetime.fromisoformat(ev["timestamp"]).strftime("%H:%M")
            emo  = ev.get("emotion") or "-"
            live = ev.get("liveness_passed")
            conf = ev.get("confidence", 0)
            emo_html  = _emotion_badge_html(emo) if emo != "-" else f'<span style="color:{MUTED};">-</span>'
            live_html = _liveness_badge_html(None if live is None else bool(live))
            body += f"""
<div style="display:grid;grid-template-columns:3fr 2fr 2fr 2fr 2fr;gap:8px;
     background:{SURFACE};border-radius:8px;padding:9px 14px;margin-bottom:5px;
     font-size:13px;align-items:center;">
  <b style="color:#E8EAF6;">{n}</b>
  <span style="color:{MUTED};">{ts}</span>
  {emo_html}
  {live_html}
  <span style="color:{SUCCESS};font-weight:700;">{conf:.0%}</span>
</div>"""
        log_html = header + body
    else:
        log_html = f'<div style="color:{MUTED};padding:20px;text-align:center;">No attendance recorded for {today}.</div>'

    bar_fig    = _bar_chart(daily)
    donut_fig  = _donut_chart(emotions)
    arrivals_fig = _arrivals_chart(hourly)
    valence_fig  = _valence_timeline_chart(attendees, today)

    return kpi, log_html, bar_fig, donut_fig, arrivals_fig, valence_fig


def cb_export_csv():
    if not PD_OK:
        return None
    evs = get_events()
    if not evs:
        return None
    df = pd.DataFrame(evs)
    out = Path("logs") / "attendance_export.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(out), index=False)
    return str(out)


# - Settings -
def cb_apply_settings(thr, enf, lthr, model_choice, cam_idx, log_gate, emo_conf, log_live):
    with _state.lock:
        _state.threshold            = thr
        _state.enforce_liveness     = enf
        _state.liveness_threshold   = lthr
        _state.cam_index            = int(cam_idx)
        _state.log_gate_enabled     = bool(log_gate)
        _state.min_emotion_conf_log = float(emo_conf)
        _state.require_liveness_log = bool(log_live)
    _sync_logger_gates()
    path = MODEL_CHOICES.get(model_choice, list(MODEL_CHOICES.values())[0])
    with _state.lock:
        _state.face_model_path = path
    gate_note = f"log gate on (emo≥{emo_conf:.2f})" if log_gate else "log gate off"
    return (
        f"✅ Applied - sim {thr:.2f} · liveness {lthr:.2f} · camera {int(cam_idx)} · {gate_note}"
    )


def cb_reload_models(model_choice):
    path = MODEL_CHOICES.get(model_choice, list(MODEL_CHOICES.values())[0])
    return _load_models_safe(path)


def cb_test_camera(cam_idx):
    if not CV2_OK:
        return gr.update(visible=False), "❌ OpenCV not installed."
    cap = cv2.VideoCapture(int(cam_idx))
    if not cap.isOpened():
        return gr.update(visible=False), f"❌ Cannot open camera {int(cam_idx)}."
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return gr.update(visible=False), "❌ Opened but couldn't read a frame."
    return gr.update(value=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), visible=True), "✅ Camera OK."


def cb_seed_demo():
    try:
        init_db()
        today_d = date.today()
        rows    = 0
        db_path = Path("nathan_innovation/attendance.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(db_path)) as conn:
            for off in range(6, -1, -1):
                day = today_d - timedelta(days=off)
                if day.weekday() >= 5:
                    continue
                present = random.sample(EMPLOYEES, k=random.randint(3, len(EMPLOYEES)))
                for emp in present:
                    mo  = random.randint(0, 120)
                    arr = datetime(
                        day.year, day.month, day.day,
                        8 + mo // 60, mo % 60, random.randint(0, 59),
                        tzinfo=timezone.utc,
                    )
                    emo  = random.choices(EMOTIONS, weights=EMOTION_W)[0]
                    conf = round(random.uniform(0.62, 0.97), 4)
                    live = 1 if random.random() > 0.05 else 0
                    conn.execute(
                        "INSERT INTO attendance_events "
                        "(timestamp,person_id,emotion,liveness_passed,confidence) VALUES (?,?,?,?,?)",
                        (arr.strftime("%Y-%m-%dT%H:%M:%S"), emp, emo, live, conf),
                    )
                    rows += 1
        return f"✅ Seeded {rows} demo events."
    except Exception as e:
        return f"❌ {e}"


def cb_clear_db():
    try:
        db_path = Path("nathan_innovation/attendance.db")
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("DELETE FROM attendance_events")
        return "✅ Attendance DB cleared."
    except Exception as e:
        return f"❌ {e}"


#  CSS 
CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700;800&family=Space+Mono:wght@400;700&display=swap');

body, .gradio-container {{
    font-family: 'DM Sans', 'Segoe UI', sans-serif !important;
    background: {BG} !important;
    color: #E8EAF6 !important;
}}
.gradio-container {{ max-width:1400px !important; margin:0 auto !important; }}
button.primary {{
    background: linear-gradient(135deg,{ACCENT2},{ACCENT}) !important;
    color: {BG} !important; border:none !important; border-radius:8px !important;
    font-weight:700 !important; font-size:13px !important;
}}
button.secondary {{
    background: {CARD} !important; color:{ACCENT} !important;
    border:1px solid {BORDER} !important; border-radius:8px !important;
    font-weight:600 !important; font-size:13px !important;
}}
button.secondary:hover {{ border-color:{ACCENT} !important; background:{SURFACE} !important; }}
input, textarea, select {{
    background:{SURFACE} !important; color:#E8EAF6 !important;
    border:1px solid {BORDER} !important; border-radius:8px !important;
    font-family:'DM Sans',sans-serif !important; font-size:14px !important;
}}
input:focus, textarea:focus {{
    border-color:{ACCENT} !important;
    box-shadow:0 0 0 2px rgba(0,229,255,0.15) !important;
}}
.panel, .form {{
    background:{SURFACE} !important; border:1px solid {BORDER} !important;
    border-radius:12px !important;
}}
img {{ border-radius:10px; }}
::-webkit-scrollbar {{ width:5px; height:5px; }}
::-webkit-scrollbar-track {{ background:{SURFACE}; }}
::-webkit-scrollbar-thumb {{ background:{BORDER}; border-radius:3px; }}
::-webkit-scrollbar-thumb:hover {{ background:{ACCENT}; }}
.app-header {{
    padding:18px 24px 14px; border-bottom:1px solid {BORDER};
    margin-bottom:18px; display:flex; align-items:center; gap:14px;
}}
.logo-text {{
    font-size:24px; font-weight:800; letter-spacing:-0.5px;
    background:linear-gradient(90deg,{ACCENT},{ACCENT2});
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}}
.logo-sub {{ font-size:12px; color:{MUTED}; font-family:'Space Mono',monospace; }}
label span {{
    color:{MUTED} !important; font-size:12px !important;
    font-weight:600 !important; text-transform:uppercase; letter-spacing:0.6px;
}}
.js-plotly-plot {{ background:transparent !important; }}
.settings-box {{
    background:{SURFACE}; border:1px solid {BORDER}; border-radius:12px;
    padding:18px 20px; margin:10px 0;
}}
"""

HEADER_HTML = f"""
<div class="app-header">
  <div>
    <div class="logo-text">◈ FaceAttend</div>
    <div class="logo-sub">Unified Console · AI Face Recognition &amp; Attendance</div>
  </div>
</div>
"""

#  BUILD UI 
with gr.Blocks(title="FaceAttend - Unified Console", css=CSS) as demo:

    gr.HTML(HEADER_HTML)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 - LIVE ATTENDANCE
    # ══════════════════════════════════════════════════════════════════════════
    gr.HTML(_section_header("🎥", "Live Attendance", "Real-time face recognition stream"))

    status_bar = gr.HTML(value=cb_cam_status())

    with gr.Row():
        start_btn      = gr.Button("▶ Start Camera",  variant="primary",   scale=1)
        stop_btn       = gr.Button("■ Stop Camera",   variant="secondary", scale=1)
        model_dd       = gr.Dropdown(
            choices=list(MODEL_CHOICES.keys()),
            value=list(MODEL_CHOICES.keys())[0],
            label="Face Recognition Model", scale=3,
        )
        load_model_btn = gr.Button("⚡ Load AI Models", variant="primary", scale=1)

    live_ctrl_msg = gr.Textbox(label="Camera Status", interactive=False, max_lines=1)

    with gr.Row():
        with gr.Column(scale=3):
            live_feed = gr.Image(
                label="Live Feed",
                streaming=True,
                show_label=False,
                height=480,
            )
        with gr.Column(scale=2):
            results_html = gr.HTML(value=_results_html({}))

    gr.HTML(_divider())

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 - REGISTER FACE
    # ══════════════════════════════════════════════════════════════════════════
    gr.HTML(_section_header("👤", "Register Face", "Enrol a new employee into the face registry"))

    with gr.Row():
        with gr.Column(scale=1):
            reg_name = gr.Textbox(label="Full Name", placeholder="e.g. Jane Smith")
            gr.HTML(f"""
<div style="background:{CARD};border:1px solid {BORDER};border-radius:10px;
     padding:13px 16px;font-size:12px;color:#90A4AE;margin-top:6px;">
  <b style="color:#E8EAF6;">How to register</b><br>
  1. Click <b style="color:#E8EAF6;">▶ Start Camera</b> to see the live feed<br>
  2. Set frames and click <b style="color:#E8EAF6;">📸 Auto-Capture</b><br>
  3. Or upload a photo → <b style="color:#E8EAF6;">➕ Add to Queue</b><br>
  4. Click <b style="color:#E8EAF6;">✅ Register</b> when queue is ready<br>
  5. Click <b style="color:#E8EAF6;">■ Stop Camera</b> when done
</div>""")

        with gr.Column(scale=2):
            reg_preview = gr.Image(
                label="Register Camera", streaming=True,
                show_label=False, height=320,
            )
            with gr.Row():
                reg_start_btn = gr.Button("▶ Start Camera", variant="primary",   scale=1)
                reg_stop_btn  = gr.Button("■ Stop Camera",  variant="secondary", scale=1)
            reg_cam_msg = gr.Textbox(label="", interactive=False, max_lines=1,
                                     show_label=False, placeholder="Camera status…")
            gr.HTML(f'<hr style="border:none;border-top:1px solid {BORDER};margin:8px 0;">')
            with gr.Row():
                n_frames_slider = gr.Slider(
                    minimum=3, maximum=10, value=5, step=1,
                    label="Frames to capture", scale=3,
                )
                auto_cap_btn = gr.Button("📸 Auto-Capture", variant="primary", scale=2)
            auto_cap_status = gr.HTML(
                value=f'<span style="color:{MUTED};font-size:12px;">Start camera, then click Auto-Capture</span>'
            )
            gr.HTML(f'<hr style="border:none;border-top:1px solid {BORDER};margin:8px 0;">')
            reg_upload = gr.Image(
                label="Or upload a photo (JPG / PNG)",
                sources=["upload"],
                type="numpy",
            )
            with gr.Row():
                add_queue_btn   = gr.Button("➕ Add Upload to Queue", variant="secondary", scale=2)
                clear_queue_btn = gr.Button("🗑 Clear Queue",          variant="secondary", scale=1)
            queue_label = gr.HTML(
                value=f'<span style="color:{MUTED};font-size:12px;">Queue: 0 images</span>'
            )
            with gr.Row():
                reg_btn   = gr.Button("✅ Register", variant="primary",   scale=2)
                clear_btn = gr.Button("✖ Clear All", variant="secondary", scale=1)

    reg_msg = gr.Textbox(label="Registration Status", interactive=False, max_lines=15)

    gr.HTML(f'<div style="color:{MUTED};font-size:12px;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:.8px;margin:10px 0 6px;">Registered Identities</div>')
    registry_html = gr.HTML(value=_registry_html())
    refresh_registry_btn = gr.Button("🔄 Refresh Registry",        variant="secondary", scale=0)
    clear_reg_btn        = gr.Button("🗑 Clear All Registrations", variant="secondary", scale=0)
    clear_reg_msg        = gr.Textbox(label="", interactive=False, max_lines=1, visible=False)

    gr.HTML(_divider())

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3 - DASHBOARD
    # ══════════════════════════════════════════════════════════════════════════
    gr.HTML(_section_header("📊", "Dashboard",
                            f"Attendance analytics · {datetime.now().strftime('%A, %d %B %Y')}"))

    with gr.Row():
        dash_date    = gr.Textbox(
            label="View Date (YYYY-MM-DD)", value=date.today().isoformat(), scale=2)
        dash_refresh = gr.Button("🔄 Refresh Dashboard", variant="primary", scale=1)
        dash_export  = gr.Button("⬇ Export CSV",         variant="secondary", scale=1)

    dash_kpi_html  = gr.HTML()
    dash_log_html  = gr.HTML()
    dash_file      = gr.File(label="CSV Export", visible=False)

    with gr.Row():
        dash_bar_plot      = gr.Plot(label="📅 Daily Attendance (7 Days)")
        dash_donut_plot    = gr.Plot(label="😊 Emotion Distribution")

    dash_arrivals_plot = gr.Plot(label="⏰ Arrivals by Hour")
    dash_valence_plot  = gr.Plot(label="💬 Per-Person Sentiment Timeline")

    gr.HTML(_divider())

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4 - SETTINGS
    # ══════════════════════════════════════════════════════════════════════════
    gr.HTML(_section_header("⚙️", "Settings", "Models, thresholds, camera & data management"))

    gr.HTML(f'<div class="settings-box">')
    gr.HTML(f'<div style="font-size:14px;font-weight:700;color:#E8EAF6;margin-bottom:14px;">🤖 Model & Thresholds</div>')

    with gr.Row():
        s_model = gr.Dropdown(
            choices=list(MODEL_CHOICES.keys()),
            value=list(MODEL_CHOICES.keys())[0],
            label="Active Face Recognition Model", scale=3,
        )
        s_reload = gr.Button("🔄 Reload Models", variant="primary", scale=1)

    with gr.Row():
        s_thr  = gr.Slider(0.30, 0.95, value=0.55, step=0.01, label="Similarity Threshold", scale=2)
        s_enf  = gr.Checkbox(label="Enforce Liveness Gate", value=False, scale=1)
        s_lthr = gr.Slider(0.10, 0.90, value=0.40, step=0.01, label="Liveness Threshold", scale=2)

    gr.HTML(f'<div style="font-size:14px;font-weight:700;color:#E8EAF6;margin:14px 0 8px;">📋 Quality-Aware Logging</div>')
    with gr.Row():
        s_log_gate = gr.Checkbox(label="Gate attendance logs (similarity + emotion confidence)", value=True, scale=2)
        s_log_live = gr.Checkbox(label="Require liveness pass to log", value=False, scale=1)
        s_emo_conf = gr.Slider(
            0.0, 0.90, value=0.20, step=0.05,
            label="Min emotion confidence to log", scale=2,
        )

    gr.HTML(f'<div style="font-size:14px;font-weight:700;color:#E8EAF6;margin:16px 0 10px;">📷 Camera</div>')
    with gr.Row():
        s_cam_idx = gr.Number(label="Camera Index (0 = default)", value=0, precision=0, scale=1)
        s_apply   = gr.Button("✅ Apply All Settings", variant="primary", scale=1)
        s_test    = gr.Button("📷 Test Camera",        variant="secondary", scale=1)

    s_test_img  = gr.Image(label="Test Frame", height=220, visible=False)
    s_msg       = gr.Textbox(label="Settings Status", interactive=False, max_lines=1)

    # Model info table
    if PD_OK:
        model_df = pd.DataFrame([
            {"Model": "A - Classification", "AUC": 0.8924, "Embedding dim": 512},
            {"Model": "B - Triplet Loss",   "AUC": 0.9025, "Embedding dim": 128},
            {"Model": "C - ArcFace ★",      "AUC": 0.9250, "Embedding dim": 512},
        ])
        gr.Dataframe(value=model_df, label="Model Performance Summary",
                     interactive=False, wrap=True)

    gr.HTML(f'<div style="font-size:14px;font-weight:700;color:#E8EAF6;margin:16px 0 10px;">🛡 Anti-Spoofing</div>')
    gr.HTML(f"""
<div style="background:{CARD};border:1px solid {BORDER};border-radius:10px;
     padding:13px 16px;font-size:12px;color:#90A4AE;">
  <b style="color:#E8EAF6;">Anti-spoof model</b> · MobileNetV2 on AxonData<br>
  <b style="color:#E8EAF6;">Behavioral fusion</b> · head-turn + eye-blink + CNN<br>
  <code>saved_models/antispoof_v3.h5</code>
</div>""")

    gr.HTML(f'<div style="font-size:14px;font-weight:700;color:#E8EAF6;margin:16px 0 10px;">🗄 Data Management</div>')
    with gr.Row():
        data_seed  = gr.Button("🌱 Seed Demo Data (7 days)",    variant="secondary", scale=1)
        data_clear = gr.Button("🗑 Clear Attendance DB",         variant="secondary", scale=1)
    data_msg = gr.Textbox(label="Data Status", interactive=False, max_lines=1)

    gr.HTML("</div>")  # end settings-box

    gr.HTML(
        f'<div style="color:{MUTED};font-size:11px;padding:14px 0 6px;text-align:right;">'
        f'◈ FaceAttend · COS30082 Applied ML · Unified Gradio Console</div>'
    )

    #  Wire events 

    # Live Attendance
    start_btn.click(cb_start_camera, outputs=live_ctrl_msg)
    stop_btn.click(cb_stop_camera,   outputs=live_ctrl_msg)
    load_model_btn.click(cb_load_models, inputs=model_dd, outputs=live_ctrl_msg)
    gr.Timer(value=2.0).tick(cb_cam_status, outputs=status_bar)
    demo.load(_camera_stream, outputs=[live_feed, results_html], concurrency_limit=None)

    # Register Face
    def cb_add_to_queue(img):
        if img is None:
            return f'<span style="color:{WARN};font-size:12px;">⚠ No image — upload or capture one first</span>'
        _reg_buffer.append(img.copy() if isinstance(img, np.ndarray) else img)
        n     = len(_reg_buffer)
        color = SUCCESS if n >= 3 else WARN
        tip   = "" if n >= 3 else " (3+ recommended for quality scoring)"
        return f'<span style="color:{color};font-size:12px;">✓ Queue: {n} image(s){tip}</span>'

    def cb_clear_queue():
        _reg_buffer.clear()
        return f'<span style="color:{MUTED};font-size:12px;">Queue: 0 images — add multiple for best quality scoring</span>'

    demo.load(_reg_preview_stream, outputs=reg_preview, concurrency_limit=None)
    reg_start_btn.click(cb_start_camera, outputs=reg_cam_msg)
    reg_stop_btn.click(cb_stop_camera,   outputs=reg_cam_msg)

    auto_cap_btn.click(
        cb_auto_capture,
        inputs=n_frames_slider,
        outputs=[auto_cap_status, queue_label],
    )

    add_queue_btn.click(cb_add_to_queue, inputs=reg_upload, outputs=queue_label)
    clear_queue_btn.click(cb_clear_queue, outputs=queue_label)

    reg_btn.click(
        cb_enroll,
        inputs=reg_name,
        outputs=[reg_msg, registry_html, queue_label],
    )

    def _clear_reg():
        _reg_buffer.clear()
        empty = f'<span style="color:{MUTED};font-size:12px;">Queue: 0 images</span>'
        return None, "", empty

    clear_btn.click(_clear_reg, outputs=[reg_upload, reg_name, queue_label])
    refresh_registry_btn.click(cb_get_registry, outputs=registry_html)
    clear_reg_btn.click(
        cb_clear_registry,
        outputs=[clear_reg_msg, registry_html],
    )

    # Dashboard
    def _do_dashboard(d):
        kpi, log, bar, donut, arr, valence  = cb_dashboard(d)
        return kpi, log, bar, donut, arr, valence

    dash_refresh.click(
        _do_dashboard,
        inputs=dash_date,
        outputs=[dash_kpi_html, dash_log_html, dash_bar_plot, dash_donut_plot, dash_arrivals_plot, dash_valence_plot],
    )

    def _do_export():
        path = cb_export_csv()
        if path:
            return gr.update(value=path, visible=True)
        return gr.update(visible=False)

    dash_export.click(_do_export, outputs=dash_file)

    # Load dashboard on page open
    demo.load(
        _do_dashboard,
        inputs=dash_date,
        outputs=[dash_kpi_html, dash_log_html, dash_bar_plot, dash_donut_plot, dash_arrivals_plot, dash_valence_plot],
    )

    # Settings
    s_apply.click(
        cb_apply_settings,
        inputs=[s_thr, s_enf, s_lthr, s_model, s_cam_idx, s_log_gate, s_emo_conf, s_log_live],
        outputs=s_msg,
    )
    s_reload.click(cb_reload_models, inputs=s_model, outputs=s_msg)
    s_test.click(cb_test_camera, inputs=s_cam_idx, outputs=[s_test_img, s_msg])
    data_seed.click(cb_seed_demo,  outputs=data_msg)
    data_clear.click(cb_clear_db,  outputs=data_msg)


#  Launch 
if __name__ == "__main__":
    demo.launch(share=False, show_error=True)