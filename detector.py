"""
Core Drowsiness Detection Engine using YOLOv8
Provides cross-platform inference, stateful driver fatigue tracking,
visual overlays, and structured telemetry for web applications and desktop clients.
"""

import os
import logging
import urllib.request
import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional, List

# Configure logging
logger = logging.getLogger("DrowsinessDetector")
logger.setLevel(logging.INFO)

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    logger.warning("Ultralytics not installed. Please install with 'pip install ultralytics'.")

# Remote URLs for weights if not present or if Git LFS pointers
REMOTE_WEIGHTS = {
    "eye": "https://media.githubusercontent.com/media/rishavraj69/Drousiness-Detection-System-Using-YOLOv8/main/model/eye/runs/detect/train3/weights/best.pt",
    "yawn": "https://media.githubusercontent.com/media/rishavraj69/Drousiness-Detection-System-Using-YOLOv8/main/model/yawn/runs/detect/train/weights/best.pt"
}


class DrowsinessDetector:
    """
    Stateful Drowsiness Detector using YOLOv8 models for eye and yawn detection.
    """

    def __init__(
        self,
        eye_model_path: Optional[str] = None,
        yawn_model_path: Optional[str] = None,
        eye_closure_threshold: int = 40,
        yawn_threshold: int = 50,
        tiredness_threshold: int = 3,
        auto_fallback: bool = True
    ):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Default model paths
        self.eye_model_path = eye_model_path or os.path.join(
            self.base_dir, "model", "eye", "runs", "detect", "train3", "weights", "best.pt"
        )
        self.yawn_model_path = yawn_model_path or os.path.join(
            self.base_dir, "model", "yawn", "runs", "detect", "train", "weights", "best.pt"
        )
        
        # Thresholds
        self.eye_closure_threshold = eye_closure_threshold
        self.yawn_threshold = yawn_threshold
        self.tiredness_threshold = tiredness_threshold
        self.auto_fallback = auto_fallback
        
        # Stateful counters
        self.frame_count = 0
        self.eye_closure_count = 0
        self.yawn_count = 0
        self.tiredness_level = 0
        self.is_drowsy = False
        self.is_yawning = False
        self.eyes_closed = False
        self.alert_status = "NORMAL"  # "NORMAL", "WARNING", "DANGER"
        
        # Loaded model references
        self.eye_model = None
        self.yawn_model = None
        self.fallback_model = None
        self.using_fallback = False
        
        self._load_models()

    def _is_lfs_pointer(self, file_path: str) -> bool:
        """Check if a file is missing or an un-downloaded Git LFS pointer."""
        if not os.path.exists(file_path):
            return True
        try:
            if os.path.getsize(file_path) < 1024 * 1024:  # Less than 1MB is likely a pointer
                with open(file_path, "r", errors="ignore") as f:
                    content = f.read(100)
                    if "git-lfs" in content or "oid sha256" in content:
                        return True
        except Exception:
            pass
        return False

    def _ensure_weights(self, file_path: str, weight_type: str):
        """Auto-download weights if missing or Git LFS pointer."""
        if self._is_lfs_pointer(file_path):
            url = REMOTE_WEIGHTS.get(weight_type)
            if url:
                try:
                    logger.info(f"Downloading real {weight_type} model weights from {url}...")
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    urllib.request.urlretrieve(url, file_path)
                    logger.info(f"Successfully downloaded {weight_type} weights to {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to auto-download {weight_type} weights: {e}")

    def _load_models(self):
        """Load YOLO models with fallback handling."""
        if not ULTRALYTICS_AVAILABLE:
            logger.warning("Ultralytics is not available. Detector operating in mock mode.")
            return

        # Ensure actual weights exist
        self._ensure_weights(self.eye_model_path, "eye")
        self._ensure_weights(self.yawn_model_path, "yawn")

        # 1. Attempt loading Eye Model
        if os.path.exists(self.eye_model_path) and not self._is_lfs_pointer(self.eye_model_path):
            try:
                self.eye_model = YOLO(self.eye_model_path)
                logger.info(f"Loaded custom eye detection model: {self.eye_model_path}")
            except Exception as e:
                logger.warning(f"Could not load eye model at {self.eye_model_path}: {e}")
                self.eye_model = None

        # 2. Attempt loading Yawn Model
        if os.path.exists(self.yawn_model_path) and not self._is_lfs_pointer(self.yawn_model_path):
            try:
                self.yawn_model = YOLO(self.yawn_model_path)
                logger.info(f"Loaded custom yawn detection model: {self.yawn_model_path}")
            except Exception as e:
                logger.warning(f"Could not load yawn model at {self.yawn_model_path}: {e}")
                self.yawn_model = None

        # 3. Fallback to standard YOLOv8 model only if custom weights failed
        if self.eye_model is None and self.yawn_model is None and self.auto_fallback:
            try:
                logger.info("Initializing fallback YOLOv8 model (yolov8n.pt)...")
                self.fallback_model = YOLO("yolov8n.pt")
                self.using_fallback = True
            except Exception as e:
                logger.warning(f"Failed to load fallback YOLOv8 model: {e}")
        else:
            self.using_fallback = False

    def reset_state(self):
        """Reset all tracking counters to initial state."""
        self.frame_count = 0
        self.eye_closure_count = 0
        self.yawn_count = 0
        self.tiredness_level = 0
        self.is_drowsy = False
        self.is_yawning = False
        self.eyes_closed = False
        self.alert_status = "NORMAL"

    def process_frame(
        self,
        frame: np.ndarray,
        conf_eye: float = 0.35,
        conf_yawn: float = 0.40,
        draw_overlay: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process a single image frame (BGR format) and return the annotated frame along with metrics.
        """
        if frame is None or frame.size == 0:
            return frame, self.get_metrics()

        self.frame_count += 1
        annotated_frame = frame.copy()
        
        detected_eyes_closed = False
        detected_eyes_open = False
        detected_yawn = False
        detections: List[Dict[str, Any]] = []

        # 1. Custom Eye Model Inference
        if self.eye_model is not None:
            try:
                results_eye = self.eye_model.track(frame, stream=False, conf=conf_eye, verbose=False, persist=True)
                for r in results_eye:
                    for box in r.boxes:
                        cls_id = int(box.cls[0]) if hasattr(box.cls, "__len__") else int(box.cls)
                        conf = float(box.conf[0]) if hasattr(box.conf, "__len__") else float(box.conf)
                        xyxy = [int(v) for v in (box.xyxy[0].tolist() if hasattr(box.xyxy, "__len__") else [])]
                        
                        # In eye dataset: class 0 = 'Closed', class 1 = 'Open'
                        label = "Closed" if cls_id == 0 else "Open"
                        if cls_id == 0:
                            detected_eyes_closed = True
                            box_color = (0, 0, 245)  # Bright Red for Closed
                        else:
                            detected_eyes_open = True
                            box_color = (0, 220, 60)  # Bright Green for Open
                            
                        detections.append({
                            "type": "eye",
                            "class": label,
                            "confidence": round(conf, 3),
                            "box": xyxy
                        })
                        
                        if draw_overlay and len(xyxy) == 4:
                            # Draw Eye Bounding Box
                            cv2.rectangle(annotated_frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), box_color, 2)
                            tag = f"Eye: {label} ({conf:.2f})"
                            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                            cv2.rectangle(annotated_frame, (xyxy[0], max(0, xyxy[1] - 20)), (xyxy[0] + tw + 6, max(20, xyxy[1])), box_color, -1)
                            cv2.putText(annotated_frame, tag, (xyxy[0] + 3, max(15, xyxy[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            except Exception as e:
                logger.error(f"Error in eye detection: {e}")

        # 2. Custom Yawn Model Inference
        if self.yawn_model is not None:
            try:
                results_yawn = self.yawn_model.track(frame, stream=False, conf=conf_yawn, verbose=False, persist=True)
                for r in results_yawn:
                    for box in r.boxes:
                        conf = float(box.conf[0]) if hasattr(box.conf, "__len__") else float(box.conf)
                        xyxy = [int(v) for v in (box.xyxy[0].tolist() if hasattr(box.xyxy, "__len__") else [])]
                        detected_yawn = True
                        box_color = (255, 140, 0)  # Orange / Magenta for Yawn
                        
                        detections.append({
                            "type": "yawn",
                            "class": "Yawn",
                            "confidence": round(conf, 3),
                            "box": xyxy
                        })
                        
                        if draw_overlay and len(xyxy) == 4:
                            # Draw Yawn Bounding Box
                            cv2.rectangle(annotated_frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), box_color, 3)
                            tag = f"YAWNING ({conf:.2f})"
                            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_DUPLEX, 0.6, 2)
                            cv2.rectangle(annotated_frame, (xyxy[0], max(0, xyxy[1] - 24)), (xyxy[0] + tw + 8, max(24, xyxy[1])), box_color, -1)
                            cv2.putText(annotated_frame, tag, (xyxy[0] + 4, max(18, xyxy[1] - 6)), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 2)
            except Exception as e:
                logger.error(f"Error in yawn detection: {e}")

        # 3. Fallback Model (if neither eye nor yawn custom models are available)
        if self.eye_model is None and self.yawn_model is None and self.fallback_model is not None:
            try:
                results = self.fallback_model(frame, conf=0.4, verbose=False)
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0]) if hasattr(box.cls, "__len__") else int(box.cls)
                        conf = float(box.conf[0]) if hasattr(box.conf, "__len__") else float(box.conf)
                        xyxy = [int(v) for v in (box.xyxy[0].tolist() if hasattr(box.xyxy, "__len__") else [])]
                        if cls_id == 0 and len(xyxy) == 4:
                            detections.append({
                                "type": "person",
                                "class": "Driver Detected",
                                "confidence": round(conf, 3),
                                "box": xyxy
                            })
                            if draw_overlay:
                                cv2.rectangle(annotated_frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 128), 2)
                                cv2.putText(
                                    annotated_frame,
                                    f"Driver {conf:.2f}",
                                    (xyxy[0], max(xyxy[1] - 10, 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    (0, 255, 128),
                                    2
                                )
            except Exception as e:
                logger.error(f"Error in fallback model inference: {e}")

        # Update stateful counters
        self.eyes_closed = detected_eyes_closed
        self.is_yawning = detected_yawn

        if detected_eyes_closed:
            self.eye_closure_count += 2
        elif detected_eyes_open:
            self.eye_closure_count = max(0, self.eye_closure_count - 5)
            if self.eye_closure_count < 10:
                self.eye_closure_count = 0

        if detected_yawn:
            self.yawn_count += 1

        # Evaluate fatigue and drowsiness alerts
        if self.eye_closure_count > self.eye_closure_threshold or self.yawn_count > self.yawn_threshold or self.tiredness_level >= self.tiredness_threshold:
            self.is_drowsy = True
            self.alert_status = "DANGER"
        elif self.eye_closure_count > (self.eye_closure_threshold // 2) or self.yawn_count > (self.yawn_threshold // 2):
            self.is_drowsy = False
            self.alert_status = "WARNING"
        else:
            self.is_drowsy = False
            self.alert_status = "NORMAL"

        # Update tiredness level periodically
        if self.yawn_count > (self.yawn_threshold * 0.7) and self.frame_count % 30 == 0:
            self.tiredness_level += 1

        # Periodic resets to avoid unbounded drift
        if self.frame_count > 1800:
            self.frame_count = 0
            self.yawn_count = max(0, self.yawn_count - 20)
        if self.frame_count > 9000:
            self.tiredness_level = max(0, self.tiredness_level - 1)

        # Draw HUD overlay if requested
        if draw_overlay:
            annotated_frame = self._draw_hud(annotated_frame)

        metrics = self.get_metrics()
        metrics["detections"] = detections
        return annotated_frame, metrics

    def _draw_hud(self, frame: np.ndarray) -> np.ndarray:
        """Draw driver telemetry HUD overlay onto frame."""
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # Header banner styling
        status_colors = {
            "NORMAL": (46, 175, 80),    # Emerald Green (BGR)
            "WARNING": (0, 190, 245),   # Amber/Yellow (BGR)
            "DANGER": (34, 43, 230)     # Crimson Red (BGR)
        }
        status_color = status_colors.get(self.alert_status, (46, 175, 80))

        # Top banner background
        cv2.rectangle(overlay, (0, 0), (w, 65), (15, 15, 20), -1)
        # Status accent bar
        cv2.rectangle(overlay, (0, 0), (w, 5), status_color, -1)

        # Blend top banner
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Draw Status Badge
        status_text = f"STATUS: {self.alert_status}"
        if self.alert_status == "DANGER":
            status_text = "ALERT: DROWSINESS DETECTED! TAKE A BREAK!"
        elif self.alert_status == "WARNING":
            status_text = "WARNING: SIGNS OF FATIGUE DETECTED"

        cv2.putText(frame, status_text, (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.7, status_color, 2)

        # Telemetry info string
        eye_state_str = "CLOSED" if self.eyes_closed else "OPEN"
        yawn_state_str = "YES" if self.is_yawning else "NO"
        telemetry = f"Eyes: {eye_state_str} ({self.eye_closure_count}/{self.eye_closure_threshold}) | Yawn: {yawn_state_str} ({self.yawn_count}) | Fatigue: {self.tiredness_level}"
        cv2.putText(frame, telemetry, (20, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # Progress bar for Eye Closure (Bottom Right)
        bar_w, bar_h = 220, 16
        bx, by = w - bar_w - 20, 35
        cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (50, 50, 50), -1)
        ratio = min(1.0, self.eye_closure_count / max(1, self.eye_closure_threshold))
        fill_w = int(bar_w * ratio)
        fill_color = (0, 255, 0) if ratio < 0.5 else ((0, 200, 255) if ratio < 0.8 else (0, 0, 255))
        if fill_w > 0:
            cv2.rectangle(frame, (bx, by), (bx + fill_w, by + bar_h), fill_color, -1)
        cv2.putText(frame, f"Eye Closure: {int(ratio * 100)}%", (bx, by - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)

        return frame

    def get_metrics(self) -> Dict[str, Any]:
        """Return current detection metrics and driver alertness state."""
        return {
            "frame_count": self.frame_count,
            "eye_closure_count": self.eye_closure_count,
            "eye_closure_threshold": self.eye_closure_threshold,
            "eyes_closed": self.eyes_closed,
            "yawn_count": self.yawn_count,
            "yawn_threshold": self.yawn_threshold,
            "is_yawning": self.is_yawning,
            "tiredness_level": self.tiredness_level,
            "is_drowsy": self.is_drowsy,
            "alert_status": self.alert_status,
            "using_fallback": self.using_fallback
        }
