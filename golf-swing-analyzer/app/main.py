from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from mediapipe import solutions as mp_solutions

APP_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(APP_DIR, "templates")
STATIC_DIR = os.path.join(APP_DIR, "static")

app = FastAPI(title="Golf Swing Analyzer (Local MVP)")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


@dataclass
class FrameMetrics:
    frame_index: int
    shoulder_angle_deg: float
    spine_angle_deg: float
    nose: Tuple[float, float]
    shoulder_width: float
    wrist_min_y: float


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    ext = os.path.splitext(file.filename.lower())[1]
    if ext not in {".mp4", ".mov", ".m4v", ".avi"}:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "upload" + ext)
        with open(video_path, "wb") as f:
            f.write(await file.read())

        metrics = extract_metrics(video_path)
        if metrics is None or not metrics[0]:
            raise HTTPException(status_code=422, detail="No pose landmarks detected")

        frames, fps, stride = metrics
        summary = summarize_metrics(frames, fps, stride)
        return JSONResponse(summary)


def extract_metrics(video_path: str) -> Optional[Tuple[List[FrameMetrics], float, int]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Sample frames to keep it fast on CPU.
    stride = max(1, total_frames // 120)

    pose = mp_solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frames: List[FrameMetrics] = []
    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % stride != 0:
            frame_index += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)
        if result.pose_landmarks is not None:
            lm = result.pose_landmarks.landmark

            left_shoulder = lm[mp_solutions.pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = lm[mp_solutions.pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = lm[mp_solutions.pose.PoseLandmark.LEFT_HIP]
            right_hip = lm[mp_solutions.pose.PoseLandmark.RIGHT_HIP]
            left_wrist = lm[mp_solutions.pose.PoseLandmark.LEFT_WRIST]
            right_wrist = lm[mp_solutions.pose.PoseLandmark.RIGHT_WRIST]
            nose = lm[mp_solutions.pose.PoseLandmark.NOSE]

            shoulder_angle_deg = line_angle_deg(
                (left_shoulder.x, left_shoulder.y),
                (right_shoulder.x, right_shoulder.y),
            )
            spine_angle_deg = spine_angle(
                (left_shoulder.x, left_shoulder.y),
                (right_shoulder.x, right_shoulder.y),
                (left_hip.x, left_hip.y),
                (right_hip.x, right_hip.y),
            )

            shoulder_width = math.hypot(
                right_shoulder.x - left_shoulder.x,
                right_shoulder.y - left_shoulder.y,
            )
            wrist_min_y = min(left_wrist.y, right_wrist.y)

            frames.append(
                FrameMetrics(
                    frame_index=frame_index,
                    shoulder_angle_deg=shoulder_angle_deg,
                    spine_angle_deg=spine_angle_deg,
                    nose=(nose.x, nose.y),
                    shoulder_width=shoulder_width,
                    wrist_min_y=wrist_min_y,
                )
            )

        frame_index += 1

    cap.release()
    pose.close()

    if not frames:
        return None

    return frames, fps, stride


def line_angle_deg(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
    return angle


def spine_angle(
    left_shoulder: Tuple[float, float],
    right_shoulder: Tuple[float, float],
    left_hip: Tuple[float, float],
    right_hip: Tuple[float, float],
) -> float:
    shoulder_mid = ((left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2)
    hip_mid = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)

    v = np.array([shoulder_mid[0] - hip_mid[0], shoulder_mid[1] - hip_mid[1]], dtype=float)
    v_norm = np.linalg.norm(v)
    if v_norm == 0:
        return 0.0

    # Vertical reference vector (screen coordinates: y increases downward).
    vertical = np.array([0.0, -1.0], dtype=float)
    cos_angle = float(np.dot(v, vertical) / (v_norm * np.linalg.norm(vertical)))
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


def summarize_metrics(frames: List[FrameMetrics], fps: float, stride: int) -> dict:
    address = frames[0]
    top = min(frames, key=lambda f: f.wrist_min_y)

    shoulder_widths = [f.shoulder_width for f in frames if f.shoulder_width > 0]
    shoulder_width = float(np.median(shoulder_widths)) if shoulder_widths else 1.0

    nose_positions = np.array([f.nose for f in frames], dtype=float)
    nose_center = nose_positions.mean(axis=0)
    nose_dist = np.linalg.norm(nose_positions - nose_center, axis=1)
    head_stability = float(nose_dist.std() / shoulder_width)

    spine_change = abs(top.spine_angle_deg - address.spine_angle_deg)
    shoulder_tilt = abs(address.shoulder_angle_deg)

    frames_to_top = max(1, int((top.frame_index - address.frame_index) / stride))
    tempo_seconds = float(frames_to_top / fps * stride)

    tips = []
    tips.append("MVP assumes a face-on camera view. Results may be off for other angles.")

    if head_stability > 0.15:
        tips.append("Head movement looks high. Try keeping the head more stable through the backswing.")

    if spine_change > 10:
        tips.append("Spine angle changes a lot from address to top. Focus on maintaining posture.")

    if shoulder_tilt > 12:
        tips.append("Shoulder line looks tilted at address. Square up the shoulders if possible.")

    if tempo_seconds < 0.6:
        tips.append("Backswing tempo looks fast. A smoother backswing can improve control.")

    return {
        "metrics": {
            "head_stability": round(head_stability, 3),
            "spine_angle_address": round(address.spine_angle_deg, 1),
            "spine_angle_top": round(top.spine_angle_deg, 1),
            "spine_angle_change": round(spine_change, 1),
            "shoulder_tilt_address": round(address.shoulder_angle_deg, 1),
            "tempo_seconds": round(tempo_seconds, 2),
            "frames_analyzed": len(frames),
        },
        "tips": tips,
    }
