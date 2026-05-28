"""Simple Flask server to accept video uploads and return GrievX prediction JSON.

Usage:
  pip install -r requirements.txt
  python server.py

Endpoints:
  POST /predict  -- form field `file` with video file (mp4/mov/avi/mkv)

This keeps the model loaded in memory for faster repeated predictions.
"""
from __future__ import annotations

import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import List

import yaml
from flask import Flask, request, jsonify, send_from_directory

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover - runtime dependency
    raise SystemExit("ultralytics is required. pip install -r requirements.txt") from exc

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_ROOT / "runs" / "classify" / "grievx_yolo26" / "weights" / "best.pt"
CONFIG_PATH = PROJECT_ROOT / "grievx_config.yaml"

ALLOWED_EXT = {".mp4", ".mov", ".avi", ".mkv"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 250 * 1024 * 1024  # 250 MB upload cap


def sample_frames(video_path: Path, max_frames: int = 30, fps_extract: int = 5) -> List:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1.0, fps / float(fps_extract))
    frames = []
    idx = 0
    next_t = 0.0
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        if t + 1e-9 >= next_t:
            try:
                frames.append(cv2.resize(frame, (1280, 720)))
            except Exception:
                frames.append(frame)
            next_t += step
        idx += 1
    cap.release()
    return frames


def load_model(path: Path) -> YOLO:
    if not path.is_file():
        raise FileNotFoundError(f"Model not found: {path}")
    return YOLO(str(path))


def load_category_map() -> dict[str, dict[str, str]]:
    if not CONFIG_PATH.is_file():
        return {}
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    classes = cfg.get("classes") or {}
    return {
        name: {
            "category": info.get("problem_category", "Unknown"),
            "display_name": info.get("display_name", name),
        }
        for name, info in classes.items()
    }


# Load model at startup (fast reuse). If missing, server will refuse predict requests.
MODEL_PATH = DEFAULT_MODEL
MODEL = None
CATEGORY_MAP = load_category_map()
try:
    MODEL = load_model(MODEL_PATH)
    print(f"Loaded model: {MODEL_PATH}")
except FileNotFoundError:
    print(f"Warning: default model not found: {MODEL_PATH}")


@app.route("/", methods=["GET"])
def index():
    return send_from_directory(PROJECT_ROOT, "web_upload.html")


@app.route("/predict", methods=["POST"])
def predict_upload():
    global MODEL
    if MODEL is None:
        # try to load again if available
        try:
            MODEL = load_model(MODEL_PATH)
        except FileNotFoundError:
            return jsonify({"error": "model not found on server"}), 500

    f = request.files.get("file")
    if not f:
        return jsonify({"error": "no file provided"}), 400

    filename = f.filename or "upload"
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"error": "unsupported file type"}), 400

    max_frames = int(request.form.get("max_frames", 30))

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp_path = Path(tmp.name)
        f.save(tmp.name)

    try:
        frames = sample_frames(tmp_path, max_frames=max_frames)
        if not frames:
            return jsonify({"error": "no frames extracted"}), 500

        votes = []
        conf_sum = {}
        for frame in frames:
            try:
                results = MODEL.predict(frame, verbose=False)
            except Exception as e:
                return jsonify({"error": f"model predict failed: {e}"}), 500
            if not results:
                continue
            r = results[0]
            if r.probs is None:
                continue
            top_i = int(r.probs.top1)
            label = r.names[top_i]
            conf = float(r.probs.top1conf)
            votes.append(label)
            conf_sum[label] = conf_sum.get(label, 0.0) + conf

        if not votes:
            return jsonify({"error": "no predictions"}), 500

        winner, count = Counter(votes).most_common(1)[0]
        avg_conf = conf_sum[winner] / count
        meta = CATEGORY_MAP.get(winner, {})
        return jsonify({
            "label": winner,
            "category": meta.get("category", "Unknown"),
            "display_name": meta.get("display_name", winner),
            "confidence": avg_conf,
            "votes": count,
            "frames_used": len(frames),
        })
    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass


@app.errorhandler(413)
def too_large(_error):
    return jsonify({"error": "file too large"}), 413


if __name__ == "__main__":
    # Run in single-process dev mode. For production use gunicorn.
    app.run(host="0.0.0.0", port=8000)
