"""
Predict GrievX complaint type from a citizen video using trained YOLO26-cls.

Extracts frames, runs classifier, returns majority vote + confidence.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_ROOT / "runs" / "classify" / "grievx_yolo26" / "weights" / "best.pt"
CONFIG_PATH = PROJECT_ROOT / "grievx_config.yaml"
FRAMES_PER_SECOND = 5  # fewer frames needed at inference than training extract


def extract_sample_frames(video_path: Path, max_frames: int = 30) -> list:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = fps / FRAMES_PER_SECOND
    frames = []
    idx = 0
    next_t = 0.0

    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        if t + 1e-9 >= next_t:
            frames.append(cv2.resize(frame, (1280, 720)))
            next_t += step
        idx += 1

    cap.release()
    return frames


def load_category_map() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return {
        name: info.get("problem_category", name)
        for name, info in (cfg.get("classes") or {}).items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="GrievX video complaint classifier")
    parser.add_argument("--video", required=True, help="Path to complaint video")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="best.pt path")
    parser.add_argument("--max-frames", type=int, default=30)
    args = parser.parse_args()

    video_path = Path(args.video)
    model_path = Path(args.model)
    if not video_path.is_file():
        print(f"Video not found: {video_path}", file=sys.stderr)
        sys.exit(1)
    if not model_path.is_file():
        print(
            f"Model not found: {model_path}\n"
            "Train first: python train_grievx_yolo26.py",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("pip install ultralytics", file=sys.stderr)
        sys.exit(1)

    category_map = load_category_map()
    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))

    print(f"Sampling video: {video_path.name}")
    frames = extract_sample_frames(video_path, max_frames=args.max_frames)
    if not frames:
        print("No frames extracted.", file=sys.stderr)
        sys.exit(1)

    votes: list[str] = []
    conf_sum: dict[str, float] = {}

    for i, frame in enumerate(frames):
        results = model.predict(frame, verbose=False)
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
        print("No predictions.", file=sys.stderr)
        sys.exit(1)

    winner, count = Counter(votes).most_common(1)[0]
    avg_conf = conf_sum[winner] / count
    category = category_map.get(winner, "Unknown")

    print()
    print("=" * 50)
    print("GrievX prediction")
    print("=" * 50)
    print(f"  Complaint type : {winner}")
    print(f"  Category       : {category}")
    print(f"  Confidence     : {avg_conf:.2%} ({count}/{len(votes)} frames)")
    print("=" * 50)


if __name__ == "__main__":
    main()
