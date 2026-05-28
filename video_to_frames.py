"""
GrievX — video → image frames (one problem at a time).

Each problem is finished completely (all its videos) before the next starts.

Output:
    frames_output/<ProblemName>/<ProblemName>_<video_name>_f000001.jpg

Run ONE problem (recommended):
    python video_to_frames.py Crop_Damage
    python video_to_frames.py Drainage_issue

Run ALL problems in order (Crop → Drainage → … → Road):
    python video_to_frames.py --all
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# --- config (edit if needed) ---
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "frames_output"
FRAMES_PER_SECOND = 15
OUTPUT_SIZE = (1280, 720)
BLUR_THRESHOLD = 100.0
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

# Order: process one folder fully, then move to the next
PROBLEM_FOLDERS = [
    "Crop_Damage",
    "Drainage_issue",
    "Drinking_water_shortage",
    "Industrial_Pollution_complaint",
    "Road_Damage",
]


def sanitize_name(text: str) -> str:
    text = text.strip().replace(" ", "_")
    return re.sub(r"[^\w\-.]", "_", text)


def is_sharp(frame: np.ndarray, threshold: float) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score >= threshold


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    problem_label: str,
    frames_per_second: int = FRAMES_PER_SECOND,
    blur_threshold: float = BLUR_THRESHOLD,
) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"    [skip] cannot open: {video_path.name}", file=sys.stderr)
        return 0, 0

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if not video_fps or video_fps <= 0:
        video_fps = 30.0

    video_stem = sanitize_name(video_path.stem)
    prefix = f"{problem_label}_{video_stem}"

    saved = 0
    skipped_blur = 0
    frame_idx = 0
    next_sample_time = 0.0
    time_step = 1.0 / frames_per_second

    while True:
        success, frame = cap.read()
        if not success:
            break

        current_time = frame_idx / video_fps
        if current_time + 1e-9 >= next_sample_time:
            frame_resized = cv2.resize(frame, OUTPUT_SIZE)
            if is_sharp(frame_resized, blur_threshold):
                out_name = f"{prefix}_f{saved:06d}.jpg"
                cv2.imwrite(
                    str(output_dir / out_name),
                    frame_resized,
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                )
                saved += 1
            else:
                skipped_blur += 1
            next_sample_time += time_step

        frame_idx += 1

    cap.release()
    return saved, skipped_blur


def collect_videos(problem_dir: Path) -> list[Path]:
    return sorted(
        p for p in problem_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def process_problem_folder(problem_name: str) -> dict:
    """Convert every video in one problem folder; finish before returning."""
    problem_dir = PROJECT_ROOT / problem_name
    output_dir = OUTPUT_ROOT / problem_name

    if not problem_dir.is_dir():
        print(f"[error] folder not found: {problem_dir}")
        return {"problem": problem_name, "videos": 0, "frames_saved": 0, "frames_skipped_blur": 0}

    output_dir.mkdir(parents=True, exist_ok=True)
    videos = collect_videos(problem_dir)

    stats = {
        "problem": problem_name,
        "videos": len(videos),
        "frames_saved": 0,
        "frames_skipped_blur": 0,
    }

    print("=" * 60)
    print(f"START: {problem_name}")
    print(f"  Videos in : {problem_dir}")
    print(f"  Images out: {output_dir}")
    print(f"  Video count : {len(videos)}")
    print("=" * 60)

    if not videos:
        print(f"[warn] no videos in {problem_dir}")
        print(f"DONE: {problem_name} (nothing to process)\n")
        return stats

    for i, video_path in enumerate(videos, start=1):
        print(f"\n  [{i}/{len(videos)}] {video_path.name}")
        saved, skipped = extract_frames_from_video(
            video_path, output_dir, problem_name
        )
        stats["frames_saved"] += saved
        stats["frames_skipped_blur"] += skipped
        print(f"       -> {saved} images saved, {skipped} blurry skipped")

    print()
    print("=" * 60)
    print(f"DONE: {problem_name}")
    print(
        f"  Total: {stats['frames_saved']} images "
        f"({stats['frames_skipped_blur']} blurry rejected)"
    )
    print(f"  Saved under: {output_dir}")
    print("=" * 60)
    print()
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert grievance videos to frames — one problem at a time."
    )
    parser.add_argument(
        "problem",
        nargs="?",
        help="Problem folder name, e.g. Crop_Damage",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all problems in order (each fully, then next)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List valid problem folder names",
    )
    return parser.parse_args()


def resolve_problems(args: argparse.Namespace) -> list[str]:
    if args.list:
        print("Valid problem folders:")
        for name in PROBLEM_FOLDERS:
            exists = "OK" if (PROJECT_ROOT / name).is_dir() else "MISSING"
            print(f"  {exists}  {name}")
        sys.exit(0)

    if args.problem:
        name = args.problem.strip()
        if name not in PROBLEM_FOLDERS:
            print(f"[error] Unknown problem: {name}")
            print(f"        Use one of: {', '.join(PROBLEM_FOLDERS)}")
            sys.exit(1)
        return [name]

    if args.all:
        return list(PROBLEM_FOLDERS)

    print("Usage:")
    print("  python video_to_frames.py Crop_Damage     # one problem only")
    print("  python video_to_frames.py --all           # all, one after another")
    print("  python video_to_frames.py --list")
    sys.exit(1)


def main() -> None:
    args = parse_args()
    problems = resolve_problems(args)

    print("GrievX — video to frames (one problem at a time)")
    print(f"  Project root : {PROJECT_ROOT}")
    print(f"  Output root  : {OUTPUT_ROOT}")
    print(f"  Frame rate   : {FRAMES_PER_SECOND} frames / sec of video")
    print(f"  Resize       : {OUTPUT_SIZE[0]}x{OUTPUT_SIZE[1]}")
    print(f"  Blur cutoff  : >= {BLUR_THRESHOLD}")
    if len(problems) > 1:
        print(f"  Queue        : {' -> '.join(problems)}")
    print()

    all_stats = []
    for problem in problems:
        all_stats.append(process_problem_folder(problem))

    if len(all_stats) > 1:
        print("\n--- Final summary (all problems) ---")
        total_saved = 0
        for s in all_stats:
            print(
                f"  {s['problem']}: {s['videos']} videos, "
                f"{s['frames_saved']} images"
            )
            total_saved += s["frames_saved"]
        print(f"\n  Grand total: {total_saved} images in {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
