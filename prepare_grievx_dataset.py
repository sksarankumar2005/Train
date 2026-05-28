"""
Build YOLO26 classification dataset from frames_output/.

Splits by VIDEO (not random frames) to avoid data leakage.
Output layout (Ultralytics classify format):

    dataset_grievx/
      train/<class_name>/*.jpg
      val/<class_name>/*.jpg
      split_manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
FRAMES_ROOT = PROJECT_ROOT / "frames_output"
DATASET_ROOT = PROJECT_ROOT / "dataset_grievx"

CLASSES = [
    "Crop_Damage",
    "Drainage_issue",
    "Drinking_water_shortage",
    "Industrial_Pollution_complaint",
    "Road_Damage",
]

FRAME_PATTERN = re.compile(r"^(.+)_f(\d+)$")


def video_key(stem: str, class_name: str) -> str:
    """Group frames from the same source video."""
    prefix = f"{class_name}_"
    if stem.startswith(prefix):
        stem = stem[len(prefix) :]
    m = FRAME_PATTERN.match(stem)
    if m:
        return m.group(1)
    return stem


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_split(
    val_ratio: float = 0.2,
    seed: int = 42,
    clean: bool = False,
) -> dict:
    if clean and DATASET_ROOT.exists():
        shutil.rmtree(DATASET_ROOT)

    rng = random.Random(seed)
    manifest: dict = {
        "seed": seed,
        "val_ratio": val_ratio,
        "classes": CLASSES,
        "train_videos": defaultdict(list),
        "val_videos": defaultdict(list),
        "counts": {"train": {}, "val": {}},
    }

    for class_name in CLASSES:
        class_dir = FRAMES_ROOT / class_name
        if not class_dir.is_dir():
            print(f"[warn] missing: {class_dir}", file=sys.stderr)
            continue

        by_video: dict[str, list[Path]] = defaultdict(list)
        for img in sorted(class_dir.glob("*.jpg")):
            key = video_key(img.stem, class_name)
            by_video[key].append(img)

        video_ids = sorted(by_video.keys())
        if not video_ids:
            continue

        rng.shuffle(video_ids)
        n_val = max(1, int(round(len(video_ids) * val_ratio)))
        if len(video_ids) == 1:
            n_val = 0
        val_ids = set(video_ids[:n_val])
        train_ids = set(video_ids[n_val:])

        if not train_ids:
            # tiny class: keep at least one video for train
            val_ids = {video_ids[0]}
            train_ids = set(video_ids[1:]) if len(video_ids) > 1 else set(video_ids)

        for vid in train_ids:
            manifest["train_videos"][class_name].append(vid)
            for src in by_video[vid]:
                dst = DATASET_ROOT / "train" / class_name / src.name
                link_or_copy(src, dst)

        for vid in val_ids:
            manifest["val_videos"][class_name].append(vid)
            for src in by_video[vid]:
                dst = DATASET_ROOT / "val" / class_name / src.name
                link_or_copy(src, dst)

        train_n = sum(len(by_video[v]) for v in train_ids)
        val_n = sum(len(by_video[v]) for v in val_ids)
        manifest["counts"]["train"][class_name] = train_n
        manifest["counts"]["val"][class_name] = val_n

    # JSON-serializable
    manifest["train_videos"] = dict(manifest["train_videos"])
    manifest["val_videos"] = dict(manifest["val_videos"])

    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    with open(DATASET_ROOT / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare GrievX YOLO classify dataset")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete dataset_grievx before rebuild",
    )
    args = parser.parse_args()

    print("GrievX dataset preparation")
    print(f"  Source : {FRAMES_ROOT}")
    print(f"  Output : {DATASET_ROOT}")
    print(f"  Split  : {args.val_ratio:.0%} val (by video)")
    print()

    manifest = build_split(
        val_ratio=args.val_ratio,
        seed=args.seed,
        clean=args.clean,
    )

    print("--- Split summary ---")
    for split in ("train", "val"):
        print(f"\n  {split.upper()}:")
        for cls in CLASSES:
            n = manifest["counts"].get(split, {}).get(cls, 0)
            vids = len(manifest[f"{split}_videos"].get(cls, []))
            print(f"    {cls}: {n} images ({vids} videos)")

    total_train = sum(manifest["counts"]["train"].values())
    total_val = sum(manifest["counts"]["val"].values())
    print(f"\n  Total: {total_train} train, {total_val} val")
    print(f"  Manifest: {DATASET_ROOT / 'split_manifest.json'}")
    print("\nReady for: python train_grievx_yolo26.py")


if __name__ == "__main__":
    main()
