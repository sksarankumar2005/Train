"""
Train YOLO26 classification model for GrievX complaint types.

Based on YOLO26_model.ipynb (uses yolo26n-cls.pt for folder-based labels).

Prerequisites:
    python prepare_grievx_dataset.py --clean

Usage:
    python train_grievx_yolo26.py
    python train_grievx_yolo26.py --epochs 30 --model yolo26s-cls.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "dataset_grievx"
DEFAULT_MODEL = "yolo26m-cls.pt"


def check_dataset() -> None:
    train_dir = DATASET_ROOT / "train"
    val_dir = DATASET_ROOT / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        print(
            "Dataset not found. Run first:\n"
            "  python prepare_grievx_dataset.py --clean",
            file=sys.stderr,
        )
        sys.exit(1)
    train_imgs = list(train_dir.rglob("*.jpg"))
    val_imgs = list(val_dir.rglob("*.jpg"))
    if len(train_imgs) < 50 or len(val_imgs) < 10:
        print(
            f"[warn] Small dataset: {len(train_imgs)} train, {len(val_imgs)} val images",
            file=sys.stderr,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GrievX YOLO26 classifier")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="e.g. yolo26m-cls.pt")
    parser.add_argument("--epochs", type=int, default=50, help="higher accuracy default; reduce for quick tests")
    parser.add_argument("--imgsz", type=int, default=384, help="use 320 on CPU if needed")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", default="", help="cuda, cpu, or empty=auto")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--name", default="grievx_yolo26", help="run folder name")
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=0.003)
    parser.add_argument("--lrf", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    check_dataset()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Install ultralytics: pip install ultralytics", file=sys.stderr)
        sys.exit(1)

    print("GrievX YOLO26 classification training")
    print(f"  Model   : {args.model}")
    print(f"  Data    : {DATASET_ROOT}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  Image   : {args.imgsz}")
    print(f"  Batch   : {args.batch}")
    print(f"  Optim   : {args.optimizer}")
    print()

    model = YOLO(args.model)
    results = model.train(
        data=str(DATASET_ROOT),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device or None,
        workers=args.workers,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        dropout=args.dropout,
        seed=args.seed,
        project=str(PROJECT_ROOT / "runs" / "classify"),
        name=args.name,
        exist_ok=True,
        pretrained=True,
        verbose=True,
        plots=True,
    )

    best = PROJECT_ROOT / "runs" / "classify" / args.name / "weights" / "best.pt"
    print()
    print("=" * 60)
    print("Training complete")
    print(f"  Best weights: {best}")
    print(f"  Results dir : {PROJECT_ROOT / 'runs' / 'classify' / args.name}")
    print()
    print("Predict on a video:")
    print(f"  python predict_grievx_video.py --video Road_Damage/Road\\ damage.mp4 --model {best}")
    print("=" * 60)
    return results


if __name__ == "__main__":
    main()
