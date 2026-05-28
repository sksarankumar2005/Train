from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent
removed = []

# Remove runs/ and exports/ directories (common training outputs)
runs_dir = PROJECT_ROOT / "runs"
if runs_dir.exists():
    try:
        shutil.rmtree(runs_dir)
        removed.append(str(runs_dir))
    except Exception as e:
        print(f"Failed to remove {runs_dir}: {e}")

exports_dir = PROJECT_ROOT / "exports"
if exports_dir.exists():
    try:
        shutil.rmtree(exports_dir)
        removed.append(str(exports_dir))
    except Exception as e:
        print(f"Failed to remove {exports_dir}: {e}")

# Remove common checkpoint files named best.pt or last.pt anywhere under project
for name in ("best.pt", "last.pt"):
    for p in PROJECT_ROOT.rglob(name):
        try:
            p.unlink()
            removed.append(str(p))
        except Exception as e:
            print(f"Failed to remove {p}: {e}")

# Remove any "weights" directories left behind
for w in PROJECT_ROOT.rglob("weights"):
    if w.is_dir():
        try:
            shutil.rmtree(w)
            removed.append(str(w))
        except Exception as e:
            print(f"Failed to remove {w}: {e}")

if removed:
    print("Removed the following items:")
    for r in removed:
        print(" -", r)
else:
    print("No saved model artifacts found to remove.")

print("Cleanup script finished.")
