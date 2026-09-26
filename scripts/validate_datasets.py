"""
scripts/validate_datasets.py

Validates expected dataset directory structures in data/raw/.
Run this before preprocessing to confirm all required files are in place.

Usage:
    python scripts/validate_datasets.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CHECKS = {
    "Yoga-82 (train)": {
        "path": ROOT / "data" / "raw" / "yoga82" / "yoga_train",
        "type": "dir",
        "desc": "Download from https://www.kaggle.com/datasets/akashrayhan/yoga-82",
    },
    "Yoga-82 (test)": {
        "path": ROOT / "data" / "raw" / "yoga82" / "yoga_test",
        "type": "dir",
        "desc": "Same download as above",
    },
    "BlazePose keypoints (train CSV)": {
        "path": ROOT / "data" / "raw" / "blazepose_yoga82" / "keypoints_train.csv",
        "type": "file",
        "desc": "Download from https://www.kaggle.com/datasets/rashiniyasp/blazepose-skeletons-yoga-82",
    },
    "Yoga Poses 5-class (TRAIN dir)": {
        "path": ROOT / "data" / "raw" / "yoga_poses_5class" / "DATASET" / "TRAIN",
        "type": "dir",
        "desc": "Download from https://www.kaggle.com/datasets/niharika41298/yoga-poses-dataset",
    },
}


def main() -> None:
    print("\n🔍  AarogyaAI — Dataset Validation\n" + "=" * 50)
    all_ok = True
    for name, info in CHECKS.items():
        path = info["path"]
        exists = path.exists() and (
            (info["type"] == "dir" and path.is_dir()) or
            (info["type"] == "file" and path.is_file())
        )
        status = "✅ FOUND" if exists else "❌ MISSING"
        print(f"\n  {status}: {name}")
        print(f"    Path : {path}")
        if not exists:
            all_ok = False
            print(f"    Fix  : {info['desc']}")
        else:
            # Count contents
            if info["type"] == "dir":
                n_subdirs = sum(1 for x in path.iterdir() if x.is_dir())
                n_files = sum(1 for x in path.rglob("*") if x.is_file())
                print(f"    Stats: {n_subdirs} subdirs, {n_files} files")

    print("\n" + "=" * 50)
    if all_ok:
        print("✅  All datasets present. Ready to run scripts/preprocess.py --all\n")
    else:
        print("⚠️   Some datasets are missing. See data/README.md for download instructions.\n")
        print("     Run again after placing datasets in data/raw/\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
