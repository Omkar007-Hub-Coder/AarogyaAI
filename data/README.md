# AarogyaAI — Data Directory

## ⚠️ Status: Datasets NOT present

All files under `data/raw/` must be downloaded manually and placed in the correct subdirectories before running any preprocessing or training scripts.

---

## Dataset 1 — Yoga-82 (PRIMARY)

| Field       | Value |
|-------------|-------|
| **Name**    | Yoga-82 |
| **Purpose** | Primary dataset for yoga pose image classification (82 pose classes) |
| **Source**  | Google Sites / Kaggle |
| **Official URL** | https://sites.google.com/view/yoga-82/home |
| **Kaggle URL**   | https://www.kaggle.com/datasets/akashrayhan/yoga-82 |
| **License** | Research / academic use only — check original paper for terms |
| **Citation** | Verma et al., "Yoga-82: A New Dataset for Fine-grained Classification of Human Poses", CVPRW 2020 |

### What to download
From Kaggle (`akashrayhan/yoga-82`) download the full dataset zip and extract. You should get:

```
data/raw/yoga82/
├── yoga_train/
│   ├── Akarna_Dhanurasana/
│   │   ├── image_001.jpg
│   │   └── ...
│   ├── Ananda_Balasana/
│   └── ... (82 pose directories)
├── yoga_test/
│   ├── Akarna_Dhanurasana/
│   └── ... (82 pose directories)
└── yoga82_classes.txt     # list of 82 class names (one per line)
```

Expected file:
```
data/raw/yoga82/yoga82_classes.txt
data/raw/yoga82/yoga_train/<PoseName>/*.jpg
data/raw/yoga82/yoga_test/<PoseName>/*.jpg
```

### Validation
Run `python scripts/validate_datasets.py` to confirm structure.

---

## Dataset 2 — Yoga Poses Dataset (SUPPLEMENTARY)

| Field       | Value |
|-------------|-------|
| **Name**    | Yoga Poses Dataset |
| **Purpose** | Additional poses for 5 classes (downdog, goddess, tree, plank, warrior2) — used where classes overlap with Yoga-82 |
| **Source**  | Kaggle |
| **URL**     | https://www.kaggle.com/datasets/niharika41298/yoga-poses-dataset |
| **License** | Kaggle dataset terms |

### What to download
Download and extract to:
```
data/raw/yoga_poses_5class/
├── DATASET/
│   ├── TRAIN/
│   │   ├── downdog/
│   │   ├── goddess/
│   │   ├── plank/
│   │   ├── tree/
│   │   └── warrior2/
│   └── TEST/
│       └── (same structure)
```

### Usage policy
This dataset will only be incorporated where its 5 pose classes overlap with Yoga-82 classes, to augment those specific classes. It will NOT be blindly merged.

---

## Dataset 3 — BlazePose Skeletons Yoga-82 (SKELETON FEATURES)

| Field       | Value |
|-------------|-------|
| **Name**    | BlazePose Skeletons Yoga-82 |
| **Purpose** | Pre-extracted 33-keypoint skeleton data for Yoga-82 poses — used for skeleton-based classification (not image-based) |
| **Source**  | Kaggle |
| **URL**     | https://www.kaggle.com/datasets/rashiniyasp/blazepose-skeletons-yoga-82 |
| **License** | Kaggle dataset terms |

### What to download
Download and extract to:
```
data/raw/blazepose_yoga82/
├── keypoints_train.csv   (or equivalent CSV/JSON with keypoints + labels)
├── keypoints_test.csv
└── README.txt (if present)
```

The CSV should have columns for 33 landmarks × (x, y, z, visibility) = 132 features + a `label` column.

---

## Preprocessing Output (data/processed/)

After running `scripts/preprocess.py`, the following will be created:

```
data/processed/
├── yoga82/
│   ├── train/          ← resized, normalized images (224×224)
│   ├── val/            ← validation split (10% from train)
│   └── test/           ← original test set
├── skeleton/
│   ├── train_features.npy    ← skeleton feature matrix
│   ├── test_features.npy
│   ├── train_labels.npy
│   ├── test_labels.npy
│   └── label_encoder.joblib
├── user_profiles/
│   └── synthetic_profiles.csv   ← SYNTHETIC DATA — clearly labeled
├── splits/
│   └── split_manifest.json     ← reproducible split indices + random seed
└── metadata/
    ├── yoga82_classes.txt       ← canonical 82 class list
    ├── class_to_idx.json
    └── dataset_stats.json       ← image counts, class distribution
```

---

## Important Notes

1. **Do NOT commit raw data to git** — `data/raw/` is in `.gitignore`
2. **Do NOT generate fake statistics** — all counts come from actual files
3. Processed data is also in `.gitignore` (regenerate from raw)
4. The `data/processed/` directory is safe to recreate via `scripts/preprocess.py`

---

## Preprocessing Steps

1. Validate directory structure matches expected layout
2. Scan all images, flag corrupted/unreadable ones
3. Load and deduplicate where possible (hash-based)
4. Standardize class names to canonical Yoga-82 label format
5. Resize images to 224×224 (EfficientNet input)
6. Create stratified train/val/test splits (80/10/10)
7. Save processed images and metadata
8. For skeleton data: validate keypoint CSV, engineer features (angles, normalized coords)
9. Save skeleton feature matrices as `.npy`
