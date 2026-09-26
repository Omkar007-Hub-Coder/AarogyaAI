"""
train.py — Train recommendation models and save to models/.

Usage:
    python scripts/train.py

Requires processed data at data/processed/profiles_clean.csv.
Run scripts/preprocess.py first.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.features import build_feature_matrix  # noqa: E402
from ml.evaluate import evaluate_classifier, cross_validate  # noqa: E402

PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"


CATEGORY_LABEL_COLS = {
    "yoga": "yoga_label",
    "fitness": "fitness_label",
    "ahar": "ahar_label",
}


def train_category(X, y_raw: pd.Series, category: str) -> None:
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    metrics = evaluate_classifier(model, X_test, y_test)
    cv = cross_validate(model, X, y)

    print(f"\n[{category}] accuracy={metrics['accuracy']:.4f}  "
          f"cv={cv['mean_accuracy']:.4f}±{cv['std_accuracy']:.4f}")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "label_encoder": le}, MODELS_DIR / f"{category}_model.joblib")
    print(f"[{category}] Model saved to models/{category}_model.joblib")


def main() -> None:
    data_path = PROCESSED_DIR / "profiles_clean.csv"
    if not data_path.exists():
        print(f"[train] Processed data not found at {data_path}")
        print("  Run scripts/preprocess.py first.")
        return

    df = pd.read_csv(data_path)
    X = build_feature_matrix(df)

    for category, label_col in CATEGORY_LABEL_COLS.items():
        if label_col not in df.columns:
            print(f"[train] Column '{label_col}' not found in data — skipping {category}.")
            continue
        train_category(X, df[label_col], category)


if __name__ == "__main__":
    main()
