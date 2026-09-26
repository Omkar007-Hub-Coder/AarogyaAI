# Models Directory

Trained model files (`.joblib`) are saved here by `scripts/train.py`.

Files are excluded from git (see `.gitignore`). After training you will find:
- `yoga_model.joblib`
- `fitness_model.joblib`
- `ahar_model.joblib`

Each file contains a dict `{"model": sklearn_model, "label_encoder": LabelEncoder}`.
