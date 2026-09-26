# AarogyaAI

**Personalized Yoga, Fitness & Ahar (Nutrition) Recommendation System**

An academic Machine Learning project demonstrating end-to-end ML integration for personalized wellness recommendations.

> ⚕️ **Disclaimer:** AarogyaAI is a research/educational project. It is **not** a medical diagnostic or clinical treatment tool. All recommendations are general wellness guidance only.

---

## Architecture

```
AarogyaAI/
├── frontend/        React + TypeScript + Tailwind CSS (Vite)
├── backend/         FastAPI + SQLite
├── ml/
│   ├── models/      EfficientNetB0, skeleton RF, goal/difficulty classifiers
│   ├── preprocessing/  image_utils, skeleton_utils, splits
│   ├── recommendation/ yoga engine, ahar engine, metadata
│   ├── personalization/  profile, BMI, TDEE/macro estimation
│   ├── training/    training scripts
│   ├── inference/   unified predictor API
│   └── evaluation/  metrics utilities
├── data/
│   ├── raw/yoga82/               Yoga-82 image dataset (11,652 images, 82 classes)
│   ├── raw/yoga-82-skeletons-normalized/  BlazePose skeleton dataset
│   └── processed/yoga82/         Cleaned + resized images (train/valid/test)
├── models/          Trained model artifacts
├── tests/           143 tests (all passing)
└── scripts/         Training and preprocessing scripts
```

---

## ML Components

### REAL Trained ML

| Component | Dataset | Status |
|---|---|---|
| **EfficientNetB0 Yoga-82 Image Classifier** | Real Yoga-82 dataset (82 classes, 11,608 train images) | Training in progress (Phase 1, ~10–15 epochs completed) |
| **BlazePose Yoga-82 Skeleton Classifier** | Real BlazePose Yoga-82 dataset (82 classes, 14,878 samples) | ✅ Trained. Test Top-1: **86.72%**, Top-5: **95.72%**, Macro F1: **85.52%** |

### SYNTHETIC / PROTOTYPE

| Component | Status |
|---|---|
| **User Goal Classifier** | ⚠️ Trained on procedurally-generated synthetic data. Not validated on real users. |
| **Yoga Difficulty Predictor** | ⚠️ Trained on procedurally-generated synthetic data. Not validated on real users. |

### CONTENT-BASED (No ML Training)

| Component | Method |
|---|---|
| **Yoga Recommendation Engine** | Feature-vector suitability scoring (15-dim, weighted dot product) |
| **Ahar/Nutrition Recommender** | Curated food metadata + goal/macro alignment scoring. Hard allergy filters. |

---

## Dataset Sources

| Dataset | Source | Type |
|---|---|---|
| Yoga-82 Images | Provided by course/project | Real (11,652 images, 82 classes) |
| BlazePose Yoga-82 Skeletons | Kaggle (yoga-82-skeletons-normalized) | Real (14,878 .npy files, 82 classes) |
| Food metadata (Ahar) | ICMR-NIN Indian Food Composition Tables 2017, USDA FoodData Central | Curated reference data |
| Goal/Difficulty training data | Procedurally generated with domain rules | SYNTHETIC |

---

## Recommendation Methodology

### Yoga Ranking
Each pose is a 15-dimensional feature vector. Each user profile is similarly encoded. Suitability score = weighted compatibility across dimensions (difficulty, flexibility, strength, experience, duration, style preferences, goal alignment). Highest weight on goal alignment.

### Ahar (Nutrition) Ranking
Score = 0.40 × goal alignment + 0.35 × macro profile fit + 0.25 × calorie direction + preference boost. Dietary preference and allergy filters are **hard** — allergens are never included regardless of score.

### Calorie/Macro Estimates
Uses the **Mifflin-St Jeor equation** (1990). PAL multiplier based on activity level. Macro targets split by goal (e.g., Strength: 35% protein / 45% carbs / 20% fat). These are **population-level estimates**, not personalized medical advice.

### Feedback Re-ranking
When a user rates a recommendation: completed + 5-star → +10% score; not completed → −5% score. Score clamped to [0, 1]. **This is a heuristic, not reinforcement learning.**

---

## Installation

### Prerequisites
- Python 3.14 (for backend/tests) — in `.venv/`
- Python 3.9.6 (for TensorFlow/ML training) — in `.venv-tf/`
- Node.js ≥ 18

### Backend Setup
```bash
cd AarogyaAI
python3.14 -m venv .venv           # or use existing .venv
.venv/bin/pip install -r requirements.txt
```

### Frontend Setup
```bash
cd AarogyaAI/frontend
npm install
```

---

## Starting the Application

### 1. Start Backend
```bash
cd AarogyaAI/backend
../.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Backend URL: **http://localhost:8000**
API docs: **http://localhost:8000/docs**

### 2. Start Frontend
```bash
cd AarogyaAI/frontend
npm run dev
```
Frontend URL: **http://localhost:5173**

The frontend proxies all `/api/*` and `/health` requests to `http://localhost:8000`.

---

## User Flow

1. **Landing page** (`/`) — introduction to AarogyaAI
2. **Profile form** (`/profile`) — enter health profile (age, height, weight, goal, activity, yoga experience, dietary preference, allergies, etc.)
3. **Dashboard** (`/dashboard`) — personalized results:
   - Profile summary with BMI (WHO classification, not diagnosis)
   - Estimated daily energy + macro targets (Mifflin-St Jeor estimate)
   - Top yoga pose recommendations with suitability scores and reasons
   - Top Ahar (food) recommendations with nutritional info and reasons
   - Feedback on each item (satisfaction, difficulty, completion)
4. **Pose Recognition** (`/pose`) — upload a yoga pose image for EfficientNetB0 classification (requires training to complete)

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/api/v1/profile` | Create legacy profile |
| GET | `/api/v1/profile/{id}` | Get profile |
| GET | `/api/v1/recommendations/{user_id}` | Legacy recommendations |
| POST | `/api/v1/personalization/recommend` | Full personalized recommendations |
| POST | `/api/v1/feedback` | Submit item feedback |
| GET | `/api/v1/feedback/user/{user_id}` | Get user feedback |
| GET | `/api/v1/ml/status` | ML model availability |
| POST | `/api/v1/ml/recommend` | ML yoga recommendations |
| POST | `/api/v1/ml/predict/goal` | Goal prediction (SYNTHETIC) |
| POST | `/api/v1/ml/predict/difficulty` | Difficulty prediction (SYNTHETIC) |
| POST | `/api/v1/ml/predict/image` | Yoga pose image classification (REAL ML) |
| GET | `/api/v1/catalog/yoga` | Yoga catalog |
| GET | `/api/v1/catalog/fitness` | Fitness catalog |
| GET | `/api/v1/catalog/ahar` | Ahar catalog |

---

## Testing

```bash
cd AarogyaAI

# Run full Python test suite (143 tests)
.venv/bin/python -m pytest tests/ -v

# TypeScript build check
cd frontend && npm run build
```

Current test result: **143 passed in ~1.7s**

---

## Known Limitations

1. **EfficientNetB0 training still in progress** — Image pose recognition will return a 503 until training completes (EfficientNetB0, Phase 1/15 epochs, PID 9694 running). Once complete, the best checkpoint at `models/yoga_classifier_best.keras` will be used automatically.
2. **Goal/Difficulty models are SYNTHETIC prototypes** — Not validated on real user data. Always labelled as such in API responses.
3. **Ahar catalog is small (30 items)** — A production system would use a full nutritional database (e.g., complete ICMR-NIN tables).
4. **No user authentication** — This is an academic prototype; user sessions are not secured.
5. **Calorie/macro targets are estimates** — Individual metabolic variation is not captured by the Mifflin-St Jeor equation.
6. **BlazePose skeleton classifier requires a webcam/pose-estimation library** for real-time inference — currently only batch `.npy` file inference is supported in the backend.
7. **No persistent login** — The dashboard reads from `sessionStorage`; closing the browser tab loses the session.
