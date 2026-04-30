# Relevant Priors API

FastAPI service for predicting whether a prior imaging study is relevant to a current study.

The app scores each `(current study, prior study)` pair and returns a boolean relevance prediction for every prior in the request payload.

## What It Does

- Exposes a small HTTP API with health and prediction endpoints.
- Builds structured features from study descriptions and dates.
- Uses a trained scikit-learn model saved at `app/model_v2.joblib`.
- Includes scripts for local evaluation and retraining from the public challenge dataset.

## Project Structure

```text
app/
  main.py         FastAPI app and endpoints
  schemas.py      Request/response models
  features.py     Feature engineering helpers
  model.py        Model loading and batch prediction
scripts/
  train_model_v2.py  Train and save the model artifact
  eval_local.py      Run local evaluation on the dataset
requirements.txt
```

## Requirements

- Python 3.11+ recommended
- A virtual environment
- Dependencies from `requirements.txt`

Install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Running The API

Start the server:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /health` returns `{"ok": true}`
- `POST /predict` returns relevance predictions for each prior study

## Request Format

`POST /predict`

```json
{
  "challenge_id": "demo",
  "schema_version": 1,
  "generated_at": "2026-04-29T21:00:00Z",
  "cases": [
    {
      "case_id": "case-1",
      "patient_id": "p-1",
      "patient_name": "Example Patient",
      "current_study": {
        "study_id": "cur-1",
        "study_description": "CT chest with contrast",
        "study_date": "2026-04-10"
      },
      "prior_studies": [
        {
          "study_id": "prior-1",
          "study_description": "CT chest without contrast",
          "study_date": "2025-11-15"
        },
        {
          "study_id": "prior-2",
          "study_description": "XR knee 3 views",
          "study_date": "2025-10-01"
        }
      ]
    }
  ]
}
```

Example response:

```json
{
  "predictions": [
    {
      "case_id": "case-1",
      "study_id": "prior-1",
      "predicted_is_relevant": true
    },
    {
      "case_id": "case-1",
      "study_id": "prior-2",
      "predicted_is_relevant": false
    }
  ]
}
```

## Training

The training script expects the dataset file `relevant_priors_public.json` at the repository root.

Train the model:

```bash
python scripts/train_model_v2.py
```

This writes the trained artifact to `app/model_v2.joblib`.

## Local Evaluation

Run:

```bash
python scripts/eval_local.py
```

The script prints accuracy plus sample false positives and false negatives.

## Notes

- The API currently relies on `app/model_v2.joblib` being present for prediction.
- The dataset file is allowed in git by `.gitignore`, but large JSON files other than `relevant_priors_public.json` are ignored.
