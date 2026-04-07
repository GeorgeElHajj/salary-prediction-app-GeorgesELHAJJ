from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from src.api.schemas import PredictionInput, PredictionResponse


app = FastAPI(title="Salary Prediction API", version="0.1.0")

MODEL_PATH = Path("artifacts/model.joblib")


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

    return joblib.load(MODEL_PATH)


model = load_model()


@app.get("/")
def root():
    return {"message": "Salary Prediction API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predict", response_model=PredictionResponse)
def predict(
    work_year: int = Query(..., ge=2020, le=2022),
    experience_level: str = Query(...),
    employment_type: str = Query(...),
    job_title: str = Query(..., min_length=2),
    employee_residence: str = Query(..., min_length=2, max_length=2),
    remote_ratio: int = Query(...),
    company_location: str = Query(..., min_length=2, max_length=2),
    company_size: str = Query(...),
):
    if experience_level not in PredictionInput.EXPERIENCE_LEVELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid experience_level. Allowed values: {sorted(PredictionInput.EXPERIENCE_LEVELS)}",
        )

    if employment_type not in PredictionInput.EMPLOYMENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid employment_type. Allowed values: {sorted(PredictionInput.EMPLOYMENT_TYPES)}",
        )

    if company_size not in PredictionInput.COMPANY_SIZES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid company_size. Allowed values: {sorted(PredictionInput.COMPANY_SIZES)}",
        )

    if remote_ratio not in PredictionInput.REMOTE_RATIOS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid remote_ratio. Allowed values: {sorted(PredictionInput.REMOTE_RATIOS)}",
        )

    input_df = pd.DataFrame(
        [
            {
                "work_year": work_year,
                "experience_level": experience_level,
                "employment_type": employment_type,
                "job_title": job_title,
                "employee_residence": employee_residence,
                "remote_ratio": remote_ratio,
                "company_location": company_location,
                "company_size": company_size,
            }
        ]
    )

    try:
        prediction = model.predict(input_df)[0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(exc)}") from exc

    return PredictionResponse(
        predicted_salary_usd=round(float(prediction), 2),
        model_name=type(model.named_steps["model"]).__name__,
    )