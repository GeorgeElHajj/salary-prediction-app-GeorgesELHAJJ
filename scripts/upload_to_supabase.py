from pathlib import Path
import json
from datetime import datetime

import pandas as pd

from src.db.supabase_client import get_supabase_client



PREDICTIONS_PATH = Path("artifacts/predictions/predictions.csv")
SUMMARY_PATH = Path("artifacts/analysis/summary.json")
ANALYSIS_PATH = Path("artifacts/analysis/analysis.txt")
CHART_PATH = Path("artifacts/charts/salary_by_experience.png")


def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"Predictions file not found: {PREDICTIONS_PATH}")

    return pd.read_csv(PREDICTIONS_PATH)


def load_summary() -> dict:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Summary file not found: {SUMMARY_PATH}")

    with open(SUMMARY_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def load_analysis_text() -> str:
    if not ANALYSIS_PATH.exists():
        raise FileNotFoundError(f"Analysis file not found: {ANALYSIS_PATH}")

    with open(ANALYSIS_PATH, "r", encoding="utf-8") as file:
        return file.read()


def create_prediction_run(supabase, run_name: str, model_name: str) -> int:
    response = (
        supabase.table("prediction_runs")
        .insert(
            {
                "run_name": run_name,
                "model_name": model_name,
            }
        )
        .execute()
    )

    run_row = response.data[0]
    return run_row["id"]


def insert_predictions(supabase, run_id: int, predictions_df: pd.DataFrame) -> None:
    prediction_rows = []

    for _, row in predictions_df.iterrows():
        prediction_rows.append(
            {
                "run_id": run_id,
                "work_year": int(row["work_year"]),
                "experience_level": row["experience_level"],
                "employment_type": row["employment_type"],
                "job_title": row["job_title"],
                "employee_residence": row["employee_residence"],
                "remote_ratio": int(row["remote_ratio"]),
                "company_location": row["company_location"],
                "company_size": row["company_size"],
                "predicted_salary_usd": float(row["predicted_salary_usd"]),
                "status": row["status"],
                "error_message": None if pd.isna(row["error_message"]) else str(row["error_message"]),
            }
        )

    supabase.table("predictions").insert(prediction_rows).execute()


def insert_analysis(
    supabase,
    run_id: int,
    summary: dict,
    analysis_text: str,
    chart_path: str,
) -> None:
    supabase.table("analyses").insert(
        {
            "run_id": run_id,
            "average_salary": float(summary["average_salary"]),
            "max_salary": float(summary["max_salary"]),
            "min_salary": float(summary["min_salary"]),
            "analysis_text": analysis_text,
            "chart_path": chart_path,
        }
    ).execute()


def main() -> None:
    print("Connecting to Supabase...")
    supabase = get_supabase_client()

    print("Loading local artifacts...")
    predictions_df = load_predictions()
    summary = load_summary()
    analysis_text = load_analysis_text()

    model_name = (
        predictions_df["model_name"].dropna().iloc[0]
        if "model_name" in predictions_df.columns and not predictions_df["model_name"].dropna().empty
        else "UnknownModel"
    )

    run_name = f"salary_prediction_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print("Creating prediction run...")
    run_id = create_prediction_run(
        supabase=supabase,
        run_name=run_name,
        model_name=model_name,
    )
    print(f"Created run with ID: {run_id}")

    print("Uploading predictions...")
    insert_predictions(supabase, run_id, predictions_df)

    print("Uploading analysis...")
    insert_analysis(
        supabase=supabase,
        run_id=run_id,
        summary=summary,
        analysis_text=analysis_text,
        chart_path=str(CHART_PATH),
    )

    print("Upload complete!")
    print(f"Run ID: {run_id}")


if __name__ == "__main__":
    main()