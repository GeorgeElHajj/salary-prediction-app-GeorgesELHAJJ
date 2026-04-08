from datetime import datetime
from typing import Any

import requests

from src.db.supabase_client import get_supabase_client


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"


def call_prediction_api(api_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.get(api_url, params=payload, timeout=20)
    response.raise_for_status()
    return response.json()


def build_live_prediction_prompt(input_data: dict[str, Any], predicted_salary: float) -> str:
    return f"""
You are a professional salary analyst.

A machine learning model predicted this annual salary in USD:

Predicted salary: {predicted_salary:.2f}

Profile:
- Work year: {input_data['work_year']}
- Experience level: {input_data['experience_level']}
- Employment type: {input_data['employment_type']}
- Job title: {input_data['job_title']}
- Employee residence: {input_data['employee_residence']}
- Remote ratio: {input_data['remote_ratio']}
- Company location: {input_data['company_location']}
- Company size: {input_data['company_size']}

Write a short explanation for a non-technical user.
Keep it to 2 or 3 sentences only.
Explain what this salary suggests and mention the most important factors.
Do not invent unsupported facts.
""".strip()


def call_ollama_analysis(input_data: dict[str, Any], predicted_salary: float) -> str:
    prompt = build_live_prediction_prompt(input_data, predicted_salary)

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["response"]


def create_run(supabase, model_name: str) -> int:
    run_name = f"live_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

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

    return response.data[0]["id"]


def insert_prediction(
    supabase,
    run_id: int,
    input_data: dict[str, Any],
    predicted_salary: float,
    status: str = "success",
    error_message: str | None = None,
) -> None:
    supabase.table("predictions").insert(
        {
            "run_id": run_id,
            "work_year": int(input_data["work_year"]),
            "experience_level": input_data["experience_level"],
            "employment_type": input_data["employment_type"],
            "job_title": input_data["job_title"],
            "employee_residence": input_data["employee_residence"],
            "remote_ratio": int(input_data["remote_ratio"]),
            "company_location": input_data["company_location"],
            "company_size": input_data["company_size"],
            "predicted_salary_usd": float(predicted_salary),
            "status": status,
            "error_message": error_message,
        }
    ).execute()


def insert_analysis(
    supabase,
    run_id: int,
    predicted_salary: float,
    analysis_text: str,
) -> None:
    supabase.table("analyses").insert(
        {
            "run_id": run_id,
            "average_salary": float(predicted_salary),
            "max_salary": float(predicted_salary),
            "min_salary": float(predicted_salary),
            "analysis_text": analysis_text,
            "chart_path": None,
        }
    ).execute()


def process_live_prediction(api_url: str, input_data: dict[str, Any]) -> dict[str, Any]:
    prediction_result = call_prediction_api(api_url, input_data)

    predicted_salary = float(prediction_result["predicted_salary_usd"])
    model_name = prediction_result["model_name"]

    analysis_text = call_ollama_analysis(input_data, predicted_salary)

    supabase = get_supabase_client()
    run_id = create_run(supabase, model_name)
    insert_prediction(supabase, run_id, input_data, predicted_salary)
    insert_analysis(supabase, run_id, predicted_salary, analysis_text)

    return {
        "run_id": run_id,
        "predicted_salary_usd": predicted_salary,
        "model_name": model_name,
        "analysis_text": analysis_text,
    }