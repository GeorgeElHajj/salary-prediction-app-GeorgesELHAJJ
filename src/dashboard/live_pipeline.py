from datetime import datetime
from typing import Any

import requests

from src.config import settings
from src.db.supabase_client import get_supabase_client


OLLAMA_MODEL = "llama3.2:3b"

# Local label maps kept here to avoid circular imports
EXPERIENCE_LABELS = {
    "EN": "Entry-level",
    "MI": "Mid-level",
    "SE": "Senior",
    "EX": "Executive",
}

EMPLOYMENT_LABELS = {
    "FT": "Full-time",
    "PT": "Part-time",
    "CT": "Contract",
    "FL": "Freelance",
}

COMPANY_SIZE_LABELS = {
    "S": "Small",
    "M": "Medium",
    "L": "Large",
}

COUNTRY_LABELS = {
    "AE": "United Arab Emirates", "AT": "Austria", "AU": "Australia",
    "BE": "Belgium", "BR": "Brazil", "CA": "Canada", "CH": "Switzerland",
    "CL": "Chile", "CN": "China", "CO": "Colombia", "DE": "Germany",
    "DK": "Denmark", "DZ": "Algeria", "EE": "Estonia", "ES": "Spain",
    "FR": "France", "GB": "United Kingdom", "GR": "Greece", "HN": "Honduras",
    "HR": "Croatia", "HU": "Hungary", "IE": "Ireland", "IN": "India",
    "IQ": "Iraq", "IR": "Iran", "IT": "Italy", "JP": "Japan",
    "KE": "Kenya", "LU": "Luxembourg", "MD": "Moldova", "MT": "Malta",
    "MX": "Mexico", "MY": "Malaysia", "NG": "Nigeria", "NL": "Netherlands",
    "NZ": "New Zealand", "PK": "Pakistan", "PL": "Poland", "PR": "Puerto Rico",
    "PT": "Portugal", "RO": "Romania", "RU": "Russia", "SG": "Singapore",
    "SI": "Slovenia", "TN": "Tunisia", "TR": "Turkey", "UA": "Ukraine",
    "US": "United States", "VN": "Vietnam",
}


# ─── API calls ───────────────────────────────────────────────────────────────

def call_prediction_api(api_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Call the FastAPI prediction endpoint and return the JSON response."""
    response = requests.get(api_url, params=payload, timeout=20)
    response.raise_for_status()
    return response.json()


def _human_remote_label(remote_ratio: int) -> str:
    if remote_ratio == 100:
        return "fully remote"
    if remote_ratio == 50:
        return "hybrid"
    return "on-site"


def build_live_prediction_prompt(input_data: dict[str, Any], predicted_salary: float) -> str:
    """Build a plain-English prompt for the LLM salary explanation."""
    exp = EXPERIENCE_LABELS.get(input_data["experience_level"], input_data["experience_level"])
    emp = EMPLOYMENT_LABELS.get(input_data["employment_type"], input_data["employment_type"])
    company_size = COMPANY_SIZE_LABELS.get(input_data["company_size"], input_data["company_size"])
    residence = COUNTRY_LABELS.get(input_data["employee_residence"], input_data["employee_residence"])
    company_location = COUNTRY_LABELS.get(input_data["company_location"], input_data["company_location"])
    remote = _human_remote_label(int(input_data["remote_ratio"]))

    return f"""
You are a professional salary analyst writing for a non-technical audience.

A machine learning model predicted this annual salary in USD:
Predicted salary: ${predicted_salary:,.2f}

Profile details:
- Work year: {input_data['work_year']}
- Job title: {input_data['job_title']}
- Experience level: {exp}
- Employment type: {emp}
- Employee residence: {residence}
- Work arrangement: {remote}
- Company location: {company_location}
- Company size: {company_size}

Write 2-3 short, friendly sentences.
Make it easy for a non-technical person to understand.
Mention the 2 most important factors influencing the estimate.
Do not mention machine learning, features, models, or jargon.
Do not invent unsupported facts.
Use confident but careful language.
""".strip()


def call_ollama_analysis(input_data: dict[str, Any], predicted_salary: float) -> str:
    """Generate a plain-English salary explanation using the local Ollama model."""
    prompt = build_live_prediction_prompt(input_data, predicted_salary)

    response = requests.post(
        f"{settings.OLLAMA_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def _fallback_analysis(input_data: dict[str, Any], predicted_salary: float) -> str:
    """Return a simple, readable fallback explanation when Ollama is disabled."""
    exp = EXPERIENCE_LABELS.get(input_data["experience_level"], input_data["experience_level"])
    emp = EMPLOYMENT_LABELS.get(input_data["employment_type"], input_data["employment_type"])
    loc = COUNTRY_LABELS.get(input_data["company_location"], input_data["company_location"])
    size = COMPANY_SIZE_LABELS.get(input_data["company_size"], input_data["company_size"])
    title = input_data["job_title"]
    remote_str = _human_remote_label(int(input_data["remote_ratio"]))

    return (
        f"This profile is estimated at about ${predicted_salary:,.0f} per year. "
        f"The biggest reasons are the {exp.lower()} level for a {title} role and the fact that it is a "
        f"{emp.lower()} position at a {size.lower()} company in {loc}. "
        f"The {remote_str} work setup also contributes to the overall estimate."
    )


# ─── Database helpers ────────────────────────────────────────────────────────

def create_run(supabase, model_name: str) -> int:
    """Insert a new prediction run and return its ID."""
    run_name = f"live_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    response = (
        supabase.table("prediction_runs")
        .insert({"run_name": run_name, "model_name": model_name})
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
    """Save a single prediction row to the predictions table."""
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
    """Save the generated explanation for a run."""
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


# ─── Main pipeline ───────────────────────────────────────────────────────────

def process_live_prediction(api_url: str, input_data: dict[str, Any]) -> dict[str, Any]:
    """
    Full pipeline:
      1. Call the FastAPI model to get a salary prediction.
      2. Generate a plain-English explanation (Ollama or fallback).
      3. Save the run, prediction, and analysis to Supabase.
      4. Return all results for display.
    """
    prediction_result = call_prediction_api(api_url, input_data)

    predicted_salary = float(prediction_result["predicted_salary_usd"])
    model_name = prediction_result["model_name"]

    if settings.ENABLE_LIVE_OLLAMA:
        analysis_text = call_ollama_analysis(input_data, predicted_salary)
    else:
        analysis_text = _fallback_analysis(input_data, predicted_salary)

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