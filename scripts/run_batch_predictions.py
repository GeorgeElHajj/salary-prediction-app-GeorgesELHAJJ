from pathlib import Path
import json

from dotenv import load_dotenv
import pandas as pd
import requests
load_dotenv()

import os
API_URL = os.getenv(
    "FASTAPI_PREDICT_URL",
    "http://127.0.0.1:8000/predict"
)
OUTPUT_DIR = Path("artifacts/predictions")
OUTPUT_CSV = OUTPUT_DIR / "predictions.csv"
OUTPUT_JSON = OUTPUT_DIR / "predictions.json"
REQUEST_TIMEOUT = 10


SCENARIOS = [
    {
        "work_year": 2022,
        "experience_level": "EN",
        "employment_type": "FT",
        "job_title": "Data Scientist",
        "employee_residence": "US",
        "remote_ratio": 50,
        "company_location": "US",
        "company_size": "S",
    },
    {
        "work_year": 2022,
        "experience_level": "MI",
        "employment_type": "FT",
        "job_title": "Data Scientist",
        "employee_residence": "US",
        "remote_ratio": 100,
        "company_location": "US",
        "company_size": "M",
    },
    {
        "work_year": 2022,
        "experience_level": "SE",
        "employment_type": "FT",
        "job_title": "Data Scientist",
        "employee_residence": "US",
        "remote_ratio": 100,
        "company_location": "US",
        "company_size": "L",
    },
    {
        "work_year": 2022,
        "experience_level": "EX",
        "employment_type": "FT",
        "job_title": "Data Scientist",
        "employee_residence": "US",
        "remote_ratio": 100,
        "company_location": "US",
        "company_size": "L",
    },
    {
        "work_year": 2022,
        "experience_level": "SE",
        "employment_type": "FT",
        "job_title": "Machine Learning Engineer",
        "employee_residence": "US",
        "remote_ratio": 100,
        "company_location": "US",
        "company_size": "L",
    },
    {
        "work_year": 2022,
        "experience_level": "MI",
        "employment_type": "FT",
        "job_title": "Data Analyst",
        "employee_residence": "GB",
        "remote_ratio": 50,
        "company_location": "GB",
        "company_size": "M",
    },
    {
        "work_year": 2022,
        "experience_level": "EN",
        "employment_type": "FT",
        "job_title": "Research Scientist",
        "employee_residence": "DE",
        "remote_ratio": 0,
        "company_location": "DE",
        "company_size": "S",
    },
    {
        "work_year": 2022,
        "experience_level": "SE",
        "employment_type": "FT",
        "job_title": "Data Engineer",
        "employee_residence": "US",
        "remote_ratio": 50,
        "company_location": "US",
        "company_size": "M",
    },
    {
        "work_year": 2022,
        "experience_level": "MI",
        "employment_type": "FT",
        "job_title": "Machine Learning Scientist",
        "employee_residence": "CA",
        "remote_ratio": 100,
        "company_location": "CA",
        "company_size": "L",
    },
    {
        "work_year": 2022,
        "experience_level": "SE",
        "employment_type": "FT",
        "job_title": "Analytics Engineer",
        "employee_residence": "US",
        "remote_ratio": 100,
        "company_location": "US",
        "company_size": "M",
    },
]
def call_prediction_api(payload: dict) -> dict:
    try:
        response = requests.get(API_URL, params=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        prediction_data = response.json()

        return {
            **payload,
            "predicted_salary_usd": prediction_data["predicted_salary_usd"],
            "model_name": prediction_data["model_name"],
            "status": "success",
            "error_message": None,
        }

    except requests.exceptions.RequestException as exc:
        return {
            **payload,
            "predicted_salary_usd": None,
            "model_name": None,
            "status": "failed",
            "error_message": str(exc),
        }


def main():
    print("Starting batch predictions...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for index, scenario in enumerate(SCENARIOS, start=1):
        print(f"Calling API for scenario {index}/{len(SCENARIOS)}...")
        result = call_prediction_api(scenario)
        results.append(result)

    results_df = pd.DataFrame(results)

    results_df.to_csv(OUTPUT_CSV, index=False)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    success_count = (results_df["status"] == "success").sum()
    failed_count = (results_df["status"] == "failed").sum()

    print(f"Predictions completed. Success: {success_count}, Failed: {failed_count}")
    print(f"CSV saved to: {OUTPUT_CSV}")
    print(f"JSON saved to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()