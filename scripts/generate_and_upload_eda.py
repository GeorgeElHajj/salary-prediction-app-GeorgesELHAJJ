from pathlib import Path
import uuid

import matplotlib.pyplot as plt
import pandas as pd

from src.db.supabase_client import get_supabase_client
from src.data.load_data import load_dataset
from src.data.preprocess import preprocess_data


DATA_PATH = Path("data/raw/ds_salaries.csv")
OUTPUT_DIR = Path("artifacts/eda")
BUCKET_NAME = "eda-assets"


def save_chart(fig, filename: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    file_path = OUTPUT_DIR / filename
    fig.savefig(file_path, bbox_inches="tight")
    plt.close(fig)
    return file_path


def create_eda_charts(df: pd.DataFrame) -> list[dict]:
    charts = []

    # 1. Salary by experience level
    fig, ax = plt.subplots(figsize=(8, 5))
    (
        df.groupby("experience_level")["salary_in_usd"]
        .mean()
        .reindex(["EN", "MI", "SE", "EX"])
        .plot(kind="bar", ax=ax)
    )
    ax.set_title("Average Salary by Experience Level")
    ax.set_xlabel("Experience Level")
    ax.set_ylabel("Salary in USD")
    path = save_chart(fig, "salary_by_experience.png")
    charts.append({
        "title": "Average Salary by Experience Level",
        "description": "Shows how salary generally increases as experience level grows.",
        "chart_type": "bar",
        "file_path": path,
    })

    # 2. Salary by employment type
    fig, ax = plt.subplots(figsize=(8, 5))
    (
        df.groupby("employment_type")["salary_in_usd"]
        .mean()
        .sort_values(ascending=False)
        .plot(kind="bar", ax=ax)
    )
    ax.set_title("Average Salary by Employment Type")
    ax.set_xlabel("Employment Type")
    ax.set_ylabel("Salary in USD")
    path = save_chart(fig, "salary_by_employment_type.png")
    charts.append({
        "title": "Average Salary by Employment Type",
        "description": "Compares average salaries across full-time, part-time, contract, and freelance roles.",
        "chart_type": "bar",
        "file_path": path,
    })

    # 3. Top 10 job titles by salary
    fig, ax = plt.subplots(figsize=(10, 6))
    (
        df.groupby("job_title")["salary_in_usd"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .sort_values()
        .plot(kind="barh", ax=ax)
    )
    ax.set_title("Top 10 Job Titles by Average Salary")
    ax.set_xlabel("Salary in USD")
    ax.set_ylabel("Job Title")
    path = save_chart(fig, "top_job_titles_salary.png")
    charts.append({
        "title": "Top 10 Job Titles by Average Salary",
        "description": "Highlights which data roles tend to have the highest average salaries.",
        "chart_type": "barh",
        "file_path": path,
    })

    # 4. Salary distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    df["salary_in_usd"].plot(kind="hist", bins=30, ax=ax)
    ax.set_title("Salary Distribution")
    ax.set_xlabel("Salary in USD")
    ax.set_ylabel("Frequency")
    path = save_chart(fig, "salary_distribution.png")
    charts.append({
        "title": "Salary Distribution",
        "description": "Shows how salaries are distributed across the dataset.",
        "chart_type": "histogram",
        "file_path": path,
    })

    return charts


def upload_file_to_storage(supabase, file_path: Path) -> str:
    unique_name = f"{uuid.uuid4()}_{file_path.name}"

    with open(file_path, "rb") as f:
        supabase.storage.from_(BUCKET_NAME).upload(
            path=unique_name,
            file=f,
            file_options={"content-type": "image/png"}
        )

    public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(unique_name)
    return public_url


def insert_eda_asset(supabase, title: str, description: str, chart_type: str, file_name: str, public_url: str):
    supabase.table("eda_assets").insert({
        "title": title,
        "description": description,
        "chart_type": chart_type,
        "file_name": file_name,
        "public_url": public_url,
    }).execute()


def main():
    print("Loading dataset...")
    raw_df = load_dataset(str(DATA_PATH))
    _, _, processed_df = preprocess_data(raw_df)

    print("Creating EDA charts...")
    charts = create_eda_charts(processed_df)

    supabase = get_supabase_client()

    for chart in charts:
        print(f"Uploading {chart['title']}...")
        public_url = upload_file_to_storage(supabase, chart["file_path"])
        insert_eda_asset(
            supabase=supabase,
            title=chart["title"],
            description=chart["description"],
            chart_type=chart["chart_type"],
            file_name=chart["file_path"].name,
            public_url=public_url,
        )

    print("EDA charts uploaded and saved successfully.")


if __name__ == "__main__":
    main()