import pandas as pd


FEATURES = [
    "work_year",
    "experience_level",
    "employment_type",
    "job_title",
    "employee_residence",
    "remote_ratio",
    "company_location",
    "company_size"
]

TARGET = "salary_in_usd"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop unnecessary columns
    df = df.drop(columns=["Unnamed: 0", "salary", "salary_currency"], errors="ignore")

    # Normalize column names (safety)
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    # Drop duplicates (just in case)
    df = df.drop_duplicates()

    return df


def select_features(df: pd.DataFrame):
    df = df.copy()

    X = df[FEATURES]
    y = df[TARGET]

    return X, y


def preprocess_data(df: pd.DataFrame):
    df = clean_data(df)
    X, y = select_features(df)

    return X, y, df