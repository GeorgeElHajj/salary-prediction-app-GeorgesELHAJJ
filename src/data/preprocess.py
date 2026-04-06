import pandas as pd


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    cleaned_df = df.copy()
    cleaned_df.columns = [col.strip().lower().replace(" ", "_") for col in cleaned_df.columns]
    cleaned_df = cleaned_df.drop_duplicates()
    return cleaned_df