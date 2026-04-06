from pathlib import Path
import pandas as pd


def load_dataset(file_path: str) -> pd.DataFrame:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {file_path}")

    return pd.read_csv(path)