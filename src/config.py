import os
from dotenv import load_dotenv

load_dotenv()

try:
    import streamlit as st
except ImportError:
    st = None


def get_secret(key: str, default: str = "") -> str:
    value = os.getenv(key)
    if value:
        return value

    if st is not None:
        try:
            return st.secrets[key]
        except Exception:
            pass

    return default


class Settings:
    SUPABASE_URL: str = get_secret("SUPABASE_URL")
    SUPABASE_KEY: str = get_secret("SUPABASE_KEY")
    MODEL_PATH: str = get_secret("MODEL_PATH", "artifacts/model.joblib")
    OLLAMA_URL: str = get_secret("OLLAMA_URL", "http://localhost:11434")
    FASTAPI_PREDICT_URL: str = get_secret("FASTAPI_PREDICT_URL", "http://127.0.0.1:8000/predict")


settings = Settings()