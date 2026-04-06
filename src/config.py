from dotenv import load_dotenv
import os

load_dotenv()


class Settings:
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")
    MODEL_PATH: str = os.getenv("MODEL_PATH", "artifacts/model.joblib")
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")


settings = Settings()