from supabase import Client, create_client

from src.config import settings


def get_supabase_client() -> Client:
    if not settings.SUPABASE_URL:
        raise ValueError("SUPABASE_URL is missing in environment variables.")

    if not settings.SUPABASE_KEY:
        raise ValueError("SUPABASE_KEY is missing in environment variables.")

    return create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)