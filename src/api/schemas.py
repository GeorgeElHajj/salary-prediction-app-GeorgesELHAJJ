from pydantic import BaseModel


class PredictionResponse(BaseModel):
    predicted_salary_usd: float