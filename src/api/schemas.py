from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    predicted_salary_usd: float = Field(..., example=145000.0)
    model_name: str = Field(..., example="DecisionTreeRegressor")


class PredictionInput:
    EXPERIENCE_LEVELS = {"EN", "MI", "SE", "EX"}
    EMPLOYMENT_TYPES = {"FT", "PT", "CT", "FL"}
    COMPANY_SIZES = {"S", "M", "L"}
    REMOTE_RATIOS = {0, 50, 100}