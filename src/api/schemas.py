from datetime import date
from pydantic import BaseModel, Field

# Data received from the user
class PredictionRequest(BaseModel):

    date: date
    hour: int = Field(ge=0, le=23) # ge = greater than or equal , le = less than or equal

    temperature: float
    humidity: float = Field(ge=0, le=100)
    windspeed: float = Field(ge=0)
    visibility: float = Field(ge=0)
    dewpoint: float

    seasons: str
    holiday: str
    functioning_day: str

    previous_7_rented_bikes_per_hour: list[int] = Field(
        min_length=7,
        max_length=7
    )

    rented_bikes_same_hour_yesterday: int

# Data returned to the user
class PredictionResponse(BaseModel):
    prediction: int