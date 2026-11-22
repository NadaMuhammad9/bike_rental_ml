from __future__ import annotations
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import config
from src.models.model_io import load_model

"""
Inference Service

 - The preprocessing performed during inference MUST MATCH the preprocessing used 
 during training so that the model receives the same features in the same format
 and order
 - This ensures that the prediction is based on the correct feature
 representation and AVOIDS INCONSISTENCIES BETWEEN TRAINING AND INFERENCE.

"""


class PredictionService:
    def __init__(
        self,
        model_path: str | Path | None = None,
    ) -> None:
        # Model path from argument or config
        self.model_path = Path(
            model_path or config.model_path
        )

        self.model = None
        self.feature_names: list[str] = []

    def load_model(self):
        # Load the model only once
        if self.model is None:
            self.model, self.feature_names = load_model(
                self.model_path
            )

        return self.model

    def create_features(
        self,
        data: dict,
    ) -> pd.DataFrame:

        # date 
        date = pd.to_datetime(data["date"])

        # Previous 7 Rentals provided by the user
        Rentals = data["previous_7_rented_bikes_per_hour"]

        features = {
            # Time features

            "Day": date.day,
            "Week_of_Year": date.isocalendar().week,
            "Month": date.month,
            "Year": date.year,

            "Hour": data["hour"],

            "Hour_Sin": np.sin(
                2 * np.pi * data["hour"] / 24
            ),

            "Hour_Cos": np.cos(
                2 * np.pi * data["hour"] / 24
            ),

            # Weather and categorical features
            "Temperature(°C)": data["temperature"],
            "Humidity(%)": data["humidity"],
            "Wind speed (m/s)": data["windspeed"],
            "Visibility (10m)": data["visibility"],
            "Dew point temperature(°C)": data["dewpoint"],

            "Season_Autumn": int(data["seasons"] == "Autumn"),
            "Season_Spring": int(data["seasons"] == "Spring"),
            "Season_Summer": int(data["seasons"] == "Summer"),
            "Season_Winter": int(data["seasons"] == "Winter"),
            "Holiday": 1 if data["holiday"] == "Holiday" else 0,
            "Functioning Day": 1 if data["functioning_day"] == "Yes" else 0,


            # Lag features
            "target_lag_1": Rentals[-1],
            "target_lag_3": Rentals[-3],
            "target_lag_7": Rentals[-7],

            "target_lag_24": (
                data["rented_bikes_same_hour_yesterday"]
            ),

            # Rolling features
            "target_roll_mean_3": (
                sum(Rentals[-3:]) / 3
            ),

            "target_roll_mean_7": (
                sum(Rentals) / 7
            ),

            "target_roll_std_7": (
                pd.Series(Rentals).std()
            ),

            "target_roll_min_7": min(Rentals),

            "target_roll_max_7": max(Rentals),
        }

        return pd.DataFrame([features])

    def predict(
        self,
        data: dict,
    ) -> int:

        # Load trained model
        model = self.load_model()

        # Create features from user input
        features = self.create_features(data)

        # Check that all model features are available
        missing_features = [
            name
            for name in self.feature_names
            if name not in features.columns
        ]

        if missing_features:
            raise ValueError(
                f"Missing required features: {missing_features}"
            )

        # Keep the same feature order used during training
        X = features[self.feature_names]

        # Generate prediction
        prediction = model.predict(X)[0]

        return round(float(prediction))