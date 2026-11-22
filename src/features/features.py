from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import config


TARGET_COLUMN = config.target_column


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()
    if "Date" in featured.columns:
         featured["Date"] = pd.to_datetime(
            featured["Date"],
        )

    # Extract temporal features from the Date column
    featured["Day"] = featured["Date"].dt.day
    featured["Week_of_Year"] = featured["Date"].dt.isocalendar().week
    featured["Month"] = featured["Date"].dt.month
    featured["Year"] = featured["Date"].dt.year
    
    # Cyclic encoding for Hour (to make it understandable for the model that 23 and 0 are close)
    if "Hour" in featured.columns:
        featured["Hour"] = featured["Hour"].astype(int)

        featured["Hour_Sin"] = np.sin(
            2 * np.pi * featured["Hour"] / 24
        )

        featured["Hour_Cos"] = np.cos(
            2 * np.pi * featured["Hour"] / 24
        )

    # lag and rolling features
    if TARGET_COLUMN in featured.columns:
        if len(featured) >= 2:

            # Lag features
            for lag in [1, 3, 7, 24]:
                featured[f"target_lag_{lag}"] = (
                    featured[TARGET_COLUMN].shift(lag)
                )

            # Rolling features
            previous_target = featured[TARGET_COLUMN].shift(1)

            featured["target_roll_mean_3"] = (
                previous_target
                .rolling(window=3)
                .mean()
            )

            featured["target_roll_mean_7"] = (
                previous_target
                .rolling(window=7)
                .mean()
            )

            featured["target_roll_std_7"] = (
                previous_target
                .rolling(window=7)
                .std()
            )

            featured["target_roll_min_7"] = (
                previous_target
                .rolling(window=7)
                .min()
            )

            featured["target_roll_max_7"] = (
                previous_target
                .rolling(window=7)
                .max()
            )

            # Remove rows with insufficient history
            featured = (
                featured
                .dropna()
                .reset_index(drop=True)
            )

    return featured