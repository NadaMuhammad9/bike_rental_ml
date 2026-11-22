from __future__ import annotations

import pandas as pd

from src.data.preprocessor import preprocess_dataframe
from src.features.features import create_features
"""
TESTING THE GENERATED FEATURES 
 generated features : temporal features like (Hour_Sin) , lag/rolling features like (target_lag_1 , target_roll_mean_7)
 
"""

# Create 50 hours of sample bike rental data
def create_sample_data() -> pd.DataFrame:
    dates = pd.date_range(
        start="01-01-2023",
        periods=50,
        freq="h",
    )

    return pd.DataFrame(
        {
            "Date": dates.strftime("%d/%m/%Y"),
            "Hour": dates.hour,
            "Seasons": ["Winter"] * 50,
            "Holiday": ["No Holiday"] * 50,
            "Functioning Day": ["Yes"] * 50,
            "Rented Bike Count": list(range(30, 80)),
        }
    )


def test_create_features_generates_time_features() -> None:
    df = create_sample_data()

    processed = preprocess_dataframe(df)
    result = create_features(processed)

    assert "Day" in result.columns
    assert "Week_of_Year" in result.columns
    assert "Month" in result.columns
    assert "Year" in result.columns

    assert "Hour_Sin" in result.columns
    assert "Hour_Cos" in result.columns


def test_create_features_generates_lag_features() -> None:
    df = create_sample_data()

    processed = preprocess_dataframe(df)
    result = create_features(processed)

    assert "target_lag_1" in result.columns
    assert "target_lag_3" in result.columns
    assert "target_lag_7" in result.columns
    assert "target_lag_24" in result.columns


def test_create_features_generates_rolling_features() -> None:
    df = create_sample_data()

    processed = preprocess_dataframe(df)
    result = create_features(processed)

    assert "target_roll_mean_3" in result.columns
    assert "target_roll_mean_7" in result.columns
    assert "target_roll_std_7" in result.columns
    assert "target_roll_min_7" in result.columns
    assert "target_roll_max_7" in result.columns


def test_create_features_returns_non_empty_dataframe() -> None:
    df = create_sample_data()

    processed = preprocess_dataframe(df)
    result = create_features(processed)

    assert len(result) > 0