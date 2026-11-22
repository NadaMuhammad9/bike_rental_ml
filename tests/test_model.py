from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.models.train import train_model


def test_train_model_creates_model_and_metrics(tmp_path: Path) -> None:
    """Test that train_model creates model artifacts and computes valid metrics.
    
    This test verifies that the training pipeline:
    - Creates a model file at the specified path
    - Generates processed data at the output path
    - Computes valid metrics (non-negative RMSE values)
    """
    # Set up a temporary directory structure with sample data
    data_path: Path = tmp_path / "raw"
    data_path.mkdir(parents=True)
    sample_file: Path = data_path / "sample.csv"

    # Create sample training data with hourly values for testing
    dates = pd.date_range("2023-01-01", periods=50, freq="h")

    df: pd.DataFrame = pd.DataFrame(
    {
        "Date": dates.strftime("%d/%m/%Y"),
        "Hour": dates.hour,
        "Seasons": ["Winter"] * 50,
        "Holiday": ["No Holiday"] * 50,
        "Functioning Day": ["Yes"] * 50,
        "Temperature": [25.4] * 50,
        "Humidity": [72.0] * 50,
        "Wind speed": [12.5] * 50,
        "Visibility": [10.0] * 50,
        "Dew point temperature": [15.2] * 50,
        "Rented Bike Count": list(range(50, 100)),
    }
)
    df.to_csv(sample_file, index=False)

    # Define output paths for model and processed data
    model_path: Path = tmp_path / "models" / "lightgbm.joblib"
    processed_path: Path = tmp_path / "processed" / "seoul_bike_processed.csv"

    # Call the training function with sample data
    result: dict[str, Any] = train_model(
        data_path=sample_file,
        processed_path=processed_path,
        model_path=model_path,
    )

    # Verify that the model and processed data paths are returned correctly
    assert result["model_path"] == str(model_path)
    assert result["processed_path"] == str(processed_path)
    
    # Verify that computed metrics are valid (non-negative RMSE)
    assert result["metrics"]["rmse"] >= 0
    
    # Verify that the model and processed data files were created
    assert model_path.exists()
    assert processed_path.exists()
