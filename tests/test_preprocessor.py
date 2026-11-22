from __future__ import annotations

from matplotlib import dates
import pandas as pd

from src.data.preprocessor import preprocess_dataframe


def test_preprocess_dataframe_handles_bike_data() -> None:
    """Test that preprocess_dataframe correctly processes bike rental data """

    # Create 24 hours of bike rental data
    dates = pd.date_range(start="2023-01-01",periods=24,freq="h")

    df: pd.DataFrame = pd.DataFrame(
    {
        "Date": dates.strftime("%d/%m/%Y"),
        "Hour": dates.hour,
        "Seasons": ["Winter"] * 24,
        "Holiday": ["No Holiday"] * 24,
        "Functioning Day": ["Yes"] * 24,
        "Rented Bike Count": list(range(100, 124)),
    }
    )
    
    # Call the preprocessing function
    result: pd.DataFrame = preprocess_dataframe(df)
    
    # Verify that the target column is preserved after preprocessing
    assert "Rented Bike Count" in result.columns

    assert "Date" in result.columns
    assert "Hour" in result.columns
    assert "Holiday" in result.columns
    assert "Functioning Day" in result.columns
    assert "Season_Winter" in result.columns

    
    # Verify that the output is not empty
    assert len(result) > 0
