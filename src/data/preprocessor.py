from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import config

TARGET_COLUMN = config.target_column
DATE_COLUMN = config.date_column
DROP_COLUMNS = ["Solar Radiation (MJ/m2)", "Rainfall(mm)", "Snowfall (cm)"]
BINARY_MAPPINGS = {
    "No Holiday": 0,
    "Holiday": 1,
    "No": 0,
    "Yes": 1,
}


def preprocess_dataframe(
    df: pd.DataFrame,
    save_path: str | Path | None = None,
) -> pd.DataFrame:
    cleaned = df.copy()

    # Convert Date to datetime
    if DATE_COLUMN in cleaned.columns:
        cleaned[DATE_COLUMN] = pd.to_datetime(
            cleaned[DATE_COLUMN], format="%d/%m/%Y", errors="coerce"
        )
        cleaned = cleaned.dropna(subset=[DATE_COLUMN]).reset_index(drop=True)

    # Drop unnecessary columns
    columns_to_drop = [column for column in DROP_COLUMNS if column in cleaned.columns]
    if columns_to_drop:
        cleaned = cleaned.drop(columns=columns_to_drop)

    # Convert binary categorical columns to integers
    for column in ["Holiday", "Functioning Day"]:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].map(BINARY_MAPPINGS).astype("Int64")

    # One-hot encode Seasons
    if "Seasons" in cleaned.columns:
        cleaned = pd.get_dummies(
            cleaned,
            columns=["Seasons"],
            dtype=int,
            prefix="Season",
        )

    # Sort data chronologically
    if DATE_COLUMN in cleaned.columns:
        sort_keys = [DATE_COLUMN]

        if "Hour" in cleaned.columns:
            sort_keys.append("Hour")

        cleaned = (
            cleaned
            .sort_values(by=sort_keys)
            .reset_index(drop=True)
        )


    # Save processed data
    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        cleaned.to_csv(
            output_path,
            index=False,
        )

    return cleaned