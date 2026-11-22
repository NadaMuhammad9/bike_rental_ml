from __future__ import annotations
from pathlib import Path
import pandas as pd

# Load a CSV file and return it as a pandas DataFrame.
def load_csv_data(path: str | Path) -> pd.DataFrame:

    # Convert the input path to a Path object (to handle both string and Path inputs)
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # If a directory is provided, look for CSV files inside it
    if data_path.is_dir():
        csv_files = list(data_path.glob("*.csv"))

        # Raise an error if no CSV files are found
        if len(csv_files) == 0:
            raise FileNotFoundError(f"No CSV file found in directory: {data_path}")

        # Raise an error if multiple CSV files are found
        if len(csv_files) > 1:
            raise ValueError(
                f"Multiple CSV files found in directory {data_path}. Please provide a specific file path."
            )

        # Use the only CSV file found in the directory
        data_path = csv_files[0]

    # Load the CSV file into a pandas DataFrame
    return pd.read_csv(data_path , encoding="cp1252")
