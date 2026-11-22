from __future__ import annotations
from pathlib import Path
import joblib

# Handles loading and saving model files cleanly, so the rest of the app doesn't have to.
# Makes the model easy to store, update, and reuse across the application.

# Save the trained model and feature names to disk.
def save_model(model: object, feature_names: list[str], path: str | Path) -> None:
    output_path = Path(path)

    # Create the output directory if it does not exist.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Store the model and feature names together.
    joblib.dump({"model": model, "feature_names": feature_names}, output_path)

# Load a saved model and its feature names.
def load_model(path: str | Path) -> tuple[object, list[str]]:
    model_path = Path(path)

    # Check that the model file exists.
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    data = joblib.load(model_path)

    # Validate the structure of the saved model file.
    if not isinstance(data, dict) or "model" not in data or "feature_names" not in data:
        raise ValueError("Saved model file must contain 'model' and 'feature_names'.")

    return data["model"], list(data["feature_names"])
