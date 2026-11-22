from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _safe_float(value: float) -> float:
    value = float(value)
    if not np.isfinite(value):
        return 0.0
    return value


def evaluate_model(model, X_test, y_test) -> dict[str, float]:
    predictions = model.predict(X_test)

    # Rental count cannot be negative
    predictions = np.maximum(predictions, 0.0)
    metrics = {
        "mae": mean_absolute_error(y_test, predictions),
        "mse": mean_squared_error(y_test, predictions),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2": r2_score(y_test, predictions),
    }

 
    return {key: _safe_float(value) for key, value in metrics.items()}


