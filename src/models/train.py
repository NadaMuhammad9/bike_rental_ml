from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlflow
import mlflow.lightgbm
import numpy as np
from lightgbm import LGBMRegressor

from src.config import config
from src.data.loader import load_csv_data
from src.data.preprocessor import preprocess_dataframe
from src.features.features import create_features
from src.models.evaluate import evaluate_model
from src.models.model_io import save_model

# Module logger
logger = logging.getLogger(__name__)


def build_model() -> LGBMRegressor:

    # Build a new LightGBM model using config parameters
    return LGBMRegressor(
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        num_leaves=config.num_leaves,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        objective=config.objective,
        random_state=config.random_state,
        verbose=config.verbose,
    )

# Optional paths , use config values when not provided.
def train_model(
    data_path: str | Path | None = None,
    processed_path: str | Path | None = None,
    model_path: str | Path | None = None,
) -> dict[str, Any]:

    # Use provided paths or config defaults
    raw_path = Path(data_path or config.data_path)
    output_processed_path = Path(
        processed_path or config.processed_path
    )
    output_model_path = Path(
        model_path or config.model_path
    )

    # MLflow configuration (sqlite backend , filesystem tracking is deprecated in MLflow 3.x)
    project_root = Path(__file__).resolve().parents[2]
    mlruns_dir = project_root / "mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    tracking_db = mlruns_dir / "mlruns.db"

    mlflow.set_tracking_uri(
        f"sqlite:///{tracking_db.as_posix()}"
    )

    mlflow.set_experiment(
        "lightgbm_time_series"
    )

    # Start MLflow run
    with mlflow.start_run() as run:

        logger.info(
            "MLflow run started: %s",
            run.info.run_id,
        )

        # Load raw data
        logger.info(
            "Loading raw data from %s",
            raw_path,
        )

        df = load_csv_data(raw_path)

        mlflow.log_param(
            "raw_rows",
            len(df),
        )

        mlflow.log_param(
            "raw_columns",
            len(df.columns),
        )

        # Preprocessing
        logger.info(
            "Preprocessing raw data"
        )

        processed = preprocess_dataframe(df)

        # Feature engineering
        logger.info(
            "Creating features"
        )

        featured = create_features(processed)

        # Save processed dataset
        output_processed_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        featured.to_csv(
            output_processed_path,
            index=False,
        )

        mlflow.log_artifact(
            str(output_processed_path),
            artifact_path="data",
        )

        # Check target column
        if config.target_column not in featured.columns:
            raise ValueError(
                f"Target column {config.target_column!r} "
                "not found after preprocessing and feature engineering"
            )

        # Separate features and target. Exclude the date field and any other
        # non-numeric metadata columns so LightGBM only sees model inputs
        X = featured.drop(
            columns=[config.target_column, config.date_column],
            errors="ignore",
        )

        y = featured[
            config.target_column
        ]

        # Time-based train/test split
        train_size = int(
            len(X) * 0.8
        )

        if train_size < 1:
            raise ValueError(
                "Training set must contain at least one row "
                "after splitting"
            )

        X_train = X.iloc[:train_size]
        X_test = X.iloc[train_size:]

        y_train = y.iloc[:train_size]
        y_test = y.iloc[train_size:]

        logger.info(
            "Train size: %s rows",
            len(X_train),
        )

        logger.info(
            "Test size: %s rows",
            len(X_test),
        )

        # Log dataset information
        mlflow.log_params(
            {
                "target_column": config.target_column,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "n_features": X_train.shape[1],
                "train_ratio": 0.8,
            }
        )

        # Build model
        model = build_model()

        # Log LightGBM hyperparameters
        mlflow.log_params(
            {
                "n_estimators": config.n_estimators,
                "learning_rate": config.learning_rate,
                "max_depth": config.max_depth,
                "num_leaves": config.num_leaves,
                "subsample": config.subsample,
                "colsample_bytree": config.colsample_bytree,
                "objective": config.objective,
                "random_state": config.random_state,
            }
        )

        # Train
        logger.info(
            "Training LightGBM model with %s features",
            X_train.shape[1],
        )

        model.fit(
            X_train,
            y_train,
        )

        # Evaluate
        metrics = evaluate_model(
            model,
            X_test,
            y_test,
        )

        logger.info(
            "Evaluation metrics: %s",
            metrics,
        )

        # Log metrics. Guard against NaN/infinite values from edge-case splits so
        # MLflow can safely persist them in the SQLite tracking database.
        mlflow.log_metrics(
            {
                key: float(value)
                if np.isfinite(value)
                else 0.0
                for key, value in metrics.items()
            }
        )

        # Save trained model
        save_model(
            model,
            list(X_train.columns),
            output_model_path,
        )

        logger.info(
            "Saved trained model to %s",
            output_model_path,
        )

        # Log model to MLflow
        mlflow.lightgbm.log_model(
            model,
            artifact_path="model",
        )

        # Log model metadata
        mlflow.set_tags(
            {
                "model_type": "LightGBM",
                "task": "time_series_regression",
                "split_type": "time_based",
            }
        )

        logger.info(
            "MLflow run completed: %s",
            run.info.run_id,
        )

        return {
            "model_path": str(output_model_path),
            "processed_path": str(output_processed_path),
            "metrics": metrics,
            "mlflow_run_id": run.info.run_id,
        }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO
    )

    result = train_model()

    print(result)