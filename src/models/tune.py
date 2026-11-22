from __future__ import annotations

import numpy as np
import optuna
from lightgbm import LGBMRegressor, early_stopping
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from src.config import config


def tune_lightgbm(
    X,
    y,
    n_trials: int = 100,
    n_splits: int = 5,
) -> dict[str, float]:

    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)

    def objective(trial: optuna.Trial) -> float:

        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                100,
                1000,
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.005,
                0.1,
                log=True,
            ),
            "max_depth": trial.suggest_int(
                "max_depth",
                3,
                12,
            ),
            "num_leaves": trial.suggest_int(
                "num_leaves",
                10,
                100,
            ),
            "subsample": trial.suggest_float(
                "subsample",
                0.6,
                1.0,
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                0.6,
                1.0,
            ),
            "objective": config.objective,
            "random_state": config.random_state,
            "verbose": -1,
        }

        scores: list[float] = []

        # Time-series cross-validation
        for fold, (train_idx, valid_idx) in enumerate(
            tscv.split(X)
        ):

            X_train = X.iloc[train_idx]
            X_valid = X.iloc[valid_idx]

            y_train = y.iloc[train_idx]
            y_valid = y.iloc[valid_idx]

            model = LGBMRegressor(**params)

            # Train on the current fold
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                callbacks=[
                    early_stopping(
                        stopping_rounds=100,
                        verbose=False,
                    )
                ],
            )

            # Validation predictions
            predictions = model.predict(
                X_valid,
                num_iteration=model.best_iteration_,
            )

            # Validation RMSE
            rmse = float(
                np.sqrt(
                    mean_squared_error(
                        y_valid,
                        predictions,
                    )
                )
            )

            scores.append(rmse)

            # Report fold score to Optuna
            trial.report(rmse, step=fold)

        # Mean RMSE across all folds
        return float(np.mean(scores))

    # Create Optuna study
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=config.random_state,
        ),
    )

    # Start with the initial parameters used during EDA
    study.enqueue_trial(
        {
            "n_estimators": 500,
            "learning_rate": 0.03,
            "max_depth": 6,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }
    )

    # Run hyperparameter optimization
    study.optimize(
        objective,
        n_trials=n_trials,
    )

    return study.best_params