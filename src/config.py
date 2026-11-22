from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml

# Get the root directory of the project
ROOT_DIR = Path(__file__).resolve().parent.parent

# Define the directory where configuration files are stored
CONFIG_DIR = ROOT_DIR / "config"


# Load a YAML configuration file and return its contents as a dictionary
def load_yaml_config(path: str | Path) -> dict[str, Any]:
    # Open the YAML file
    with open(path, "r", encoding="utf-8") as handle:
        # Safely load the YAML data or return an empty dictionary if no data exists
        return yaml.safe_load(handle) or {}


# Store and provide access to the configuration settings
class AppConfig:
    def __init__(self) -> None:
        # Load the main configuration
        self.config = load_yaml_config(CONFIG_DIR / "config.yaml")

        # Load the logging configuration
        self.logging = load_yaml_config(CONFIG_DIR / "logging.yaml")

        # Get the raw data path using a default value if it is not configured
        self.data_path = self.config.get("data_path", "data/raw")

        # Get the model path using a default LightGBM model location
        self.model_path = self.config.get("model_path", "models/lightgbm.joblib")

    # Return the random seed used for reproducible results
    @property
    def random_state(self) -> int:
        return int(self.config.get("random_state", 42))

    # Return the number of trees or estimators used by the model
    @property
    def n_estimators(self) -> int:
        return int(self.config.get("n_estimators", 200))

    # Return the maximum depth of the model trees
    @property
    def max_depth(self) -> int:
        return int(self.config.get("max_depth", 6))

    # Return the learning rate used during model training
    @property
    def learning_rate(self) -> float:
        return float(self.config.get("learning_rate", 0.03))

    # Return the maximum number of leaves allowed in each tree
    @property
    def num_leaves(self) -> int:
        return int(self.config.get("num_leaves", 31))

    # Return the fraction of training data sampled for each tree
    @property
    def subsample(self) -> float:
        return float(self.config.get("subsample", 0.8))

    # Return the fraction of features sampled for each tree
    @property
    def colsample_bytree(self) -> float:
        return float(self.config.get("colsample_bytree", 0.8))

    # Return the objective function used by the model
    @property
    def objective(self) -> str:
        return str(self.config.get("objective", "regression"))

    # Return the verbosity level used by the model
    @property
    def verbose(self) -> int:
        return int(self.config.get("verbose", -1))

    # Return the name of the target column in the dataset
    @property
    def target_column(self) -> str:
        return str(self.config.get("target_column", "Rented Bike Count"))

    # Return the name of the date column in the dataset
    @property
    def date_column(self) -> str:
        return str(self.config.get("date_column", "Date"))

    # Return the path where the processed dataset is stored
    @property
    def processed_path(self) -> str:
        return str(self.config.get("processed_path", "data/processed/seoul_bike_processed.csv"))


# Create a global configuration object
config = AppConfig()
