import logging
import logging.config
from pathlib import Path
import yaml

"""
Loads and applies the logging configuration from logging.yaml
"""


def setup_logging(config_path: str | Path | None = None) -> logging.Logger:

    # Use the project's default logging configuration if no path is provided
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config" / "logging.yaml"

    # Load logging settings from YAML
    with open(config_path, "r", encoding="utf-8") as handle:
        logging_config = yaml.safe_load(handle) or {}

    # Apply the logging configuration
    logging.config.dictConfig(logging_config)

    # Return the root logger for use across the application
    return logging.getLogger()
