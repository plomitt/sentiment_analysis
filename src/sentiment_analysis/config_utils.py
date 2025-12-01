#!/usr/bin/env python3
"""
Configuration management utilities.

This module provides configuration loading and argument parsing functionality
for the sentiment analysis pipeline, supporting both command-line arguments
and configuration file settings.
"""

from __future__ import annotations

import datetime
import tomllib
from pathlib import Path
from pprint import pprint
from typing import Any, cast

from sentiment_analysis.logging_utils import setup_logging

logger = setup_logging(__name__)

CONFIG_FILENAME = "config.toml"
CONFIG_DEFINITION = {
    "pipeline": {
        "use_similarity_scoring": True,
        "temperature": 0.1,
        "use_reasoning": True,
    },
    "scheduler": {
        "interval_min": 5,
        "start_datetime": None,
        "end_datetime": None,
        "max_cycles": None,
        "max_duration_minutes": None,
    },
    "google":{
        "query": "bitcoin",
        "article_count": 10,
        "no_content": True,
        "use_smart_search": False,
        "request_delay": 0,
    },
    "alpaca": {
        "news_symbols": ["BTCUSD"],
    },
    "telegram": {
        "channels_to_monitor": [],
    },
    "restart_policy": {
        "max_restarts": 3,
        "initial_backoff_seconds": 1.0,
        "backoff_multiplier": 2.0,
        "max_backoff_seconds": 60.0,
        "enable_auto_restart": True,
        "health_check_interval": 5.0,
    }
}


def get_default_values() -> dict[str, Any]:
    """Extract flattened default values from CONFIG_DEFINITION."""
    defaults = {}
    for section_config in CONFIG_DEFINITION.values():
        for key, default_val in section_config.items():
            defaults[key] = default_val
    return defaults


def load_file_config(cfg: dict) -> dict[str, Any]:
    """Load configuration from TOML file using CONFIG_DEFINITION."""
    file_config = {}

    for section_name, section_config in CONFIG_DEFINITION.items():
        config_section = cfg.get(section_name, {})
        for key_name in section_config.keys():
            file_config[key_name] = config_section.get(key_name)

    return file_config


def parse_datetime_fields(config: dict[str, Any]) -> None:
    """Parse datetime string fields to datetime objects."""
    for field in ["start_datetime", "end_datetime"]:
        if config[field] is not None and isinstance(config[field], str):
            config[field] = datetime.datetime.fromisoformat(config[field])


def merge_config(
    file_config: dict[str, Any],
    defaults: dict[str, Any]
) -> dict[str, Any]:
    """Merge configuration from file with defaults.
    
    Args:
        file_config: Configuration loaded from TOML file
        defaults: Default configuration values
        
    Returns:
        Merged configuration where file values override defaults
    """
    merged_config: dict[str, str | int | float | bool | None | datetime.datetime] = {}
    for key, default_val in defaults.items():
        merged_config[key] = cast("str | int | float | bool | None | datetime.datetime", default_val)
        if file_config.get(key) is not None:
            merged_config[key] = file_config[key]
    return merged_config


def load_toml_config() -> dict[str, Any]:
    """Load configuration from TOML file if it exists.
    
    Returns:
        Configuration dictionary from file, or empty dict if file not found.
        Logs appropriate info/warning messages.
    """
    config_path = Path(__file__).parent / CONFIG_FILENAME
    file_config = {}

    if config_path.exists():
        logger.info("Found config file, loading")
        with open(config_path, "rb") as f:
            cfg = tomllib.load(f)
        file_config = load_file_config(cfg)
    else:
        logger.warning("No config file found, using defaults")

    return file_config


def get_config() -> dict[str, str | int | float | bool | None | datetime.datetime]:
    """
    Get config for the supervisor. Parameter order: config file > defaults.

    Returns:
        dict[str, str | int | float | bool | None | datetime.datetime]: Config dictionary with various types.
    """

    # Load configuration from file
    file_config = load_toml_config()

    # Get default values
    defaults = get_default_values()

    # Merge configurations
    final_config = merge_config(file_config, defaults)

    # Parse datetime fields
    parse_datetime_fields(final_config)

    return final_config


# Define a global CONFIG variable for easy access
CONFIG = get_config()

__all__ = [
    "CONFIG",
    "get_config"
]

if __name__ == "__main__":
    pprint(CONFIG)
