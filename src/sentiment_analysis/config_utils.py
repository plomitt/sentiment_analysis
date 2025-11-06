import argparse
import datetime
from pathlib import Path
from pprint import pprint
import tomllib
from typing import Dict

from sentiment_analysis.utils import setup_logging

logger = setup_logging(__name__)

CONFIG_FILENAME = "config.toml"
DEFAULTS = {
    "query": "bitcoin",
    "article_count": 10,
    "no_content": False,
    "minutes": 5,
    "start_datetime": None,
    "end_datetime": None,
    "max_cycles": None,
    "max_duration_minutes": None,
}


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the pipeline supervisor.

    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description="Run pipeline every N minutes with optional end conditions.")
    parser.add_argument("--minutes", type=int, default=None, help="Interval in minutes between runs")
    parser.add_argument("--query", type=str, default=None, help="Query string for pipeline")
    parser.add_argument("--article_count", type=int, default=None, help="Number of articles per run")
    parser.add_argument("--no_content", action='store_true', default=None, help="Flag: no_content = True (skip content fetching)")
    parser.add_argument("--start_datetime", type=str, default=None, help="ISO datetime at which to start, e.g. 2025-11-06T17:00:00")
    parser.add_argument("--end_datetime", type=str, default=None, help="ISO datetime at which to stop, e.g. 2025-11-06T18:00:00")
    parser.add_argument("--max_cycles", type=int, default=None, help="Maximum number of pipeline runs")
    parser.add_argument("--max_duration_minutes", type=int, default=None, help="Maximum duration in minutes to keep running")
    args = parser.parse_args()
    return args


def get_config() -> Dict[str, str]:
    """
    Get config for the supervisor. Parameter order: config file > CLI > defaults.

    Returns:
        Dict[str, str]: Config dictionary
    """

    # CLI args
    args = parse_args()
    cli_args = {k: getattr(args, k) for k in DEFAULTS.keys()}

    # Config file args
    config_path = Path(__file__).parent / CONFIG_FILENAME
    file_config = {}
    if config_path.exists():
        logger.info("Found config file, loading")
        with open(config_path, "rb") as f:
            cfg = tomllib.load(f)

        file_config = {
            "query": cfg.get("pipeline", {}).get("query"),
            "article_count": cfg.get("pipeline", {}).get("article_count"),
            "no_content": cfg.get("pipeline", {}).get("no_content"),
            "minutes": cfg.get("scheduler", {}).get("minutes"),
            "start_datetime": cfg.get("scheduler", {}).get("start_datetime"),
            "end_datetime": cfg.get("scheduler", {}).get("end_datetime"),
            "max_cycles": cfg.get("scheduler", {}).get("max_cycles"),
            "max_duration_minutes": cfg.get("scheduler", {}).get("max_duration_minutes"),
        }
    else:
        logger.warning("No config file found, using defaults")
    
    # Merge: config file > CLI > defaults
    final_config = {}
    for key, default_val in DEFAULTS.items():
        if file_config.get(key) is not None:
            final_config[key] = file_config[key]
        elif cli_args.get(key) is not None:
            final_config[key] = cli_args[key]
        else:
            final_config[key] = default_val
    
    # Parse date
    if final_config["end_datetime"] is not None:
        final_config["end_datetime"] = datetime.datetime.fromisoformat(final_config["end_datetime"])
    
    return final_config

if __name__ == "__main__":
    config = get_config()
    pprint(config)