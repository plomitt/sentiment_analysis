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
    "searxng_url": None,
    "request_delay": 0,
    "use_similarity_scoring": False,
    "use_smart_search": False,
    "temperature": 0.1,
    "use_reasoning": None,
    "interval_min": None,
    "start_datetime": None,
    "end_datetime": None,
    "max_cycles": None,
    "max_duration_minutes": None,
}


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the pipeline and the supervisor.

    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description="Run pipeline every N minutes with optional end conditions.")
    parser.add_argument("--query", type=str, default=None, help="Query string for pipeline")
    parser.add_argument("--article_count", type=int, default=None, help="Number of articles per run")
    parser.add_argument("--no_content", action='store_true', default=None, help="Flag: no_content = True (skip content fetching)")
    parser.add_argument("--searxng_url", type=str, default=None, help="SearXNG instance URL (default: from SEARXNG_BASE_URL env var or http://localhost:8080)")
    parser.add_argument("--request_delay", type=int, default=0, help="Delay between SearXNG requests in seconds (default: 0)")
    parser.add_argument("--use_similarity_scoring", action='store_true', default=None, help="Flag: use_similarity_scoring = True (use similarity-based scoring)")
    parser.add_argument("--use_smart_search", action='store_true', default=None, help="Flag: use_smart_search = True (use smart search)")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for sentiment analysis (default: 0.1)")
    parser.add_argument("--use_reasoning", action='store_true', default=None, help="Flag: use_reasoning = True (use reasoning in sentiment analysis)")
    parser.add_argument("--interval_min", type=int, default=None, help="Interval in minutes between runs")
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
            "searxng_url": cfg.get("pipeline", {}).get("searxng_url"),
            "request_delay": cfg.get("pipeline", {}).get("request_delay"),
            "use_similarity_scoring": cfg.get("pipeline", {}).get("use_similarity_scoring"),
            "use_smart_search": cfg.get("pipeline", {}).get("use_smart_search"),
            "temperature": cfg.get("pipeline", {}).get("temperature"),
            "use_reasoning": cfg.get("pipeline", {}).get("use_reasoning"),
            "interval_min": cfg.get("scheduler", {}).get("interval_min"),
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
        final_config[key] = default_val
        if file_config.get(key) is not None:
            final_config[key] = file_config[key]
        if cli_args.get(key) is not None:
            final_config[key] = cli_args[key]
    
    # Parse dates
    if final_config["end_datetime"] is not None:
        final_config["end_datetime"] = datetime.datetime.fromisoformat(final_config["end_datetime"])

    if final_config["start_datetime"] is not None:
        final_config["start_datetime"] = datetime.datetime.fromisoformat(final_config["start_datetime"])
    
    return final_config

__all__ = ["get_config"]

if __name__ == "__main__":
    config = get_config()
    pprint(config)