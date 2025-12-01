#!/usr/bin/env python3
"""
Utility functions for sentiment analysis application.

This module provides common utility functions for file operations, logging,
timestamp handling, and JSON data management used throughout the application.
"""

from __future__ import annotations

import calendar
import glob
import json
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from dotenv import load_dotenv

from sentiment_analysis.embedding_model import truncate_text_to_model_limit
from sentiment_analysis.logging_utils import setup_logging

logger = setup_logging(__name__)


def extract_timestamp_from_filename(filepath: str, name_start: str) -> str | None:
    """
    Extract timestamp from a news filename for use in output filename.

    Expected format: [name_start]_[sortable]_[readable].[filetype]
    Example: news_99998238678017_2025-10-24_18-06-22.json

    Args:
        filepath: Full path to the input file.
        name_start: Expected filename prefix (e.g., "news", "sentiments").

    Returns:
        Timestamp string (sortable_readable) or None if extraction fails.
    """

    try:
        logger.debug(f"Extracting timestamp from: {filepath}, name_start: {name_start}")
        filename = os.path.basename(filepath)

        if not filename.startswith(name_start):
            logger.warning(f"Filename doesn't match expected pattern: {filename}")
            return None

        # Remove name_start prefix and filetype suffix
        name_without_extension = filename.rsplit(".", 1)[0]
        timestamp_part = name_without_extension[len(name_start) + 1 :]

        # Validate that the timestamp part contains the expected format
        if "_" not in timestamp_part or timestamp_part.count("_") < 2:
            logger.warning(
                f"Timestamp part doesn't contain expected format: {timestamp_part}"
            )
            return None

        logger.info(f"Extracted timestamp from filename: {timestamp_part}")
        return timestamp_part

    except Exception as e:
        logger.error(f"Error extracting timestamp from filename {filepath}: {e!s}")
        return None


def make_timestamped_filename(
    input_file: str | None = None,
    input_name: str | None = None,
    output_name: str | None = None,
    output_filetype: str = "json",
) -> str | None:
    """
    Generate timestamped filename using timestamp from input file or generate new one.

    Args:
        input_file: Path to input file to extract timestamp from.
        input_name: Expected filename prefix for timestamp extraction.
        output_name: Desired output filename prefix.
        output_filetype: Desired output file extension (defaults to "json").

    Returns:
        Generated filename with timestamp, or None if generation fails.
    """
    try:
        timestamp = None

        if input_file and input_name:
            timestamp = extract_timestamp_from_filename(input_file, input_name)
            logger.info("Timestamp extracted from input file")

        if not timestamp:
            # Fallback: generate new timestamp if extraction fails
            now = datetime.now()
            sortable_timestamp = f"{99999999999999 - int(now.timestamp())}"
            readable_timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
            timestamp = f"{sortable_timestamp}_{readable_timestamp}"
            logger.info("Timestamp generated from current time")

        if output_name is None:
            raise ValueError("output_name must be provided")

        base_filename = f"{output_name}_{timestamp}.{output_filetype}"
        return base_filename

    except Exception as e:
        logger.error(f"Error making timestamped filename: {e!s}")
        return None


def load_json_data(file_path: str) -> Any:
    """
    Generic JSON file loading with standardized error handling.

    Args:
        file_path: Path to the JSON file to load.

    Returns:
        Loaded JSON data.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        json.JSONDecodeError: If the file contains invalid JSON.
        Exception: For other file reading errors.
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        logger.debug(f"Successfully loaded JSON data from {file_path}")
        return data

    except FileNotFoundError:
        error_msg = f"File not found: {file_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON in file {file_path}: {e!s}"
        logger.error(error_msg)
        raise json.JSONDecodeError(error_msg, e.doc, e.pos)
    except Exception as e:
        error_msg = f"Error loading file {file_path}: {e!s}"
        logger.error(error_msg)
        raise Exception(error_msg)


def save_json_data(
    data: Any,
    file_path: str,
    indent: int = 4,
    ensure_ascii: bool = False,
) -> None:
    """
    Generic JSON file saving with standardized formatting and error handling.

    Args:
        data: Data to save as JSON.
        file_path: Path where to save the JSON file.
        indent: JSON indentation (defaults to 4).
        ensure_ascii: Whether to escape non-ASCII characters (defaults to False).

    Raises:
        Exception: For file writing errors.
    """
    try:
        # Ensure directory exists
        directory = os.path.dirname(file_path)
        if directory:
            ensure_directory(directory)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)

        logger.info(f"Successfully saved JSON data to {file_path}")

    except Exception as e:
        error_msg = f"Error saving file {file_path}: {e!s}"
        logger.error(error_msg)
        raise Exception(error_msg)


def find_latest_file(
    directory: str,
    pattern_prefix: str,
    file_extension: str = "json",
) -> str | None:
    """
    Generic function to find the latest file by pattern matching.

    Args:
        directory: Directory to search in.
        pattern_prefix: File name prefix pattern (e.g., "news", "sentiments").
        file_extension: File extension to match (defaults to "json").

    Returns:
        Path to the latest file, or None if no files found.
    """
    try:
        pattern = os.path.join(directory, f"{pattern_prefix}_*.{file_extension}")
        files = glob.glob(pattern)

        if not files:
            logger.warning(f"No {pattern_prefix} files found in {directory}")
            return None

        # Sort files alphabetically (reverse chronological naming)
        files.sort()
        latest_file = files[0]

        logger.debug(f"Found latest {pattern_prefix} file: {latest_file}")

        return latest_file

    except Exception as e:
        error_msg = f"Error finding latest {pattern_prefix} file in {directory}: {e!s}"
        logger.error(error_msg)
        return None


def ensure_directory(directory_path: str) -> None:
    """
    Ensure directory exists with consistent error handling.

    Args:
        directory_path: Path to the directory to create.

    Raises:
        Exception: For directory creation errors.
    """
    try:
        path = Path(directory_path)
        path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        error_msg = f"Error creating directory {directory_path}: {e!s}"
        raise Exception(error_msg)


def validate_env_config(required_vars: list[str]) -> bool:
    """
    Validate that required environment variables are set.

    Args:
        required_vars: List of required environment variable names.

    Returns:
        bool: True if all required environment variables are present, False otherwise.
    """
    load_dotenv()

    missing_vars = []

    # Check required variables
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        return False

    logger.debug("Environment variables configuration validated.")
    return True


def convert_google_rss_to_iso(rss_parsed: time.struct_time) -> str:
    """
    Convert Google RSS timestamp to ISO 8601 UTC format.

    Args:
        rss_parsed: Parsed time.struct_time from feedparser

    Returns:
        ISO 8601 UTC string (e.g., "2025-11-11T19:15:21Z")
    """
    dt = datetime(*rss_parsed[:6], tzinfo=UTC)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def convert_google_rss_to_unix(rss_parsed: time.struct_time) -> int:
    """
    Convert Google RSS timestamp to Unix epoch time.

    Args:
        rss_parsed: Parsed time.struct_time from feedparser

    Returns:
        Unix epoch timestamp (seconds since 1970-01-01)
    """
    return calendar.timegm(rss_parsed)


def convert_alpaca_to_iso(alpaca_input: str | datetime) -> str:
    """
    Convert Alpaca news timestamp to ISO 8601 UTC format.

    Args:
        alpaca_input: Alpaca timestamp string (e.g., "2025-11-13T16:01:45Z" or "2025-11-14T19:15:49+00:00")
        or datetime object

    Returns:
        ISO 8601 UTC string (e.g., "2025-11-13T16:01:45Z")
    """
    if isinstance(alpaca_input, datetime):
        dt = alpaca_input
    else:
        # Parse string input
        try:
            # Try parsing with Z suffix first
            dt = datetime.strptime(alpaca_input, "%Y-%m-%dT%H:%M:%SZ")
            dt = dt.replace(tzinfo=UTC)
        except ValueError:
            # Try parsing with timezone offset
            dt = datetime.fromisoformat(alpaca_input)

    # Convert to UTC if needed
    if dt.tzinfo is not None:
        dt = dt.astimezone(UTC)
    else:
        dt = dt.replace(tzinfo=UTC)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def convert_alpaca_to_unix(alpaca_input: str | datetime) -> int:
    """
    Convert Alpaca news timestamp to Unix epoch time.

    Args:
        alpaca_input: Alpaca timestamp string (e.g., "2025-11-13T16:01:45Z" or "2025-11-14T19:15:49+00:00") 
        or datetime object

    Returns:
        Unix epoch timestamp (seconds since 1970-01-01)
    """
    if isinstance(alpaca_input, datetime):
        dt = alpaca_input
    else:
        # Parse string input
        try:
            # Try parsing with Z suffix first
            dt = datetime.strptime(alpaca_input, "%Y-%m-%dT%H:%M:%SZ")
            dt = dt.replace(tzinfo=UTC)
        except ValueError:
            # Try parsing with timezone offset
            dt = datetime.fromisoformat(alpaca_input)

    # Convert to UTC if needed
    if dt.tzinfo is not None:
        dt = dt.astimezone(UTC)
    else:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp())


def clean_up_body_text(text: str) -> str:
    """
    Clean up article body text by removing HTML tags and unwanted elements.

    Args:
        text: Raw article body text.

    Returns:
        str: Cleaned article body text.
    """
    # Step 1: Parse HTML
    soup = BeautifulSoup(text, "html.parser")

    # Step 2: Remove unwanted elements
    for tag in soup(["script", "style", "table", "img", "a"]):
        tag.decompose()

    # Step 3: Extract text
    clean_text = soup.get_text(separator=" ", strip=True)

    # Step 4: Clean up whitespace and artifacts
    clean_text = re.sub(r"\s+", " ", clean_text)

    return clean_text


def make_embedding_text(article: dict[str, Any]) -> str:
    """
    Create embedding text from article title and body.

    Combines title and body text, truncating to fit within the model's maximum token limit
    to create suitable input for text embedding models.

    Args:
        article: Article dictionary containing title and body keys.

    Returns:
        str: Combined text from title and body, truncated to 1000 characters.
    """
    title = article.get("title", "")
    body = article.get("body", "")
    text = f"{title} {body}"
    truncated_text = truncate_text_to_model_limit(text)
    return truncated_text


# Define the public API for this module
__all__ = [
    "clean_up_body_text",
    "convert_alpaca_to_iso",
    "convert_alpaca_to_unix",
    "convert_google_rss_to_iso",
    "convert_google_rss_to_unix",
    "ensure_directory",
    "extract_timestamp_from_filename",
    "find_latest_file",
    "load_json_data",
    "make_embedding_text",
    "make_timestamped_filename",
    "save_json_data",
    "setup_logging",
    "validate_env_config",
]