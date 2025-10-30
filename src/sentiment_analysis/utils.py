from __future__ import annotations

from datetime import datetime
import json
import logging
import os
import glob
import sys
from typing import List, Optional, Dict, Any
from pathlib import Path
from pydantic import BaseModel, Field, field_validator
from instructor import Instructor, Mode

def extract_timestamp_from_filename(filepath: str, name_start: str, logger: logging.Logger = None) -> Optional[str]:
    """
    Extract timestamp from a news filename for use in output filename.

    Expected format: [name_start]_[sortable]_[readable].[filetype]
    Example: news_99998238678017_2025-10-24_18-06-22.json

    Args:
        filepath: Full path to the input file
        name_start: Expected filename prefix (e.g., "news", "sentiments")
        filetype: Expected file extension (defaults to "json")
        logger: Optional logger instance for logging

    Returns:
        Timestamp string (sortable_readable) or None if extraction fails
    """
    try:
        logger.debug(f'Extracting timestamp from: {filepath}, name_start: {name_start}')
        filename = os.path.basename(filepath)

        if not (filename.startswith(name_start)):
            logger.warning(f"Filename doesn't match expected pattern: {filename}")
            return None

        # Remove name_start prefix and filetype suffix
        name_without_extension = filename.rsplit('.', 1)[0]
        timestamp_part = name_without_extension[len(name_start)+1:]

        # Validate that the timestamp part contains the expected format
        if "_" not in timestamp_part or timestamp_part.count("_") < 2:
            logger.warning(f"Timestamp part doesn't contain expected format: {timestamp_part}")
            return None

        logger.info(f"Extracted timestamp from filename: {timestamp_part}")
        return timestamp_part

    except Exception as e:
        logger.error(f"Error extracting timestamp from filename {filepath}: {str(e)}")
        return None

def make_timestamped_filename(input_file: Optional[str] = None, input_name: Optional[str] = None, output_name: str = None, output_filetype: Optional[str] = "json", logger: logging.Logger = None) -> str:
    """
    Generate timestamped filename using timestamp from input file or generate new one.

    Args:
        input_file: Path to input file to extract timestamp from
        input_name: Expected filename prefix for timestamp extraction
        output_name: Desired output filename prefix
        input_filetype: Expected input file extension (defaults to "json")
        output_filetype: Desired output file extension (defaults to "json")
        logger: Optional logger instance for logging

    Returns:
        Generated filename with timestamp
    """
    if input_file:
        extracted_timestamp = extract_timestamp_from_filename(input_file, input_name, logger)
        if extracted_timestamp:
            base_filename = f"{output_name}_{extracted_timestamp}.{output_filetype}"
        else:
            # Fallback: generate new timestamp if extraction fails
            now = datetime.now()
            sortable_timestamp = f"{99999999999999 - int(now.timestamp())}"
            readable_timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
            base_filename = f"{output_name}_{sortable_timestamp}_{readable_timestamp}.{output_filetype}"
    else:
        # Fallback: generate new timestamp if extraction fails
        now = datetime.now()
        sortable_timestamp = f"{99999999999999 - int(now.timestamp())}"
        readable_timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        base_filename = f"{output_name}_{sortable_timestamp}_{readable_timestamp}.{output_filetype}"

    return base_filename

def setup_logging(name: str = None, level: int = logging.INFO,
                  format_string: str = None) -> logging.Logger:
    """
    Standardized logging setup with optional custom format.

    Args:
        name: Logger name (defaults to __name__ of calling module)
        level: Logging level (defaults to INFO)
        format_string: Custom format string (defaults to standard format)

    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = '%(asctime)s - %(levelname)s - %(message)s'

    # Only configure basic logging if not already configured
    if not logging.getLogger().handlers:
        logging.basicConfig(level=level, format=format_string)

    logger_name = name if name else __name__
    return logging.getLogger(logger_name)


def load_json_data(file_path: str, logger: logging.Logger = None) -> Any:
    """
    Generic JSON file loading with standardized error handling.

    Args:
        file_path: Path to the JSON file to load
        logger: Optional logger instance for error reporting

    Returns:
        Loaded JSON data

    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
        Exception: For other file reading errors
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger.debug(f"Successfully loaded JSON data from {file_path}")
        return data

    except FileNotFoundError:
        error_msg = f"File not found: {file_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON in file {file_path}: {str(e)}"
        logger.error(error_msg)
        raise json.JSONDecodeError(error_msg, e.doc, e.pos)
    except Exception as e:
        error_msg = f"Error loading file {file_path}: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)


def save_json_data(data: Any, file_path: str, logger: logging.Logger = None, indent: int = 4, ensure_ascii: bool = False) -> None:
    """
    Generic JSON file saving with standardized formatting and error handling.

    Args:
        data: Data to save as JSON
        file_path: Path where to save the JSON file
        logger: Optional logger instance for error reporting
        indent: JSON indentation (defaults to 4)
        ensure_ascii: Whether to escape non-ASCII characters (defaults to False)

    Raises:
        Exception: For file writing errors
    """
    try:
        # Ensure directory exists
        directory = os.path.dirname(file_path)
        if directory:
            ensure_directory(directory)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)

        logger.info(f"Successfully saved JSON data to {file_path}")

    except Exception as e:
        error_msg = f"Error saving file {file_path}: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)


def find_latest_file(directory: str, pattern_prefix: str, file_extension: str = "json", logger: logging.Logger = None) -> Optional[str]:
    """
    Generic function to find the latest file by pattern matching.

    Args:
        directory: Directory to search in
        pattern_prefix: File name prefix pattern (e.g., "news", "sentiments")
        file_extension: File extension to match (defaults to "json")
        logger: Optional logger instance for error reporting

    Returns:
        Path to the latest file, or None if no files found
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
        error_msg = f"Error finding latest {pattern_prefix} file in {directory}: {str(e)}"
        logger.error(error_msg)
        return None


def ensure_directory(directory_path: str) -> None:
    """
    Ensure directory exists with consistent error handling.

    Args:
        directory_path: Path to the directory to create

    Raises:
        Exception: For directory creation errors
    """
    try:
        path = Path(directory_path)
        path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        error_msg = f"Error creating directory {directory_path}: {str(e)}"
        raise Exception(error_msg)