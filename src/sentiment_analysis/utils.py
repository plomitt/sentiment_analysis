from __future__ import annotations

from datetime import datetime
import json
import logging
import os
import glob
import sys
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from instructor import Instructor, Mode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_timestamp_from_filename(filepath: str, name_start: str, filetype: str = "json") -> Optional[str]:
    """
    Extract timestamp from a news filename for use in output filename.

    Expected format: [name_start]_[sortable]_[readable].[filetype]
    Example: news_99998238678017_2025-10-24_18-06-22.json

    Args:
        filepath: Full path to the input file

    Returns:
        Timestamp string (sortable_readable) or None if extraction fails
    """
    try:
        print(f'DEBUG: filepath: {filepath}, name_start: {name_start}, filetype: {filetype}')
        filename = os.path.basename(filepath)

        if not (filename.startswith(name_start) and filename.endswith(filetype)):
            logger.warning(f"Filename doesn't match expected pattern: {filename}")
            return None

        # Remove name_start prefix and filetype suffix
        timestamp_part = filename[len(name_start)+1:-len(filetype)+1]

        # Validate that the timestamp part contains the expected format
        if "_" not in timestamp_part or timestamp_part.count("_") < 2:
            logger.warning(f"Timestamp part doesn't contain expected format: {timestamp_part}")
            return None

        logger.info(f"Extracted timestamp from filename: {timestamp_part}")
        return timestamp_part

    except Exception as e:
        logger.error(f"Error extracting timestamp from filename {filepath}: {str(e)}")
        return None

def make_timestamped_filename(input_file: Optional[str] = None, input_name: Optional[str] = None, output_name: str = None, input_filetype: Optional[str] = "json", output_filetype: Optional[str] = "json") -> str:
    # Generate output filename using timestamp from input or generate new one
    # Auto-detected file case - use timestamp from auto-detected sentiment file
    # print(f'DEBUG: input_file: {input_file}, input_name: {input_name}, output_name: {output_name}, filetype: {filetype}')
    if input_file:
        extracted_timestamp = extract_timestamp_from_filename(input_file, input_name, input_filetype)
        base_filename = f"{output_name}_{extracted_timestamp}.{output_filetype}"
    else:
        # Fallback: generate new timestamp if extraction fails
        now = datetime.now()
        sortable_timestamp = f"{99999999999999 - int(now.timestamp())}"
        readable_timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        base_filename = f"{output_name}_{sortable_timestamp}_{readable_timestamp}.{output_filetype}"

    return base_filename