#!/usr/bin/env python3
"""
Script to convert news.json timestamps to Unix timestamps and sort entries chronologically.
"""

import json
import re
from datetime import datetime
import sys


def parse_js_timestamp(timestamp_str):
    """
    Parse JavaScript-style timestamp to Unix timestamp.

    Example input: "Thu Oct 16 2025 15:14:00 GMT+0200 (Central European Summer Time)"
    """
    # Extract the main part without the timezone name in parentheses
    # "Thu Oct 16 2025 15:14:00 GMT+0200"
    main_part = timestamp_str.split(' (')[0]

    # Remove "GMT" from the timezone part to make it parseable
    # "Thu Oct 16 2025 15:14:00 +0200"
    cleaned_part = main_part.replace('GMT', '')

    # Parse using datetime.strptime
    # Format: "%a %b %d %Y %H:%M:%S %z"
    try:
        dt = datetime.strptime(cleaned_part, "%a %b %d %Y %H:%M:%S %z")
        return int(dt.timestamp())
    except ValueError as e:
        print(f"Error parsing timestamp '{timestamp_str}': {e}")
        print(f"Cleaned part: '{cleaned_part}'")
        return None


def process_news_file(input_file, output_file):
    """
    Process news.json file to add Unix timestamps and sort entries.

    Args:
        input_file (str): Path to input news.json file
        output_file (str): Path to output JSON file
    """
    print(f"Reading from: {input_file}")

    # Read the input file
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            news_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{input_file}': {e}")
        return

    print(f"Loaded {len(news_data)} news entries")

    # Process each entry to add Unix timestamp
    processed_entries = []
    for i, entry in enumerate(news_data):
        print(f"Processing entry {i+1}/{len(news_data)}: {entry.get('title', 'Unknown title')[:50]}...")

        timestamp_str = entry.get('timestamp')
        if not timestamp_str:
            print(f"Warning: Entry {i+1} has no timestamp field")
            continue

        unix_timestamp = parse_js_timestamp(timestamp_str)
        if unix_timestamp is None:
            print(f"Warning: Could not parse timestamp for entry {i+1}")
            continue

        # Create a new entry with the Unix timestamp added
        new_entry = entry.copy()
        new_entry['unix_timestamp'] = unix_timestamp
        processed_entries.append(new_entry)

    print(f"Successfully processed {len(processed_entries)} entries")

    # Sort entries by Unix timestamp (oldest first)
    sorted_entries = sorted(processed_entries, key=lambda x: x['unix_timestamp'])

    # Write to output file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sorted_entries, f, indent=2, ensure_ascii=False)
        print(f"Successfully wrote sorted entries to: {output_file}")
    except Exception as e:
        print(f"Error writing to output file: {e}")
        return

    # Display some statistics
    if sorted_entries:
        oldest_entry = sorted_entries[0]
        newest_entry = sorted_entries[-1]

        print(f"\n--- Statistics ---")
        print(f"Total entries: {len(sorted_entries)}")
        print(f"Oldest entry: {oldest_entry.get('title', 'Unknown')[:50]}...")
        print(f"Oldest timestamp: {oldest_entry.get('timestamp')}")
        print(f"Oldest Unix timestamp: {oldest_entry.get('unix_timestamp')}")
        print(f"Newest entry: {newest_entry.get('title', 'Unknown')[:50]}...")
        print(f"Newest timestamp: {newest_entry.get('timestamp')}")
        print(f"Newest Unix timestamp: {newest_entry.get('unix_timestamp')}")


def main():
    """Main function to run the script."""
    input_file = "src/sentiment_analysis/news.json"
    output_file = "news_sorted_with_unix_timestamps.json"

    print("=== News Timestamp Converter ===")
    print("Converting timestamps to Unix format and sorting entries...")

    process_news_file(input_file, output_file)

    print("\nDone!")


if __name__ == "__main__":
    main()