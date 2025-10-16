#!/usr/bin/env python3
"""
Script to convert news.txt file to JSON format.
Each news entry is separated by a blank line and contains fields:
title, body, timestamp, url
"""

import json
import re
from typing import List, Dict


def parse_news_file(file_path: str) -> List[Dict[str, str]]:
    """
    Parse the news.txt file and extract news entries.

    Args:
        file_path: Path to the news.txt file

    Returns:
        List of dictionaries containing news entries
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Split by double newlines to separate entries
    entries = content.strip().split('\n\n')

    news_list = []

    for entry in entries:
        if not entry.strip():
            continue

        lines = entry.strip().split('\n')
        news_entry = {}

        for line in lines:
            line = line.strip()
            if line.startswith('title:'):
                news_entry['title'] = line[6:].strip()
            elif line.startswith('body:'):
                news_entry['body'] = line[5:].strip()
            elif line.startswith('timestamp:'):
                news_entry['timestamp'] = line[10:].strip()
            elif line.startswith('url:'):
                news_entry['url'] = line[4:].strip()

        # Only add entries that have at least a title
        if news_entry.get('title'):
            news_list.append(news_entry)

    return news_list


def save_to_json(data: List[Dict[str, str]], output_file: str) -> None:
    """
    Save the parsed data to a JSON file.

    Args:
        data: List of news entries
        output_file: Path to the output JSON file
    """
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def main():
    """Main function to convert news.txt to JSON."""
    input_file = 'news.txt'
    output_file = 'news.json'

    print(f"Reading news from {input_file}...")

    try:
        news_data = parse_news_file(input_file)

        print(f"Found {len(news_data)} news entries")

        save_to_json(news_data, output_file)

        print(f"Successfully converted to {output_file}")

        # Print a sample entry
        if news_data:
            print("\nSample entry:")
            print(json.dumps(news_data[0], indent=2, ensure_ascii=False))

    except FileNotFoundError:
        print(f"Error: {input_file} not found")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()