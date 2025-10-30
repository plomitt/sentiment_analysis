#!/usr/bin/env python3
"""
Sentiment Grapher Script

Creates high-quality time series graphs of sentiment analysis data from JSON files.
Supports rolling averages, time window filtering, and multi-chart output for large datasets.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, Union, Optional, List, Dict, Any

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import base64
import io

from sentiment_analysis.utils import (
    make_timestamped_filename,
    setup_logging, load_json_data, find_latest_file, ensure_directory
)

# Configure logging
logger = setup_logging(__name__)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Create sentiment analysis graphs from JSON data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            python -m sentiment_analysis.sentiment_grapher  # Auto-detect latest sentiment file
            python -m sentiment_analysis.sentiment_grapher --input-file custom.json  # Manual file
            python -m sentiment_analysis.sentiment_grapher --interval-minutes all
            python -m sentiment_analysis.sentiment_grapher --window-minutes 10 --interval-minutes 120
        """
    )

    parser.add_argument(
        '--input-file',
        help='Path to JSON file containing sentiment analysis data (optional - auto-detects if not provided)'
    )

    parser.add_argument(
        '--input-dir',
        default='src/sentiment_analysis/sentiments/',
        help='Directory containing sentiment analysis files (default: src/sentiment_analysis/sentiments/)'
    )

    parser.add_argument(
        '--window-minutes',
        type=int,
        default=5,
        help='Rolling average window in minutes (default: 5)'
    )

    parser.add_argument(
        '--interval-minutes',
        default='60',
        help='Time window for data (minutes, "all", or 0 for full dataset, default: 60)'
    )

    parser.add_argument(
        '--output',
        default='default',
        help='Output image filename (default: chart_<timestamp>.png)'
    )

    parser.add_argument(
        '--title',
        help='Custom chart title'
    )

    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='Image resolution in DPI (default: 300)'
    )

    parser.add_argument(
        '--max-points',
        type=int,
        default=400,
        help='Maximum data points per chart (default: 400)'
    )

    parser.add_argument(
        '--output-dir',
        default='src/sentiment_analysis/charts',
        help='Directory for chart output (default: src/sentiment_analysis/charts)'
    )

    args = parser.parse_args()
    return args

def parse_interval_minutes(interval_minutes):
    # Parse interval_minutes
    try:
        if str(interval_minutes).lower() == 'all':
            return 'all'

        interval_minutes = int(interval_minutes)
        return interval_minutes
    except ValueError:
        print("Error: interval_minutes must be a number, 'all', or 0")
        return None

def determine_input_file(base_input_file, input_dir):
    # Determine input file
    if base_input_file:
        # Manual file specified
        input_file = base_input_file
        print(f"📁 Using manually specified input file: {base_input_file}")
    else:
        # Auto-detect latest sentiment file
        print("🔍 Auto-detecting latest sentiment file...")
        latest_file = find_latest_file(input_dir, "sentiments", "json", logger)
        if not latest_file:
            print("❌ Error: No sentiment files found in src/sentiment_analysis/sentiments/")
            print("Please run sentiment analyzer first to generate sentiment files.")
            return None
        input_file = latest_file
    
    print(f"📁 Input file: {input_file}")
    return input_file

def load_sentiment_data(json_file: str) -> List[Dict[str, Any]]:
    """Load and parse sentiment analysis data from JSON file."""
    try:
        data = load_json_data(json_file, logger)

        if not data:
            raise ValueError("JSON file is empty")

        # Extract relevant data into DataFrame
        records = []
        for item in data:
            if 'sentiment' not in item:
                continue

            record = {
                'title': item.get('title', ''),
                'timestamp': item.get('timestamp', ''),
                'unix_timestamp': item.get('unix_timestamp', 0),
                'sentiment_score': item['sentiment'].get('score', 5.0),
                'sentiment_reasoning': item['sentiment'].get('reasoning', ''),
                'url': item.get('url', '')
            }
            records.append(record)

        if not records:
            raise ValueError("No valid sentiment data found in JSON file")

        print(f"Loaded {len(records)} sentiment records from {json_file}")
        return records

    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {json_file}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")
    except Exception as e:
        raise ValueError(f"Error loading data: {e}")


def convert_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Convert unix timestamps to datetime objects."""
    df = df.copy()

    if 'unix_timestamp' not in df.columns:
        raise ValueError("JSON file must contain 'unix_timestamp' column")

    # Convert unix timestamps to datetime
    df['datetime'] = pd.to_datetime(df['unix_timestamp'], unit='s', errors='coerce')

    # Drop rows with invalid timestamps
    initial_count = len(df)
    df = df.dropna(subset=['datetime'])
    dropped_count = initial_count - len(df)

    if dropped_count > 0:
        print(f"Warning: Dropped {dropped_count} records with invalid timestamps")

    # Sort by datetime
    df = df.sort_values('datetime').reset_index(drop=True)

    return df


def filter_data_by_time(df: pd.DataFrame, interval_minutes: Union[str, int]) -> pd.DataFrame:
    """Filter data to include only entries within specified time window from latest entry."""
    if interval_minutes in ('all', 'ALL') or interval_minutes == 0:
        print(f"Using all {len(df)} records (no time filtering)")
        return df

    if not isinstance(interval_minutes, int) or interval_minutes < 0:
        raise ValueError("interval_minutes must be a positive integer, 'all', or 0")

    # Ensure datetime column exists
    if 'datetime' not in df.columns:
        raise ValueError("DataFrame missing 'datetime' column")

    # Get the latest timestamp
    latest_time = df['datetime'].max()
    cutoff_time = latest_time - timedelta(minutes=interval_minutes)

    # Filter data
    filtered_df = df[df['datetime'] >= cutoff_time].copy()

    print(f"Filtered to {len(filtered_df)} records from last {interval_minutes} minutes")
    print(f"Time range: {filtered_df['datetime'].min()} to {filtered_df['datetime'].max()}")

    return filtered_df


def calculate_discrete_rolling_average(df: pd.DataFrame, window_minutes: int) -> pd.DataFrame:
    """Calculate discrete time-binned rolling averages for step-wise visualization."""
    if window_minutes <= 0:
        raise ValueError("window_minutes must be a positive integer")

    df = df.copy().sort_values('datetime')

    # Create time bins starting from the earliest timestamp
    start_time = df['datetime'].min()
    end_time = df['datetime'].max()

    # Round start time down to nearest window boundary
    start_time_bin = start_time.floor(f'{window_minutes}min')

    # Generate all time bins
    time_bins = []
    current_time = start_time_bin
    while current_time <= end_time:
        time_bins.append(current_time)
        current_time += pd.Timedelta(minutes=window_minutes)

    if not time_bins:
        time_bins.append(start_time_bin)

    # Calculate average sentiment for each time bin
    bin_averages = []
    bin_centers = []

    for i, bin_start in enumerate(time_bins):
        bin_end = bin_start + pd.Timedelta(minutes=window_minutes)

        # Get all data points within this bin
        mask = (df['datetime'] >= bin_start) & (df['datetime'] < bin_end)
        bin_data = df[mask]

        if not bin_data.empty:
            avg_sentiment = bin_data['sentiment_score'].mean()
        else:
            # If no data in this bin, use the previous bin's average
            avg_sentiment = bin_averages[-1] if bin_averages else 5.0  # Default neutral

        bin_averages.append(avg_sentiment)
        bin_centers.append(bin_start + pd.Timedelta(minutes=window_minutes/2))

    # Create discrete step data for plotting
    # We need to create points that will make step() function work correctly
    step_times = []
    step_values = []

    for i, (bin_start, avg_value) in enumerate(zip(time_bins, bin_averages)):
        if i == 0:
            # First bin starts at bin_start
            step_times.append(bin_start)
            step_values.append(avg_value)
        else:
            # For step function, we need the value to change at bin boundary
            step_times.append(bin_start)
            step_values.append(avg_value)

    # Add the final point to complete the last step
    if time_bins:
        final_time = time_bins[-1] + pd.Timedelta(minutes=window_minutes)
        step_times.append(final_time)
        step_values.append(bin_averages[-1])

    # Create result DataFrame
    result_df = df.copy()
    result_df['discrete_avg'] = pd.Series(step_values, index=pd.to_datetime(step_times))

    # Store the bin information for plotting
    result_df['window_minutes'] = window_minutes
    result_df.attrs['step_times'] = step_times
    result_df.attrs['step_values'] = step_values

    return result_df


def determine_chart_split(df: pd.DataFrame, max_points: int = 400) -> Tuple[int, int]:
    """Determine if data needs to be split into multiple charts."""
    total_points = len(df)

    if total_points <= max_points:
        return 1, total_points

    # Calculate number of charts needed
    num_charts = (total_points + max_points - 1) // max_points
    points_per_chart = (total_points + num_charts - 1) // num_charts

    return num_charts, points_per_chart


def create_sentiment_chart(
    df: pd.DataFrame,
    chart_num: int = 1,
    total_charts: int = 1,
    title: str = None,
    dpi: int = 300
) -> str:
    """Create a high-quality sentiment chart and return as base64 string."""
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(14, 8), dpi=dpi)
    
    # Plot sentiment scores
    ax.plot(
        df['datetime'],
        df['sentiment_score'],
        'o-',
        linewidth=1.5,
        markersize=4,
        alpha=0.7,
        color='#2E86AB',
        label='Sentiment Score',
        zorder=3
    )
    
    # Plot discrete rolling average using step function
    if 'step_times' in df.attrs and 'step_values' in df.attrs:
        step_times = df.attrs['step_times']
        step_values = df.attrs['step_values']
        window_minutes = df.iloc[0].get('window_minutes', 'N/A')
        ax.step(
            step_times,
            step_values,
            '-',
            linewidth=3,
            where='post',
            color='#A23B72',
            label=f'Discrete Average ({window_minutes} min bins)',
            zorder=4
        )
        
        # Add vertical lines at bin boundaries for clarity
        # Only show lines that are within the current data range
        data_start = df['datetime'].min()
        data_end = df['datetime'].max()
        for boundary_time in step_times[1:-1]:  # Skip first and last to avoid edge clutter
            if data_start <= boundary_time <= data_end:
                ax.axvline(
                    x=boundary_time,
                    color='#CCCCCC',
                    linestyle='--',
                    linewidth=0.8,
                    alpha=0.4,
                    zorder=1
                )
    else:
        # Fallback to regular line if step data not available
        ax.plot(
            df['datetime'],
            df.get('discrete_avg', df.get('rolling_avg', df['sentiment_score'])),
            '-',
            linewidth=3,
            color='#A23B72',
            label=f'Average ({df.iloc[0].get("window_minutes", "N/A")} min)',
            zorder=4
        )
    
    # Customize chart appearance
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('white')
    
    # Set title
    if title:
        chart_title = title
    else:
        # Get start and end datetime objects
        start_dt = df['datetime'].min()
        end_dt = df['datetime'].max()
        # Check if start and end are on the same day
        if start_dt.date() == end_dt.date():
            # Same day: "Oct 16, 09:30 - 14:45"
            time_range = f"{start_dt.strftime('%b %d, %H:%M')} - {end_dt.strftime('%H:%M')}"
        else:
            # Different days: "Oct 15, 16:30 - Oct 16, 10:15"
            time_range = f"{start_dt.strftime('%b %d, %H:%M')} - {end_dt.strftime('%b %d, %H:%M')}"
        chart_title = f"Sentiment Analysis - {time_range}"
    
    if total_charts > 1:
        chart_title += f" (Chart {chart_num}/{total_charts})"
    
    ax.set_title(chart_title, fontsize=16, fontweight='bold', pad=20)
    
    # Set labels
    ax.set_xlabel('Time', fontsize=12, fontweight='600')
    ax.set_ylabel('Sentiment Score (1-10)', fontsize=12, fontweight='600')
    
    # Set y-axis limits
    ax.set_ylim(0.5, 10.5)
    ax.set_yticks(range(1, 11))
    
    # Add sentiment zones
    ax.axhspan(1, 4, alpha=0.1, color='red', label='Bearish Zone')
    ax.axhspan(4, 7, alpha=0.1, color='yellow', label='Neutral Zone')
    ax.axhspan(7, 10, alpha=0.1, color='green', label='Bullish Zone')
    
    # Format x-axis - improved tick handling
    time_span = df['datetime'].max() - df['datetime'].min()
    if time_span.total_seconds() <= 3600:  # Less than 1 hour
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
    elif time_span.total_seconds() <= 14400:  # Less than 4 hours
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
    else:  # More than 4 hours
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Legend
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    
    # Add data info
    info_text = f"Data points: {len(df)}"
    if 'window_minutes' in df.iloc[0]:
        info_text += f" | Window: {df.iloc[0]['window_minutes']}min"
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Layout
    plt.tight_layout()
    
    # Save to buffer and convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    
    # Convert to base64 string
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return image_base64


def create_charts(
    df: pd.DataFrame,
    max_points: int = 400,
    title: str = None,
    dpi: int = 300,
) -> list:
    """Create multiple charts for large datasets and return as base64 list."""
    total_charts, points_per_chart = determine_chart_split(df, max_points)
    
    images = []
    
    if total_charts == 1:
        image_base64 = create_sentiment_chart(df, 1, 1, title, dpi)
        images.append(image_base64)
        return images
    
    # Calculate overlap (10% of chart size)
    overlap = max(1, points_per_chart // 10)
    
    for i in range(total_charts):
        # Calculate start and end indices with overlap
        start_idx = max(0, i * points_per_chart - overlap)
        end_idx = min(len(df), (i + 1) * points_per_chart + overlap)
        
        # Extract data for this chart
        chart_df = df.iloc[start_idx:end_idx].copy()
        
        # Create chart and get base64
        image_base64 = create_sentiment_chart(chart_df, i+1, total_charts, title, dpi)
        images.append(image_base64)
    
    return images



def generate_sentiment_charts(records, window_minutes=5, interval_minutes="60", title=None, dpi=300, max_points=400):
    """
    Generate sentiment charts with explicit parameters and return base64 images.
    
    This function contains the core logic for creating sentiment analysis charts.
    It processes sentiment data, applies time windows and rolling averages,
    and generates multiple chart visualizations returned as base64 strings.
    
    Args:
        records: List of sentiment records
        window_minutes: Rolling average window in minutes (default: 5)
        interval_minutes: Time window for data (minutes, "all", or 0 for full dataset, default: "60")
        title: Custom chart title (optional)
        dpi: Image resolution in DPI (default: 300)
        max_points: Maximum data points per chart (default: 400)
    
    Returns:
        list: List of base64-encoded PNG images
    """
    try:
        df = pd.DataFrame(records)

        print("Converting timestamps...")
        df = convert_timestamps(df)

        print("Filtering data by time window...")
        filtered_df = filter_data_by_time(df, interval_minutes)

        if filtered_df.empty:
            print("No data available for the specified time window")
            return None

        print("Calculating discrete rolling averages...")
        processed_df = calculate_discrete_rolling_average(filtered_df, window_minutes)

        print("Creating charts...")

        # Generate charts and get base64 images
        images = create_charts(
            processed_df,
            max_points,
            title,
            dpi,
        )

        print(f"\n✅ Chart generation complete! Generated {len(images)} chart(s)")
        
        return images

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return None


def save_results_to_files(args, images, input_file):
    timestamped_filename = make_timestamped_filename(input_file, 'sentiments', 'chart', 'json', 'png', logger)
    
    # Save images to files
    ensure_directory(args.output_dir)
    
    if len(images) == 1:
        # Single chart case
        output_path = os.path.join(args.output_dir, timestamped_filename)
        image_data = base64.b64decode(images[0])
        with open(output_path, "wb") as f:
            f.write(image_data)
        print(f"✅ Chart saved to: {output_path}")
    else:
        # Multiple charts case
        stem = Path(timestamped_filename).stem
        suffix = Path(timestamped_filename).suffix
        
        for i, image_b64 in enumerate(images):
            chart_filename = f"{stem}_{i+1}{suffix}"
            output_path = os.path.join(args.output_dir, chart_filename)
            
            # Decode base64 and save
            image_data = base64.b64decode(image_b64)
            with open(output_path, "wb") as f:
                f.write(image_data)
            
        print(f"✅ {len(images)} charts saved to {args.output_dir}")

def main():
    """Main function with CLI interface."""
    args = parse_args()
    interval_minutes = parse_interval_minutes(args.interval_minutes)

    input_file = determine_input_file(args.input_file, args.input_dir)

    # Load and process data
    print("Loading sentiment data...")
    records = load_sentiment_data(input_file)
    
    # Generate charts and get base64 images
    images = generate_sentiment_charts(
        records,
        window_minutes=args.window_minutes,
        interval_minutes=interval_minutes,
        title=args.title,
        dpi=args.dpi,
        max_points=args.max_points,
    )

    save_results_to_files(args, images, input_file)

__all__ = ["generate_sentiment_charts"]

if __name__ == '__main__':
    main()