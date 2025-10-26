#!/usr/bin/env python3
"""
Consistency Testing Framework for Sentiment Analyzer

This script evaluates the deterministic consistency of the LLM-based sentiment analyzer
by running multiple iterations and using statistical measures to quantify reliability.
"""

import argparse
import glob
import json
import logging
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import time

from instructor import Mode
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from scipy.stats import variation, pearsonr
try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not available. Some robust statistical methods will be disabled.")

# Import the existing sentiment analyzer
from sentiment_analysis.client_manager import build_client
from sentiment_analysis.sentiment_analyzer import analyze_article, load_articles_from_json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


__all__ = ["run_consistency_test"]

def find_latest_news_file(news_dir: str) -> Optional[str]:
    """
    Find the latest news file from the news directory.

    Since news files are named with sortable prefixes for reverse chronological order,
    the first file alphabetically is the newest.

    Args:
        news_dir: Path to the news directory

    Returns:
        Path to the latest news file, or None if no files found
    """
    try:
        # Look for all news JSON files
        pattern = os.path.join(news_dir, "news_*.json")
        news_files = glob.glob(pattern)

        if not news_files:
            logger.error(f"No news files found in {news_dir}")
            return None

        # Sort alphabetically - with the new naming scheme, this puts newest first
        news_files.sort()
        latest_file = news_files[0]

        logger.info(f"Found latest news file: {latest_file}")
        return latest_file

    except Exception as e:
        logger.error(f"Error finding latest news file: {str(e)}")
        return None


def extract_timestamp_from_filename(filepath: str) -> Optional[str]:
    """
    Extract timestamp from a news filename for use in output filename.

    Expected format: news_[sortable]_[readable].json
    Example: news_99998238678017_2025-10-24_18-06-22.json

    Args:
        filepath: Full path to the input news file

    Returns:
        Timestamp string (sortable_readable) or None if extraction fails
    """
    try:
        filename = os.path.basename(filepath)

        # Expected pattern: news_[sortable]_[readable].json
        # Example: news_99998238678017_2025-10-24_18-06-22.json

        if not (filename.startswith("news_") and filename.endswith(".json")):
            logger.warning(f"Filename doesn't match expected pattern: {filename}")
            return None

        # Remove "news_" prefix and ".json" suffix
        # From: "news_99998238678017_2025-10-24_18-06-22.json"
        # To:   "99998238678017_2025-10-24_18-06-22"
        timestamp_part = filename[5:-5]  # "news_" is 5 chars, ".json" is 5 chars

        # Validate that the timestamp part contains the expected format
        # Should have at least one underscore (separating sortable and readable parts)
        # The readable part should also contain underscores for date formatting
        if "_" not in timestamp_part or timestamp_part.count("_") < 2:
            logger.warning(f"Timestamp part doesn't contain expected format: {timestamp_part}")
            return None

        logger.info(f"Extracted timestamp from filename: {timestamp_part}")
        return timestamp_part

    except Exception as e:
        logger.error(f"Error extracting timestamp from filename {filepath}: {str(e)}")
        return None


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test sentiment analyzer consistency across multiple runs"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of iterations to run for each article (default: 10)"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input JSON file containing articles to analyze (optional - auto-detects from news directory if not provided)"
    )
    parser.add_argument(
        "--news-dir",
        type=str,
        default="src/sentiment_analysis/news",
        help="Directory containing news articles (default: src/sentiment_analysis/news)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="src/sentiment_analysis/consistency",
        help="Output directory for results (default: src/sentiment_analysis/consistency)"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=0.0,
        help="Timeout in seconds between API calls to avoid rate limiting (default: 0.0)"
    )
    return parser.parse_args()


def collect_sentiment_data(articles: List[Dict], iterations: int, timeout: float = 0.0) -> List[Dict]:
    """
    Run sentiment analysis multiple times on each article.

    Args:
        articles: List of article dictionaries
        iterations: Number of times to analyze each article
        timeout: Seconds to wait between API calls

    Returns:
        List of article results with multiple sentiment scores
    """

    logger.info(f"Starting data collection: {len(articles)} articles × {iterations} iterations")

    config = {
        "mode": Mode.JSON
    }
    client = build_client(config=config)
    
    results = []

    for i, article in enumerate(articles, 1):
        logger.info(f"Processing article {i}/{len(articles)}: {article['title'][:50]}...")

        article_result = {
            "title": article["title"],
            "body": article.get("body", ""),  # Handle missing body field gracefully
            "timestamp": article["timestamp"],
            "url": article["url"],
            "unix_timestamp": article.get("unix_timestamp"),
            "scores": [],
            "reasonings": [],
            "run_timestamps": []
        }

        for iteration in range(iterations):
            try:
                logger.info(f"  Running iteration {iteration + 1}/{iterations}")

                # Run sentiment analysis
                sentiment = analyze_article(article["title"], article.get("body", ""), client)

                # Store results
                article_result["scores"].append(sentiment.score)
                article_result["reasonings"].append(sentiment.reasoning)
                article_result["run_timestamps"].append(datetime.now().isoformat())

                # Add timeout between API calls to avoid rate limiting
                if iteration < iterations - 1:  # Don't wait after the last iteration
                    time.sleep(timeout)

            except Exception as e:
                logger.error(f"  Error in iteration {iteration + 1}: {str(e)}")
                # Continue with other iterations even if one fails

        results.append(article_result)
        logger.info(f"  Completed {len(article_result['scores'])} successful iterations")

    return results


def calculate_article_statistics(scores: List[float]) -> Dict[str, Any]:
    """
    Calculate statistical measures for a single article's scores.

    Args:
        scores: List of sentiment scores for an article

    Returns:
        Dictionary containing various statistical measures
    """
    if not scores:
        return {}

    if len(scores) == 1:
        # Only one data point - return basic info only
        return {
            "mean": float(scores[0]),
            "std_dev": 0.0,
            "variance": 0.0,
            "cv": 0.0,
            "min": float(scores[0]),
            "max": float(scores[0]),
            "range": 0.0,
            "median": float(scores[0]),
            "q1": float(scores[0]),
            "q3": float(scores[0]),
            "iqr": 0.0,
            "consistency_rate": 1.0,
            "ci_lower": float(scores[0]),
            "ci_upper": float(scores[0]),
            "margin_error": 0.0,
            "sample_size": 1
        }

    scores_array = np.array(scores)

    # Basic statistics
    mean_score = np.mean(scores_array)
    std_dev = np.std(scores_array, ddof=1)  # Sample standard deviation
    variance = np.var(scores_array, ddof=1)
    min_score = np.min(scores_array)
    max_score = np.max(scores_array)
    median = np.median(scores_array)

    # Relative measures
    cv = variation(scores_array)  # Coefficient of variation
    score_range = max_score - min_score

    # Consistency measures
    consistency_threshold = 0.5
    consistent_scores = np.abs(scores_array - mean_score) <= consistency_threshold
    consistency_rate = np.sum(consistent_scores) / len(scores_array)

    # Confidence interval (95%)
    confidence_level = 0.95
    degrees_freedom = len(scores_array) - 1
    t_critical = stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)
    margin_error = t_critical * (std_dev / np.sqrt(len(scores_array)))
    ci_lower = mean_score - margin_error
    ci_upper = mean_score + margin_error

    # Quartiles
    q1, q3 = np.percentile(scores_array, [25, 75])
    iqr = q3 - q1

    # Calculate enhanced consistency metrics
    frequency_metrics = calculate_mode_frequency(scores)
    robust_stats = calculate_robust_statistics(scores)

    # Calculate frequency-weighted CV
    frequency_weighted_cv = calculate_frequency_weighted_cv(
        scores, float(cv) if not np.isnan(cv) else 0.0
    )

    # Enhanced classification
    enhanced_classification = classify_enhanced_consistency(
        float(cv) if not np.isnan(cv) else 0.0,
        frequency_weighted_cv,
        robust_stats.get("robust_cv", 0.0),
        frequency_metrics.get("mode_frequency", 0.0)
    )

    return {
        "mean": float(mean_score),
        "std_dev": float(std_dev),
        "variance": float(variance),
        "cv": float(cv) if not np.isnan(cv) else 0.0,
        "min": float(min_score),
        "max": float(max_score),
        "range": float(score_range),
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "consistency_rate": float(consistency_rate),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "margin_error": float(margin_error),
        "sample_size": len(scores_array),
        # Enhanced metrics
        "frequency_metrics": frequency_metrics,
        "robust_statistics": robust_stats,
        "frequency_weighted_cv": frequency_weighted_cv,
        "enhanced_classification": enhanced_classification
    }


def calculate_mode_frequency(scores: List[float]) -> Dict[str, Any]:
    """
    Calculate frequency-based metrics for consistency analysis.

    Args:
        scores: List of sentiment scores for an article

    Returns:
        Dictionary containing frequency-based metrics
    """
    if not scores:
        return {}

    scores_array = np.array(scores)
    unique_values, counts = np.unique(scores_array, return_counts=True)
    mode_count = np.max(counts)
    mode_value = unique_values[np.argmax(counts)]
    mode_frequency = mode_count / len(scores_array)

    # Calculate outlier information
    outlier_threshold = 0.1  # Values that appear less than 10% of time
    outlier_mask = counts < (outlier_threshold * len(scores_array))
    outlier_values = unique_values[outlier_mask]
    outlier_count = np.sum(counts[outlier_mask])
    outlier_frequency = outlier_count / len(scores_array)

    return {
        "mode_value": float(mode_value),
        "mode_count": int(mode_count),
        "mode_frequency": float(mode_frequency),
        "outlier_values": [float(v) for v in outlier_values],
        "outlier_count": int(outlier_count),
        "outlier_frequency": float(outlier_frequency),
        "unique_values": len(unique_values)
    }


def calculate_frequency_weighted_cv(scores: List[float], traditional_cv: float,
                                 frequency_weight: float = 0.3) -> float:
    """
    Calculate frequency-weighted coefficient of variation.

    This metric reduces the CV penalty when most scores are consistent.

    Args:
        scores: List of sentiment scores for an article
        traditional_cv: Traditional coefficient of variation
        frequency_weight: Weight for frequency adjustment (0-1)

    Returns:
        Frequency-weighted CV
    """
    if not scores or len(scores) == 1:
        return traditional_cv

    freq_metrics = calculate_mode_frequency(scores)
    mode_freq = freq_metrics["mode_frequency"]

    # Adjustment factor: higher mode frequency = less penalty
    adjustment_factor = 1 - (1 - mode_freq) * frequency_weight

    # Ensure adjustment factor doesn't go below 0.5 (minimum 50% of original CV)
    adjustment_factor = max(adjustment_factor, 0.5)

    return traditional_cv * adjustment_factor


def calculate_robust_statistics(scores: List[float]) -> Dict[str, Any]:
    """
    Calculate robust statistical measures using Huber M-estimators and other robust methods.

    Args:
        scores: List of sentiment scores for an article

    Returns:
        Dictionary containing robust statistical measures
    """
    if not scores:
        return {}

    if len(scores) == 1:
        return {
            "huber_location": float(scores[0]),
            "huber_scale": 0.0,
            "mad": 0.0,
            "robust_cv": 0.0
        }

    scores_array = np.array(scores)
    robust_stats = {}

    # Median Absolute Deviation (MAD)
    mad = np.median(np.abs(scores_array - np.median(scores_array)))
    # Normalize MAD to be comparable to standard deviation for normal distribution
    mad_normalized = mad * 1.4826 if mad != 0 else 0.0
    robust_stats["mad"] = float(mad_normalized)

    # Robust CV using MAD
    median_val = np.median(scores_array)
    robust_cv = mad_normalized / median_val if median_val != 0 else 0.0
    robust_stats["robust_cv"] = float(robust_cv)

    # Huber M-estimator (if statsmodels is available)
    if STATSMODELS_AVAILABLE:
        try:
            huber = sm.robust.scale.Huber()
            huber_loc, huber_scale = huber(scores_array)
            robust_stats["huber_location"] = float(huber_loc)
            robust_stats["huber_scale"] = float(huber_scale)

            # Huber-based CV
            huber_cv = huber_scale / huber_loc if huber_loc != 0 else 0.0
            robust_stats["huber_cv"] = float(huber_cv)
        except Exception as e:
            logger.warning(f"Could not calculate Huber statistics: {e}")
            robust_stats["huber_location"] = float(median_val)
            robust_stats["huber_scale"] = float(mad_normalized)
            robust_stats["huber_cv"] = float(robust_cv)
    else:
        # Fallback to MAD-based robust statistics
        robust_stats["huber_location"] = float(median_val)
        robust_stats["huber_scale"] = float(mad_normalized)
        robust_stats["huber_cv"] = float(robust_cv)

    return robust_stats


def classify_enhanced_consistency(traditional_cv: float, frequency_weighted_cv: float,
                                robust_cv: float, mode_frequency: float) -> Dict[str, str]:
    """
    Enhanced classification system that considers multiple consistency metrics.

    Args:
        traditional_cv: Traditional coefficient of variation
        frequency_weighted_cv: Frequency-weighted CV
        robust_cv: Robust CV (using Huber/MAD)
        mode_frequency: Frequency of the most common value

    Returns:
        Dictionary containing multiple classification results
    """
    # Traditional classification
    if traditional_cv <= 0.05:
        traditional_class = "Highly Consistent"
    elif traditional_cv <= 0.10:
        traditional_class = "Moderately Consistent"
    else:
        traditional_class = "Inconsistent"

    # Frequency-adjusted classification
    if frequency_weighted_cv <= 0.05:
        freq_class = "Highly Consistent"
    elif frequency_weighted_cv <= 0.10:
        freq_class = "Moderately Consistent"
    else:
        freq_class = "Inconsistent"

    # Robust classification
    if robust_cv <= 0.05:
        robust_class = "Highly Consistent"
    elif robust_cv <= 0.10:
        robust_class = "Moderately Consistent"
    else:
        robust_class = "Inconsistent"

    # Overall classification (combines all metrics with emphasis on frequency)
    if mode_frequency >= 0.8:  # 80% or more consistency
        if frequency_weighted_cv <= 0.08:  # Slightly relaxed threshold
            overall_class = "Highly Consistent"
        elif frequency_weighted_cv <= 0.15:
            overall_class = "Moderately Consistent"
        else:
            overall_class = "Inconsistent"
    else:
        # Use majority vote from the three methods
        classes = [traditional_class, freq_class, robust_class]
        if classes.count("Highly Consistent") >= 2:
            overall_class = "Highly Consistent"
        elif classes.count("Inconsistent") >= 2:
            overall_class = "Inconsistent"
        else:
            overall_class = "Moderately Consistent"

    return {
        "traditional": traditional_class,
        "frequency_adjusted": freq_class,
        "robust": robust_class,
        "overall": overall_class
    }


def classify_consistency(cv: float) -> str:
    """
    Classify consistency level based on coefficient of variation.

    Args:
        cv: Coefficient of variation

    Returns:
        Consistency category string
    """
    if cv <= 0.05:
        return "Highly Consistent"
    elif cv <= 0.10:
        return "Moderately Consistent"
    else:
        return "Inconsistent"


def calculate_overall_statistics(all_results: List[Dict]) -> Dict[str, Any]:
    """
    Calculate overall statistics across all articles.

    Args:
        all_results: List of article results with statistics

    Returns:
        Dictionary containing overall statistics
    """
    if not all_results:
        return {}

    # Extract statistics from all articles
    all_std_devs = [r["statistics"]["std_dev"] for r in all_results if "statistics" in r]
    all_cvs = [r["statistics"]["cv"] for r in all_results if "statistics" in r]
    all_consistency_rates = [r["statistics"]["consistency_rate"] for r in all_results if "statistics" in r]
    all_ranges = [r["statistics"]["range"] for r in all_results if "statistics" in r]

    # Extract enhanced metrics
    all_freq_weighted_cvs = [r["statistics"].get("frequency_weighted_cv", r["statistics"]["cv"])
                            for r in all_results if "statistics" in r]
    all_robust_cvs = [r["statistics"].get("robust_statistics", {}).get("robust_cv", r["statistics"]["cv"])
                     for r in all_results if "statistics" in r]
    all_mode_frequencies = [r["statistics"].get("frequency_metrics", {}).get("mode_frequency", 0.0)
                           for r in all_results if "statistics" in r]

    # Extract enhanced classifications
    all_traditional_classifications = [r["statistics"].get("enhanced_classification", {}).get("traditional", "Inconsistent")
                                      for r in all_results if "statistics" in r]
    all_frequency_adjusted_classifications = [r["statistics"].get("enhanced_classification", {}).get("frequency_adjusted", "Inconsistent")
                                            for r in all_results if "statistics" in r]
    all_robust_classifications = [r["statistics"].get("enhanced_classification", {}).get("robust", "Inconsistent")
                                 for r in all_results if "statistics" in r]
    all_overall_classifications = [r["statistics"].get("enhanced_classification", {}).get("overall", "Inconsistent")
                                 for r in all_results if "statistics" in r]

    if not all_std_devs:
        return {}

    overall_stats = {
        "total_articles": len(all_results),
        "avg_std_dev": float(np.mean(all_std_devs)),
        "median_std_dev": float(np.median(all_std_devs)),
        "max_std_dev": float(np.max(all_std_devs)),
        "min_std_dev": float(np.min(all_std_devs)),

        "avg_cv": float(np.mean(all_cvs)),
        "median_cv": float(np.median(all_cvs)),

        "avg_consistency_rate": float(np.mean(all_consistency_rates)),
        "avg_range": float(np.mean(all_ranges)),

        # Enhanced metrics
        "avg_frequency_weighted_cv": float(np.mean(all_freq_weighted_cvs)),
        "avg_robust_cv": float(np.mean(all_robust_cvs)),
        "avg_mode_frequency": float(np.mean(all_mode_frequencies)),

        # Traditional classification distribution
        "consistency_distribution": {
            "highly_consistent": sum(1 for cv in all_cvs if cv <= 0.05),
            "moderately_consistent": sum(1 for cv in all_cvs if 0.05 < cv <= 0.10),
            "inconsistent": sum(1 for cv in all_cvs if cv > 0.10)
        },

        # Enhanced classification distributions
        "traditional_classification_distribution": {
            "highly_consistent": sum(1 for cls in all_traditional_classifications if cls == "Highly Consistent"),
            "moderately_consistent": sum(1 for cls in all_traditional_classifications if cls == "Moderately Consistent"),
            "inconsistent": sum(1 for cls in all_traditional_classifications if cls == "Inconsistent")
        },

        "frequency_adjusted_classification_distribution": {
            "highly_consistent": sum(1 for cls in all_frequency_adjusted_classifications if cls == "Highly Consistent"),
            "moderately_consistent": sum(1 for cls in all_frequency_adjusted_classifications if cls == "Moderately Consistent"),
            "inconsistent": sum(1 for cls in all_frequency_adjusted_classifications if cls == "Inconsistent")
        },

        "robust_classification_distribution": {
            "highly_consistent": sum(1 for cls in all_robust_classifications if cls == "Highly Consistent"),
            "moderately_consistent": sum(1 for cls in all_robust_classifications if cls == "Moderately Consistent"),
            "inconsistent": sum(1 for cls in all_robust_classifications if cls == "Inconsistent")
        },

        "overall_classification_distribution": {
            "highly_consistent": sum(1 for cls in all_overall_classifications if cls == "Highly Consistent"),
            "moderately_consistent": sum(1 for cls in all_overall_classifications if cls == "Moderately Consistent"),
            "inconsistent": sum(1 for cls in all_overall_classifications if cls == "Inconsistent")
        }
    }

    # Add percentages for all classification distributions
    total_articles = overall_stats["total_articles"]
    if total_articles > 0:
        # Traditional distribution percentages
        overall_stats["consistency_distribution"]["highly_consistent_pct"] = (
            overall_stats["consistency_distribution"]["highly_consistent"] / total_articles * 100
        )
        overall_stats["consistency_distribution"]["moderately_consistent_pct"] = (
            overall_stats["consistency_distribution"]["moderately_consistent"] / total_articles * 100
        )
        overall_stats["consistency_distribution"]["inconsistent_pct"] = (
            overall_stats["consistency_distribution"]["inconsistent"] / total_articles * 100
        )

        # Enhanced distribution percentages
        for dist_name in ["traditional_classification_distribution", "frequency_adjusted_classification_distribution",
                         "robust_classification_distribution", "overall_classification_distribution"]:
            dist = overall_stats[dist_name]
            dist["highly_consistent_pct"] = dist["highly_consistent"] / total_articles * 100
            dist["moderately_consistent_pct"] = dist["moderately_consistent"] / total_articles * 100
            dist["inconsistent_pct"] = dist["inconsistent"] / total_articles * 100

    return overall_stats


def perform_statistical_tests(scores_list: List[List[float]]) -> Dict[str, Any]:
    """
    Perform statistical tests to validate consistency.

    Args:
        scores_list: List of score arrays for each article

    Returns:
        Dictionary containing test results
    """
    if not scores_list or len(scores_list) < 2:
        return {}

    test_results = {}

    # Test normality for each article's scores
    normality_results = []
    for i, scores in enumerate(scores_list):
        if len(scores) >= 3:  # Need at least 3 samples for Shapiro-Wilk
            try:
                statistic, p_value = stats.shapiro(scores)
                normality_results.append({
                    "article_index": i,
                    "shapiro_statistic": float(statistic),
                    "shapiro_p_value": float(p_value),
                    "is_normal": int(p_value > 0.05)
                })
            except Exception as e:
                logger.warning(f"Could not perform Shapiro-Wilk test for article {i}: {e}")

    test_results["normality_tests"] = normality_results

    # Overall normality assessment
    if normality_results:
        normal_count = sum(1 for r in normality_results if r["is_normal"] == 1)
        test_results["overall_normality"] = {
            "normal_articles": normal_count,
            "total_tested": len(normality_results),
            "normal_percentage": normal_count / len(normality_results) * 100
        }

    return test_results


def generate_visualizations(results: List[Dict], output_dir: Path, output_timestamp: str):
    """
    Generate visualization charts.

    Args:
        results: List of article results with statistics
        output_dir: Directory to save charts
        output_timestamp: Timestamp string for file naming
    """
    logger.info("Generating visualizations...")

    # Extract data for plotting
    articles_with_stats = [r for r in results if "statistics" in r and r["statistics"]["sample_size"] > 0]

    if not articles_with_stats:
        logger.warning("No valid data for visualization")
        return

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Sentiment Analyzer Consistency Analysis', fontsize=16, fontweight='bold')

    # 1. Distribution of Standard Deviations
    std_devs = [r["statistics"]["std_dev"] for r in articles_with_stats]
    if std_devs and any(not np.isnan(val) for val in std_devs):
        std_devs_clean = [val for val in std_devs if not np.isnan(val)]
        axes[0, 0].hist(std_devs_clean, bins=min(20, len(set(std_devs_clean))),
                       alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Distribution of Score Standard Deviations')
        axes[0, 0].set_xlabel('Standard Deviation')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(np.mean(std_devs_clean), color='red', linestyle='--',
                           label=f'Mean: {np.mean(std_devs_clean):.3f}')
        axes[0, 0].legend()
    else:
        axes[0, 0].text(0.5, 0.5, 'No data available', ha='center', va='center',
                       transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('Distribution of Score Standard Deviations')

    # 2. Coefficient of Variation Distribution
    cvs = [r["statistics"]["cv"] for r in articles_with_stats]
    if cvs and any(not np.isnan(val) for val in cvs):
        cvs_clean = [val for val in cvs if not np.isnan(val)]
        axes[0, 1].hist(cvs_clean, bins=min(20, len(set(cvs_clean))),
                       alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0, 1].set_title('Distribution of Coefficient of Variation')
        axes[0, 1].set_xlabel('Coefficient of Variation')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(0.05, color='green', linestyle='--', label='Highly Consistent (≤0.05)')
        axes[0, 1].axvline(0.10, color='orange', linestyle='--', label='Moderately Consistent (≤0.10)')
        axes[0, 1].legend()
    else:
        axes[0, 1].text(0.5, 0.5, 'No data available', ha='center', va='center',
                       transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Distribution of Coefficient of Variation')

    # 3. Consistency Rate Distribution
    consistency_rates = [r["statistics"]["consistency_rate"] for r in articles_with_stats]
    if consistency_rates and any(not np.isnan(val) for val in consistency_rates):
        consistency_rates_clean = [val for val in consistency_rates if not np.isnan(val)]
        axes[1, 0].hist(consistency_rates_clean, bins=min(20, len(set(consistency_rates_clean))),
                       alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1, 0].set_title('Distribution of Consistency Rates')
        axes[1, 0].set_xlabel('Consistency Rate (scores within ±0.5 of mean)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].axvline(np.mean(consistency_rates_clean), color='red', linestyle='--',
                           label=f'Mean: {np.mean(consistency_rates_clean):.3f}')
        axes[1, 0].legend()
    else:
        axes[1, 0].text(0.5, 0.5, 'No data available', ha='center', va='center',
                       transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Distribution of Consistency Rates')

    # 4. Score Ranges
    ranges = [r["statistics"]["range"] for r in articles_with_stats]
    if ranges and any(not np.isnan(val) for val in ranges):
        ranges_clean = [val for val in ranges if not np.isnan(val)]
        axes[1, 1].hist(ranges_clean, bins=min(20, len(set(ranges_clean))),
                       alpha=0.7, color='gold', edgecolor='black')
        axes[1, 1].set_title('Distribution of Score Ranges')
        axes[1, 1].set_xlabel('Score Range (Max - Min)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].axvline(np.mean(ranges_clean), color='red', linestyle='--',
                           label=f'Mean: {np.mean(ranges_clean):.3f}')
        axes[1, 1].legend()
    else:
        axes[1, 1].text(0.5, 0.5, 'No data available', ha='center', va='center',
                       transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Distribution of Score Ranges')

    plt.tight_layout()

    # Save the plot
    chart_path = output_dir / f"consistency_charts_{output_timestamp}.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Charts saved to {chart_path}")

    # Create consistency classification pie chart
    consistency_counts = {
        "Highly Consistent": sum(1 for r in articles_with_stats if r["statistics"]["cv"] <= 0.05),
        "Moderately Consistent": sum(1 for r in articles_with_stats if 0.05 < r["statistics"]["cv"] <= 0.10),
        "Inconsistent": sum(1 for r in articles_with_stats if r["statistics"]["cv"] > 0.10)
    }

    if sum(consistency_counts.values()) > 0:
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ['green', 'orange', 'red']
        wedges, texts, autotexts = ax.pie(
            consistency_counts.values(),
            labels=consistency_counts.keys(),
            colors=colors,
            autopct='%1.1f%%',
            startangle=90
        )
        ax.set_title('Consistency Classification Distribution', fontsize=14, fontweight='bold')

        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        plt.tight_layout()

        pie_chart_path = output_dir / f"consistency_classification_pie_{output_timestamp}.png"
        plt.savefig(pie_chart_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Pie chart saved to {pie_chart_path}")

    # Generate enhanced visualizations
    generate_enhanced_visualizations(articles_with_stats, output_dir, output_timestamp)


def generate_enhanced_visualizations(articles_with_stats: List[Dict], output_dir: Path, output_timestamp: str):
    """
    Generate enhanced visualization charts with frequency analysis.

    Args:
        articles_with_stats: List of article results with statistics
        output_dir: Directory to save charts
        output_timestamp: Timestamp string for file naming
    """
    logger.info("Generating enhanced visualizations...")

    # Extract enhanced metrics for plotting
    freq_weighted_cvs = [r["statistics"].get("frequency_weighted_cv", r["statistics"]["cv"])
                         for r in articles_with_stats]
    robust_cvs = [r["statistics"].get("robust_statistics", {}).get("robust_cv", r["statistics"]["cv"])
                 for r in articles_with_stats]
    mode_frequencies = [r["statistics"].get("frequency_metrics", {}).get("mode_frequency", 0.0)
                       for r in articles_with_stats]
    unique_values = [r["statistics"].get("frequency_metrics", {}).get("unique_values", 1)
                    for r in articles_with_stats]

    # Create enhanced figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Enhanced Consistency Analysis with Frequency Metrics', fontsize=16, fontweight='bold')

    # 1. Traditional vs Frequency-Weighted CV Comparison
    traditional_cvs = [r["statistics"]["cv"] for r in articles_with_stats]
    if traditional_cvs and freq_weighted_cvs:
        axes[0, 0].scatter(traditional_cvs, freq_weighted_cvs, alpha=0.6, color='blue', s=30)
        # Add diagonal line
        min_cv, max_cv = 0, max(max(traditional_cvs), max(freq_weighted_cvs))
        axes[0, 0].plot([min_cv, max_cv], [min_cv, max_cv], 'r--', alpha=0.8, label='Perfect Correlation')
        axes[0, 0].set_xlabel('Traditional CV')
        axes[0, 0].set_ylabel('Frequency-Weighted CV')
        axes[0, 0].set_title('Traditional vs Frequency-Weighted CV')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

    # 2. Mode Frequency Distribution
    if mode_frequencies:
        axes[0, 1].hist(mode_frequencies, bins=min(20, len(set(mode_frequencies))),
                       alpha=0.7, color='purple', edgecolor='black')
        axes[0, 1].set_xlabel('Mode Frequency')
        axes[0, 1].set_ylabel('Number of Articles')
        axes[0, 1].set_title('Distribution of Mode Frequencies')
        axes[0, 1].axvline(np.mean(mode_frequencies), color='red', linestyle='--',
                           label=f'Mean: {np.mean(mode_frequencies):.2f}')
        axes[0, 1].axvline(0.8, color='green', linestyle='--', alpha=0.7, label='High Consistency (≥80%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    # 3. CV vs Mode Frequency Scatter Plot
    if traditional_cvs and mode_frequencies:
        scatter = axes[1, 0].scatter(mode_frequencies, traditional_cvs,
                                   alpha=0.6, c=traditional_cvs, cmap='RdYlGn_r', s=30)
        axes[1, 0].set_xlabel('Mode Frequency')
        axes[1, 0].set_ylabel('Traditional CV')
        axes[1, 0].set_title('CV vs Mode Frequency')
        # Add colorbar
        plt.colorbar(scatter, ax=axes[1, 0], label='CV Value')
        # Add threshold lines
        axes[1, 0].axhline(0.05, color='green', linestyle='--', alpha=0.7, label='Highly Consistent')
        axes[1, 0].axhline(0.10, color='orange', linestyle='--', alpha=0.7, label='Moderately Consistent')
        axes[1, 0].axvline(0.8, color='blue', linestyle='--', alpha=0.7, label='High Mode Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # 4. Unique Values Distribution
    if unique_values:
        unique_counts = {}
        for val in unique_values:
            unique_counts[val] = unique_counts.get(val, 0) + 1

        if unique_counts:
            axes[1, 1].bar(unique_counts.keys(), unique_counts.values(),
                           alpha=0.7, color='orange', edgecolor='black')
            axes[1, 1].set_xlabel('Number of Unique Values')
            axes[1, 1].set_ylabel('Number of Articles')
            axes[1, 1].set_title('Distribution of Unique Score Values')
            axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the enhanced plot
    enhanced_chart_path = output_dir / f"enhanced_consistency_charts_{output_timestamp}.png"
    plt.savefig(enhanced_chart_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Enhanced charts saved to {enhanced_chart_path}")

    # Create enhanced classification comparison pie charts
    create_enhanced_classification_charts(articles_with_stats, output_dir, output_timestamp)


def create_enhanced_classification_charts(articles_with_stats: List[Dict], output_dir: Path, output_timestamp: str):
    """
    Create comparison pie charts for traditional vs enhanced classifications.

    Args:
        articles_with_stats: List of article results with statistics
        output_dir: Directory to save charts
        output_timestamp: Timestamp string for file naming
    """
    # Traditional classification counts
    traditional_counts = {
        "Highly Consistent": sum(1 for r in articles_with_stats if r["statistics"]["cv"] <= 0.05),
        "Moderately Consistent": sum(1 for r in articles_with_stats if 0.05 < r["statistics"]["cv"] <= 0.10),
        "Inconsistent": sum(1 for r in articles_with_stats if r["statistics"]["cv"] > 0.10)
    }

    # Enhanced classification counts
    enhanced_counts = {
        "Highly Consistent": sum(1 for r in articles_with_stats
                               if r["statistics"].get("enhanced_classification", {}).get("overall") == "Highly Consistent"),
        "Moderately Consistent": sum(1 for r in articles_with_stats
                                   if r["statistics"].get("enhanced_classification", {}).get("overall") == "Moderately Consistent"),
        "Inconsistent": sum(1 for r in articles_with_stats
                           if r["statistics"].get("enhanced_classification", {}).get("overall") == "Inconsistent")
    }

    if sum(traditional_counts.values()) > 0 and sum(enhanced_counts.values()) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        colors = ['green', 'orange', 'red']

        # Traditional classification pie chart
        wedges1, texts1, autotexts1 = ax1.pie(
            traditional_counts.values(),
            labels=traditional_counts.keys(),
            colors=colors,
            autopct='%1.1f%%',
            startangle=90
        )
        ax1.set_title('Traditional Classification', fontsize=14, fontweight='bold')

        # Enhanced classification pie chart
        wedges2, texts2, autotexts2 = ax2.pie(
            enhanced_counts.values(),
            labels=enhanced_counts.keys(),
            colors=colors,
            autopct='%1.1f%%',
            startangle=90
        )
        ax2.set_title('Enhanced Classification (Frequency-Adjusted)', fontsize=14, fontweight='bold')

        # Make percentage text bold for both charts
        for autotext in autotexts1 + autotexts2:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        plt.tight_layout()

        comparison_chart_path = output_dir / f"classification_comparison_{output_timestamp}.png"
        plt.savefig(comparison_chart_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Classification comparison chart saved to {comparison_chart_path}")


def make_json_safe(obj):
    """
    Convert numpy types and other non-JSON-serializable objects to JSON-safe types.
    Recursively processes dictionaries, lists, and other data structures.

    Args:
        obj: Object to make JSON-safe

    Returns:
        JSON-safe version of the object
    """
    import numpy as np

    if obj is None:
        return None
    elif isinstance(obj, bool):
        return int(obj)  # Convert boolean to int
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        if np.isnan(obj):
            return None  # Convert NaN to null
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return make_json_safe(obj.tolist())
    elif isinstance(obj, dict):
        return {key: make_json_safe(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_safe(item) for item in obj]
    else:
        return obj


def save_results(results: List[Dict], overall_stats: Dict, test_results: Dict,
                metadata: Dict, output_dir: Path):
    """
    Save results in multiple formats.

    Args:
        results: List of article results with statistics
        overall_stats: Overall statistics
        test_results: Statistical test results
        metadata: Run metadata
        output_dir: Directory to save results
    """
    logger.info("Saving results...")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare full results for JSON
    full_results = {
        "metadata": metadata,
        "overall_statistics": overall_stats,
        "statistical_tests": test_results,
        "article_results": results
    }

    # Create debug dump file before JSON serialization
    dump_path = output_dir / "debug_results_dump.pkl"
    try:
        with open(dump_path, 'wb') as f:
            pickle.dump(full_results, f)
        logger.info(f"Debug dump saved to {dump_path}")
    except Exception as e:
        logger.warning(f"Could not create debug dump: {e}")

    # Get timestamp from metadata for file naming
    output_timestamp = metadata.get("input_timestamp", "results")

    # Save JSON results with error handling
    json_path = output_dir / f"consistency_{output_timestamp}.json"
    try:
        # Make data JSON-safe
        json_safe_results = make_json_safe(full_results)

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_safe_results, f, indent=2, ensure_ascii=False)
        logger.info(f"JSON results saved to {json_path}")

        # Remove debug dump if JSON save was successful
        try:
            os.remove(dump_path)
            logger.debug("Debug dump removed after successful JSON save")
        except OSError:
            logger.debug("Could not remove debug dump file")

    except (TypeError, ValueError) as e:
        logger.error(f"JSON serialization failed: {e}")
        logger.error(f"Debug dump preserved at: {dump_path}")
        logger.error("Please examine the dump file to identify serialization issues")
        raise

    # Save CSV summary
    if results:
        csv_data = []
        for i, result in enumerate(results):
            if "statistics" in result and result["statistics"]["sample_size"] > 0:
                # Get enhanced metrics if available
                freq_metrics = result["statistics"].get("frequency_metrics", {})
                robust_stats = result["statistics"].get("robust_statistics", {})
                enhanced_class = result["statistics"].get("enhanced_classification", {})

                csv_data.append({
                    "article_index": i + 1,
                    "title": result["title"][:100] + "..." if len(result["title"]) > 100 else result["title"],
                    "url": result["url"],
                    "sample_size": result["statistics"]["sample_size"],
                    "mean_score": result["statistics"]["mean"],
                    "std_dev": result["statistics"]["std_dev"],
                    "cv": result["statistics"]["cv"],
                    "frequency_weighted_cv": result["statistics"].get("frequency_weighted_cv", result["statistics"]["cv"]),
                    "robust_cv": robust_stats.get("robust_cv", result["statistics"]["cv"]),
                    "min_score": result["statistics"]["min"],
                    "max_score": result["statistics"]["max"],
                    "range": result["statistics"]["range"],
                    "consistency_rate": result["statistics"]["consistency_rate"],
                    # Traditional classification
                    "consistency_classification": classify_consistency(result["statistics"]["cv"]),
                    # Enhanced frequency metrics
                    "mode_value": freq_metrics.get("mode_value", result["statistics"]["mean"]),
                    "mode_frequency": freq_metrics.get("mode_frequency", 0.0),
                    "outlier_count": freq_metrics.get("outlier_count", 0),
                    "unique_values": freq_metrics.get("unique_values", 1),
                    # Enhanced classifications
                    "traditional_classification": enhanced_class.get("traditional", classify_consistency(result["statistics"]["cv"])),
                    "frequency_adjusted_classification": enhanced_class.get("frequency_adjusted", classify_consistency(result["statistics"]["cv"])),
                    "robust_classification": enhanced_class.get("robust", classify_consistency(result["statistics"]["cv"])),
                    "overall_classification": enhanced_class.get("overall", classify_consistency(result["statistics"]["cv"]))
                })

        if csv_data:
            df = pd.DataFrame(csv_data)
            csv_path = output_dir / f"consistency_summary_{output_timestamp}.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8')
            logger.info(f"CSV summary saved to {csv_path}")

    # Generate and save HTML report
    generate_html_report(full_results, output_dir)

    # Generate visualizations with timestamp
    generate_visualizations(results, output_dir, output_timestamp)


def generate_html_report(results: Dict, output_dir: Path):
    """
    Generate an HTML report with results and visualizations.

    Args:
        results: Full results dictionary
        output_dir: Directory to save HTML report
    """
    # Get timestamp from metadata for file naming
    output_timestamp = results.get("metadata", {}).get("input_timestamp", "report")
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentiment Analyzer Consistency Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }}
        .header {{
            text-align: center;
            background-color: #f4f4f4;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        .section {{
            margin-bottom: 30px;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 8px;
        }}
        .metric {{
            display: inline-block;
            margin: 10px;
            padding: 15px;
            background-color: #f9f9f9;
            border-radius: 5px;
            text-align: center;
            min-width: 150px;
        }}
        .metric-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #2c3e50;
        }}
        .metric-label {{
            font-size: 0.9em;
            color: #7f8c8d;
        }}
        .highly-consistent {{ color: #27ae60; }}
        .moderately-consistent {{ color: #f39c12; }}
        .inconsistent {{ color: #e74c3c; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        .chart {{
            text-align: center;
            margin: 20px 0;
        }}
        .chart img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 8px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Sentiment Analyzer Consistency Report</h1>
        <p>Generated on {results['metadata']['timestamp']}</p>
        <p>Model: {results['metadata']['model']} | Temperature: {results['metadata']['temperature']} | Iterations: {results['metadata']['total_iterations']}</p>
    </div>

    <div class="section">
        <h2>Overall Statistics</h2>
        {format_overall_stats_html(results['overall_statistics'])}
    </div>

    <div class="section">
        <h2>Traditional Analysis Visualizations</h2>
        <div class="chart">
            <img src="consistency_charts_{output_timestamp}.png" alt="Consistency Charts">
        </div>
        <div class="chart">
            <img src="consistency_classification_pie_{output_timestamp}.png" alt="Consistency Classification">
        </div>
    </div>

    <div class="section">
        <h2>Enhanced Frequency Analysis Visualizations</h2>
        <div class="chart">
            <img src="enhanced_consistency_charts_{output_timestamp}.png" alt="Enhanced Consistency Charts with Frequency Analysis">
        </div>
        <div class="chart">
            <img src="classification_comparison_{output_timestamp}.png" alt="Traditional vs Enhanced Classification Comparison">
        </div>
        <p><strong>Enhanced Analysis:</strong> These charts incorporate frequency-weighted metrics that consider how often specific scores occur, providing a more nuanced view of consistency.</p>
    </div>

    <div class="section">
        <h2>Article-by-Article Results</h2>
        {format_article_results_html(results['article_results'])}
    </div>
</body>
</html>
    """

    html_path = output_dir / f"consistency_report_{output_timestamp}.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    logger.info(f"HTML report saved to {html_path}")


def format_overall_stats_html(stats: Dict) -> str:
    """Format overall statistics for HTML display."""
    if not stats:
        return "<p>No statistics available</p>"

    consistency_dist = stats.get('consistency_distribution', {})
    overall_dist = stats.get('overall_classification_distribution', {})

    html = f"""
        <div class="metric">
            <div class="metric-value">{stats['total_articles']}</div>
            <div class="metric-label">Total Articles</div>
        </div>
        <div class="metric">
            <div class="metric-value">{stats['avg_std_dev']:.3f}</div>
            <div class="metric-label">Average Std Dev</div>
        </div>
        <div class="metric">
            <div class="metric-value">{stats['avg_cv']:.3f}</div>
            <div class="metric-label">Average CV</div>
        </div>
        <div class="metric">
            <div class="metric-value">{stats['avg_consistency_rate']:.1%}</div>
            <div class="metric-label">Avg Consistency Rate</div>
        </div>
        <div class="metric">
            <div class="metric-value">{stats['avg_range']:.3f}</div>
            <div class="metric-label">Avg Score Range</div>
        </div>

        <h3>Enhanced Metrics</h3>
        <div class="metric">
            <div class="metric-value">{stats['avg_frequency_weighted_cv']:.3f}</div>
            <div class="metric-label">Avg Frequency-Weighted CV</div>
        </div>
        <div class="metric">
            <div class="metric-value">{stats['avg_robust_cv']:.3f}</div>
            <div class="metric-label">Avg Robust CV</div>
        </div>
        <div class="metric">
            <div class="metric-value">{stats['avg_mode_frequency']:.1%}</div>
            <div class="metric-label">Avg Mode Frequency</div>
        </div>

        <h3>Traditional Classification</h3>
        <div class="metric">
            <div class="metric-value highly-consistent">{consistency_dist.get('highly_consistent_pct', 0):.1f}%</div>
            <div class="metric-label">Highly Consistent</div>
        </div>
        <div class="metric">
            <div class="metric-value moderately-consistent">{consistency_dist.get('moderately_consistent_pct', 0):.1f}%</div>
            <div class="metric-label">Moderately Consistent</div>
        </div>
        <div class="metric">
            <div class="metric-value inconsistent">{consistency_dist.get('inconsistent_pct', 0):.1f}%</div>
            <div class="metric-label">Inconsistent</div>
        </div>

        <h3>Enhanced Overall Classification</h3>
        <div class="metric">
            <div class="metric-value highly-consistent">{overall_dist.get('highly_consistent_pct', 0):.1f}%</div>
            <div class="metric-label">Highly Consistent</div>
        </div>
        <div class="metric">
            <div class="metric-value moderately-consistent">{overall_dist.get('moderately_consistent_pct', 0):.1f}%</div>
            <div class="metric-label">Moderately Consistent</div>
        </div>
        <div class="metric">
            <div class="metric-value inconsistent">{overall_dist.get('inconsistent_pct', 0):.1f}%</div>
            <div class="metric-label">Inconsistent</div>
        </div>
    """

    return html


def format_article_results_html(articles: List[Dict]) -> str:
    """Format article results for HTML table display."""
    if not articles:
        return "<p>No article results available</p>"

    rows = ""
    for i, article in enumerate(articles):
        if "statistics" in article and article["statistics"]["sample_size"] > 0:
            stats = article["statistics"]
            freq_metrics = stats.get("frequency_metrics", {})
            robust_stats = stats.get("robust_statistics", {})
            enhanced_class = stats.get("enhanced_classification", {})

            # Traditional classification
            traditional_class = classify_consistency(stats["cv"])
            traditional_css = traditional_class.lower().replace(" ", "-")

            # Overall classification
            overall_class = enhanced_class.get("overall", traditional_class)
            overall_css = overall_class.lower().replace(" ", "-")

            rows += f"""
            <tr>
                <td>{i + 1}</td>
                <td><a href="{article['url']}" target="_blank">{article['title'][:50]}{'...' if len(article['title']) > 50 else ''}</a></td>
                <td>{stats['sample_size']}</td>
                <td>{stats['mean']:.2f}</td>
                <td>{stats['min']:.2f}</td>
                <td>{stats['max']:.2f}</td>
                <td>{stats['range']:.2f}</td>
                <td>{stats['cv']:.3f}</td>
                <td>{stats.get('frequency_weighted_cv', stats['cv']):.3f}</td>
                <td>{robust_stats.get('robust_cv', stats['cv']):.3f}</td>
                <td>{freq_metrics.get('mode_frequency', 0):.1%}</td>
                <td>{freq_metrics.get('unique_values', 1)}</td>
                <td class="{traditional_css}">{traditional_class}</td>
                <td class="{overall_css}">{overall_class}</td>
            </tr>
            """

    return f"""
    <table>
        <thead>
            <tr>
                <th>#</th>
                <th>Title</th>
                <th>Sample Size</th>
                <th>Mean</th>
                <th>Min</th>
                <th>Max</th>
                <th>Range</th>
                <th>Unique Values</th>
                <th>Traditional CV</th>
                <th>Frequency-Weighted CV</th>
                <th>Robust CV</th>
                <th>Mode Frequency</th>
                <th>Traditional Classification</th>
                <th>Enhanced Classification</th>
            </tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>
    <p><strong>Note:</strong> Enhanced classification considers frequency of scores, giving more weight to consistent patterns.</p>
    """


def main():
    """Main function to run the consistency test."""
    args = parse_arguments()
    run_consistency_test(
        input_file=args.input,
        news_dir=args.news_dir,
        iterations=args.iterations,
        timeout=args.timeout,
        output_dir=args.output_dir
    )

def run_consistency_test(input_file=None, news_dir="src/sentiment_analysis/news", iterations=10, timeout=0.0, output_dir="src/sentiment_analysis/consistency"):
    """
    Run the consistency test with explicit parameters.
    
    This function contains the core logic for running sentiment consistency tests.
    It loads articles, runs multiple sentiment analysis iterations, calculates statistics,
    and saves results to timestamped output directories.
    
    Args:
        input_file: Manual input file path (optional - auto-detects from news_dir if None)
        news_dir: Directory containing news articles (default: "src/sentiment_analysis/news")
        iterations: Number of sentiment analysis iterations per article (default: 10)
        timeout: Timeout in seconds between API calls to avoid rate limiting (default: 0.0)
        output_dir: Base directory for output files (default: "src/sentiment_analysis/consistency")
    
    Returns:
        dict: Dictionary containing results and metadata, or None if failed
    """
    # Determine input file
    if input_file:
        # Manual file specified
        logger.info(f"📁 Using manually specified input file: {input_file}")
    else:
        # Auto-detect latest news file
        logger.info("🔍 Auto-detecting latest news file...")
        latest_file = find_latest_news_file(news_dir)
        if not latest_file:
            logger.error("❌ Error: No news files found in src/sentiment_analysis/news/")
            logger.error("Please run the RSS fetcher first to generate news files.")
            sys.exit(1)
        input_file = latest_file

    # Validate input file exists
    if not os.path.exists(input_file):
        logger.error(f"❌ Input file not found: {input_file}")
        sys.exit(1)

    # Load articles from news file
    logger.info(f"Loading articles from {input_file}")
    articles = load_articles_from_json(input_file)

    if not articles:
        logger.error("❌ No articles found in input file")
        sys.exit(1)

    logger.info(f"✅ Loaded {len(articles)} articles")

    # Collect sentiment data
    print(f"📁 Input file: {input_file}")
    print()
    results = collect_sentiment_data(articles, iterations, timeout)

    # Calculate statistics for each article
    logger.info("Calculating statistics...")
    scores_list = []
    for result in results:
        if result["scores"]:
            result["statistics"] = calculate_article_statistics(result["scores"])
            scores_list.append(result["scores"])
        else:
            logger.warning(f"No scores collected for article: {result['title'][:50]}...")

    # Calculate overall statistics
    overall_stats = calculate_overall_statistics(results)

    # Perform statistical tests
    test_results = perform_statistical_tests(scores_list)

    # Generate output timestamp and directory using timestamp from input file
    extracted_timestamp = extract_timestamp_from_filename(input_file)

    if extracted_timestamp:
        # Use same timestamp as the input file
        timestamp = extracted_timestamp
        subfolder_name = f"consistency_{timestamp}"
        logger.info(f"📋 Using input timestamp: {timestamp}")
    else:
        # Fallback: generate new timestamp if extraction fails
        logger.warning("⚠️  Could not extract timestamp from input filename, generating new timestamp")
        now = datetime.now()
        # Create sortable prefix: subtract from max timestamp to invert ordering
        sortable_timestamp = f"{99999999999999 - int(now.timestamp())}"
        readable_timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        timestamp = f"{sortable_timestamp}_{readable_timestamp}"
        subfolder_name = f"consistency_{timestamp}"

    # Create output directory with timestamped subfolder
    base_output_dir = Path(output_dir)
    output_dir = base_output_dir / subfolder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Output directory: {output_dir}")
    print()

    # Prepare metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "total_iterations": iterations,
        "total_articles": len(articles),
        "model": "ibm/granite-4-h-tiny",
        "temperature": 0.1,
        "input_file": input_file,
        "input_timestamp": timestamp,
        "timeout_between_calls": timeout
    }

    # Save results
    save_results(results, overall_stats, test_results, metadata, output_dir)

    # Print summary
    logger.info("=" * 60)
    logger.info("CONSISTENCY TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total articles processed: {overall_stats.get('total_articles', 0)}")
    logger.info(f"Average standard deviation: {overall_stats.get('avg_std_dev', 0):.3f}")
    logger.info(f"Average coefficient of variation: {overall_stats.get('avg_cv', 0):.3f}")
    logger.info(f"Average consistency rate: {overall_stats.get('avg_consistency_rate', 0):.1%}")
    logger.info(f"Average score range: {overall_stats.get('avg_range', 0):.3f}")

    consistency_dist = overall_stats.get('consistency_distribution', {})
    overall_dist = overall_stats.get('overall_classification_distribution', {})

    logger.info("Traditional Classification:")
    logger.info(f"  Highly consistent articles: {consistency_dist.get('highly_consistent_pct', 0):.1f}%")
    logger.info(f"  Moderately consistent articles: {consistency_dist.get('moderately_consistent_pct', 0):.1f}%")
    logger.info(f"  Inconsistent articles: {consistency_dist.get('inconsistent_pct', 0):.1f}%")

    logger.info("Enhanced Classification (Frequency-Adjusted):")
    logger.info(f"  Highly consistent articles: {overall_dist.get('highly_consistent_pct', 0):.1f}%")
    logger.info(f"  Moderately consistent articles: {overall_dist.get('moderately_consistent_pct', 0):.1f}%")
    logger.info(f"  Inconsistent articles: {overall_dist.get('inconsistent_pct', 0):.1f}%")

    logger.info(f"Enhanced Metrics:")
    logger.info(f"  Average frequency-weighted CV: {overall_stats.get('avg_frequency_weighted_cv', 0):.3f}")
    logger.info(f"  Average robust CV: {overall_stats.get('avg_robust_cv', 0):.3f}")
    logger.info(f"  Average mode frequency: {overall_stats.get('avg_mode_frequency', 0):.1%}")

    logger.info("=" * 60)
    logger.info(f"📁 Results saved to timestamped subfolder: {output_dir}")
    logger.info("Generated files:")
    logger.info(f"  • JSON: consistency_{timestamp}.json")
    logger.info(f"  • CSV: consistency_summary_{timestamp}.csv")
    logger.info(f"  • HTML: consistency_report_{timestamp}.html")
    logger.info(f"  • Traditional Charts: consistency_charts_{timestamp}.png")
    logger.info(f"  • Enhanced Charts: enhanced_consistency_charts_{timestamp}.png")
    logger.info(f"  • Classification Comparison: classification_comparison_{timestamp}.png")
    logger.info("=" * 60)
    
    return {
        "results": results,
        "overall_stats": overall_stats,
        "test_results": test_results,
        "metadata": metadata,
        "output_dir": output_dir,
        "timestamp": timestamp
    }


if __name__ == "__main__":
    main()