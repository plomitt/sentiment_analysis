#!/usr/bin/env python3
"""
Consistency Testing Framework for Sentiment Analyzer

This script evaluates the deterministic consistency of the LLM-based sentiment analyzer
by running multiple iterations and using statistical measures to quantify reliability.
"""

import argparse
import json
import logging
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from scipy.stats import variation, pearsonr

# Import the existing sentiment analyzer
from sentiment_analysis.client_manager import build_client
from sentiment_analyzer import analyze_article, load_articles_from_json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
        default="src/sentiment_analysis/sentiments/news_with_sentiment.json",
        help="Input JSON file containing articles to analyze (default: src/sentiment_analysis/sentiments/news_with_sentiment.json)"
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

    results = []

    for i, article in enumerate(articles, 1):
        logger.info(f"Processing article {i}/{len(articles)}: {article['title'][:50]}...")

        article_result = {
            "title": article["title"],
            "body": article["body"],
            "timestamp": article["timestamp"],
            "url": article["url"],
            "unix_timestamp": article.get("unix_timestamp"),
            "scores": [],
            "reasonings": [],
            "run_timestamps": []
        }

        client = build_client()

        for iteration in range(iterations):
            try:
                logger.info(f"  Running iteration {iteration + 1}/{iterations}")

                # Run sentiment analysis
                sentiment = analyze_article(article["title"], article["body"], client)

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
        "sample_size": len(scores_array)
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

        "consistency_distribution": {
            "highly_consistent": sum(1 for cv in all_cvs if cv <= 0.05),
            "moderately_consistent": sum(1 for cv in all_cvs if 0.05 < cv <= 0.10),
            "inconsistent": sum(1 for cv in all_cvs if cv > 0.10)
        }
    }

    # Add percentages
    total_articles = overall_stats["total_articles"]
    if total_articles > 0:
        overall_stats["consistency_distribution"]["highly_consistent_pct"] = (
            overall_stats["consistency_distribution"]["highly_consistent"] / total_articles * 100
        )
        overall_stats["consistency_distribution"]["moderately_consistent_pct"] = (
            overall_stats["consistency_distribution"]["moderately_consistent"] / total_articles * 100
        )
        overall_stats["consistency_distribution"]["inconsistent_pct"] = (
            overall_stats["consistency_distribution"]["inconsistent"] / total_articles * 100
        )

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


def generate_visualizations(results: List[Dict], output_dir: Path):
    """
    Generate visualization charts.

    Args:
        results: List of article results with statistics
        output_dir: Directory to save charts
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
    chart_path = output_dir / "consistency_charts.png"
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

        pie_chart_path = output_dir / "consistency_classification_pie.png"
        plt.savefig(pie_chart_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Pie chart saved to {pie_chart_path}")


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

    # Save JSON results with error handling
    json_path = output_dir / "consistency_results.json"
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
                csv_data.append({
                    "article_index": i + 1,
                    "title": result["title"][:100] + "..." if len(result["title"]) > 100 else result["title"],
                    "url": result["url"],
                    "sample_size": result["statistics"]["sample_size"],
                    "mean_score": result["statistics"]["mean"],
                    "std_dev": result["statistics"]["std_dev"],
                    "cv": result["statistics"]["cv"],
                    "min_score": result["statistics"]["min"],
                    "max_score": result["statistics"]["max"],
                    "range": result["statistics"]["range"],
                    "consistency_rate": result["statistics"]["consistency_rate"],
                    "consistency_classification": classify_consistency(result["statistics"]["cv"])
                })

        if csv_data:
            df = pd.DataFrame(csv_data)
            csv_path = output_dir / "consistency_summary.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8')
            logger.info(f"CSV summary saved to {csv_path}")

    # Generate and save HTML report
    generate_html_report(full_results, output_dir)

    # Generate visualizations
    generate_visualizations(results, output_dir)


def generate_html_report(results: Dict, output_dir: Path):
    """
    Generate an HTML report with results and visualizations.

    Args:
        results: Full results dictionary
        output_dir: Directory to save HTML report
    """
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
        <h2>Visualizations</h2>
        <div class="chart">
            <img src="consistency_charts.png" alt="Consistency Charts">
        </div>
        <div class="chart">
            <img src="consistency_classification_pie.png" alt="Consistency Classification">
        </div>
    </div>

    <div class="section">
        <h2>Article-by-Article Results</h2>
        {format_article_results_html(results['article_results'])}
    </div>
</body>
</html>
    """

    html_path = output_dir / "consistency_report.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    logger.info(f"HTML report saved to {html_path}")


def format_overall_stats_html(stats: Dict) -> str:
    """Format overall statistics for HTML display."""
    if not stats:
        return "<p>No statistics available</p>"

    consistency_dist = stats.get('consistency_distribution', {})

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

        <h3>Consistency Classification</h3>
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
            classification = classify_consistency(stats["cv"])
            class_css = classification.lower().replace(" ", "-")

            rows += f"""
            <tr>
                <td>{i + 1}</td>
                <td><a href="{article['url']}" target="_blank">{article['title'][:80]}{'...' if len(article['title']) > 80 else ''}</a></td>
                <td>{stats['sample_size']}</td>
                <td>{stats['mean']:.2f}</td>
                <td>{stats['std_dev']:.3f}</td>
                <td>{stats['cv']:.3f}</td>
                <td>{stats['range']:.2f}</td>
                <td>{stats['consistency_rate']:.1%}</td>
                <td class="{class_css}">{classification}</td>
            </tr>
            """

    return f"""
    <table>
        <thead>
            <tr>
                <th>#</th>
                <th>Title</th>
                <th>Sample Size</th>
                <th>Mean Score</th>
                <th>Std Dev</th>
                <th>CV</th>
                <th>Range</th>
                <th>Consistency Rate</th>
                <th>Classification</th>
            </tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>
    """


def main():
    """Main function to run the consistency test."""
    args = parse_arguments()

    # Validate input file exists
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    # Load articles
    logger.info(f"Loading articles from {args.input}")
    articles = load_articles_from_json(args.input)

    if not articles:
        logger.error("No articles found in input file")
        sys.exit(1)

    logger.info(f"Loaded {len(articles)} articles")

    # Collect sentiment data
    results = collect_sentiment_data(articles, args.iterations, args.timeout)

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

    # Prepare metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "total_iterations": args.iterations,
        "total_articles": len(articles),
        "model": "ibm/granite-4-h-tiny",
        "temperature": 0.1,
        "input_file": args.input,
        "timeout_between_calls": args.timeout
    }

    # Create output directory
    output_dir = Path(args.output_dir)

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

    consistency_dist = overall_stats.get('consistency_distribution', {})
    logger.info(f"Highly consistent articles: {consistency_dist.get('highly_consistent_pct', 0):.1f}%")
    logger.info(f"Moderately consistent articles: {consistency_dist.get('moderately_consistent_pct', 0):.1f}%")
    logger.info(f"Inconsistent articles: {consistency_dist.get('inconsistent_pct', 0):.1f}%")

    logger.info("=" * 60)
    logger.info("Results saved to:")
    logger.info(f"  • JSON: {output_dir}/consistency_results.json")
    logger.info(f"  • CSV: {output_dir}/consistency_summary.csv")
    logger.info(f"  • HTML: {output_dir}/consistency_report.html")
    logger.info(f"  • Charts: {output_dir}/consistency_charts.png")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()