#!/usr/bin/env python3
"""
Consistency testing script for LLM sentiment analyzer.

This module provides comprehensive testing functionality to validate the consistency
of sentiment analysis scores across multiple runs of the same articles.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from sentiment_analysis.google_news import (
    fetch_article_body_content,
    fetch_news_rss,
)
from sentiment_analysis.pipeline import (
    filter_duplicate_articles,
    get_analyzed_articles,
    get_embedded_articles,
    get_enriched_articles,
)
from sentiment_analysis.utils import find_latest_file, make_timestamped_filename

# Configure logging
logger = logging.getLogger(__name__)


# File I/O Functions for Test News Dataset
def save_test_news_file(
    articles: list[dict[str, Any]], consistency_dir: str = "consistency"
) -> str | None:
    """Save test news articles to timestamped JSON file.

    Args:
        articles: List of article dictionaries with embeddings
        consistency_dir: Directory to save the file in

    Returns:
        Path to saved file or None if failed
    """
    try:
        # Ensure consistency directory exists
        base_path = Path(__file__).parent / consistency_dir
        base_path.mkdir(exist_ok=True)

        # Generate timestamped filename
        filename = make_timestamped_filename(output_name="test_news")
        if not filename:
            logger.error("Failed to generate filename for test news")
            return None

        filepath = base_path / filename

        # Save articles to file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(articles, f, indent=4, ensure_ascii=False)

        logger.info(f"Saved {len(articles)} test articles to {filepath}")
        return str(filepath)

    except Exception as e:
        logger.error(f"Failed to save test news file: {e}")
        return None


def load_test_news_file(filepath: str) -> list[dict[str, Any]]:
    """Load test news articles from JSON file.

    Args:
        filepath: Path to the test news file

    Returns:
        List of article dictionaries or empty list if failed
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            articles = cast(list[dict[str, Any]], json.load(f))

        logger.info(f"Loaded {len(articles)} test articles from {filepath}")
        return articles

    except Exception as e:
        logger.error(f"Failed to load test news file {filepath}: {e}")
        return []


# Test News File Patching Function
def patch_test_news_bodies(
    filepath: str,
    searxng_url: str | None = None,
    skip_existing: bool = True,
    request_delay: float = 1.0,
    stop_on_rate_limit: bool = True,
    use_smart_search: bool = True,
) -> str | None:
    """Patch missing article body content in a test news file.

    Loads articles from a test_news file, fetches missing body content using
    SearXNG, recomputes embeddings for successfully patched articles, and saves
    the updated file back in place. This function is useful when SearXNG rate
    limiting prevents complete content fetching during dataset creation.

    Args:
        filepath: Path to the test news JSON file to patch
        searxng_url: Optional SearXNG instance URL (uses env var or default if None)
        skip_existing: If True, skip articles that already have body content
        request_delay: Delay between SearXNG requests in seconds (to be respectful)
        stop_on_rate_limit: If True, stops processing after first rate limit detection

    Returns:
        Path to the patched file (same as input filepath) or None if failed
    """
    logger.info(f"Starting patch process for test news file: {filepath}")
    logger.info(
        f"Parameters: skip_existing={skip_existing}, request_delay={request_delay}s, "
        f"stop_on_rate_limit={stop_on_rate_limit}"
    )

    try:
        # Validate input file exists
        file_path = Path(filepath)
        if not file_path.exists():
            logger.error(f"Test news file does not exist: {filepath}")
            return None

        # Load existing articles
        logger.info("Loading existing articles from test news file")
        articles = load_test_news_file(filepath)
        if not articles:
            logger.error("Failed to load articles from test news file")
            return None

        logger.info(f"Loaded {len(articles)} articles from test news file")

        # Identify articles that need body content
        articles_to_patch = []
        if skip_existing:
            for i, article in enumerate(articles):
                body = article.get("body", "")
                if not body or not body.strip():
                    articles_to_patch.append((i, article))
            logger.info(
                f"Found {len(articles_to_patch)} articles with empty body content"
            )
        else:
            articles_to_patch = [(i, article) for i, article in enumerate(articles)]
            logger.info(f"Will attempt to patch all {len(articles_to_patch)} articles")

        if not articles_to_patch:
            logger.info(
                "No articles need patching. All articles already have body content."
            )
            return filepath

        # Patch articles with missing body content
        patched_count = 0
        failed_count = 0
        successfully_patched_indices = []  # Track successfully patched article indices

        logger.info(f"Starting to patch {len(articles_to_patch)} articles")

        for i, (article_index, article) in enumerate(articles_to_patch):
            logger.info(
                f"Processing article {i+1}/{len(articles_to_patch)}: {article.get('title', 'Unknown')[:50]}..."
            )

            try:
                # Fetch article body content
                title = article.get("title", "")
                if not title:
                    logger.warning(f"Article {article_index} has no title, skipping")
                    failed_count += 1
                    continue

                logger.debug(f"Fetching body content for article: {title}")
                body_content = fetch_article_body_content(
                    title=title, searxng_url=searxng_url, use_smart_search=use_smart_search
                )

                if body_content and body_content.strip():
                    # Update article with fetched content
                    articles[article_index]["body"] = body_content
                    successfully_patched_indices.append(
                        article_index
                    )  # Track successful patch
                    patched_count += 1
                    logger.info(
                        f"Successfully fetched body content ({len(body_content)} chars) for article {article_index}"
                    )
                else:
                    logger.warning(
                        f"No body content found for article {article_index}: {title[:50]}..."
                    )
                    failed_count += 1

                    # Stop processing if rate limiting is detected
                    if stop_on_rate_limit:
                        logger.warning(
                            "Rate limiting detected. Stopping early to avoid wasting resources."
                        )
                        logger.warning(
                            "SearXNG may be rate-limited. Try restarting the SearXNG container and run this function again."
                        )
                        break

                # Add delay between requests (except for the last article)
                if i < len(articles_to_patch) - 1 and request_delay > 0:
                    logger.debug(f"Waiting {request_delay}s before next request...")
                    time.sleep(request_delay)

            except Exception as e:
                logger.error(f"Error processing article {article_index}: {e}")
                failed_count += 1
                continue

        # Log patching results
        logger.info(
            f"Body content patching completed: {patched_count} successful, {failed_count} failed"
        )

        if patched_count == 0:
            logger.warning("No articles were successfully patched")
            return filepath

        # Recompute embeddings for successfully patched articles
        logger.info(
            f"Recomputing embeddings for {len(successfully_patched_indices)} successfully patched articles"
        )

        try:
            # Extract successfully patched articles
            patched_articles = [articles[idx] for idx in successfully_patched_indices]

            # Generate fresh embeddings for patched articles using batch processing
            embedded_patched_articles = get_embedded_articles(
                patched_articles, batch_size=16
            )

            # Update original articles with new embeddings
            embedding_update_count = 0
            for i, article_index in enumerate(successfully_patched_indices):
                if i < len(embedded_patched_articles):
                    if "embedding" in embedded_patched_articles[i]:
                        articles[article_index]["embedding"] = (
                            embedded_patched_articles[i]["embedding"]
                        )
                        embedding_update_count += 1
                    else:
                        logger.warning(
                            f"No embedding generated for article {article_index}"
                        )
                else:
                    logger.warning(
                        f"Mismatch between patched articles and embeddings for index {article_index}"
                    )

            logger.info(
                f"Successfully updated embeddings for {embedding_update_count} articles"
            )

            if embedding_update_count < len(successfully_patched_indices):
                logger.warning(
                    f"Failed to update embeddings for {len(successfully_patched_indices) - embedding_update_count} articles"
                )

        except Exception as e:
            logger.error(f"Failed to recompute embeddings for patched articles: {e}")
            logger.warning(
                "Articles have updated body content but stale embeddings. Consistency may be affected."
            )
            # Continue with saving the body content updates even if embedding fails

        # Save the updated articles back to the same file
        logger.info(f"Saving patched articles to file: {filepath}")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(articles, f, indent=4, ensure_ascii=False)

            logger.info(f"Successfully saved patched articles to: {filepath}")
            logger.info(
                f"Updated file now contains {len(articles)} articles with body content"
            )

            # Summary of remaining work
            remaining_empty = sum(
                1 for article in articles if not article.get("body", "").strip()
            )
            if remaining_empty > 0:
                logger.info(
                    f"Remaining articles without body content: {remaining_empty}"
                )
                logger.info(
                    "You may need to restart SearXNG container and run this function again"
                )
            else:
                logger.info("All articles now have body content and fresh embeddings!")

            return filepath

        except Exception as e:
            logger.error(f"Failed to save patched articles to file: {e}")
            return None

    except Exception as e:
        logger.error(f"Patch process failed: {e}")
        return None


# Dataset Creation Function
def create_consistency_dataset(
    n_articles: int = 100,
    max_iterations: int = 10,
    delay_seconds: int = 1,
    consistency_dir: str = "consistency",
    use_smart_search: bool = True,
) -> str | None:
    """Create a testing dataset of unique articles with embeddings.

    Fetches news articles in batches until desired count is reached or max iterations
    exhausted. Filters duplicates and generates embeddings for all articles.

    Args:
        n_articles: Target number of unique articles to collect
        max_iterations: Maximum number of fetch attempts
        delay_seconds: Delay between iterations to allow fresh articles
        consistency_dir: Directory to save the dataset
        use_smart_search: Use smart search for body content.

    Returns:
        Path to saved test news file or None if failed
    """
    logger.info(
        f"Starting consistency dataset creation: {n_articles} articles, max {max_iterations} iterations, {delay_seconds}s delay"
    )

    collected_articles: list[dict[str, Any]] = []
    collected_urls = set()  # Track URLs to avoid cross-iteration duplicates

    for iteration in range(max_iterations):
        logger.info(f"Iteration {iteration + 1}/{max_iterations}")

        try:
            # Fetch batch of articles
            batch_size = min(50, n_articles - len(collected_articles))
            fetched_articles = fetch_news_rss(
                query="bitcoin",
                count=batch_size,
                no_content=True,  # Skip content fetching for faster processing,
                use_smart_search=use_smart_search
            )

            if not fetched_articles:
                logger.warning(f"No articles fetched in iteration {iteration + 1}")
                continue

            logger.info(
                f"Fetched {len(fetched_articles)} articles in iteration {iteration + 1}"
            )

            # Filter out duplicates (database check only)
            unique_articles = filter_duplicate_articles(fetched_articles)
            logger.info(
                f"Found {len(unique_articles)} unique articles after DB filtering"
            )

            # Additional filtering: remove articles already collected in previous iterations
            cross_iteration_filtered = [
                article
                for article in unique_articles
                if article.get("url") and article.get("url") not in collected_urls
            ]

            cross_iteration_duplicates = len(unique_articles) - len(
                cross_iteration_filtered
            )
            if cross_iteration_duplicates > 0:
                logger.info(
                    f"Filtered out {cross_iteration_duplicates} cross-iteration duplicates"
                )

            # Add to collection if we have space
            remaining_space = n_articles - len(collected_articles)
            if remaining_space > 0 and cross_iteration_filtered:
                articles_to_add = cross_iteration_filtered[:remaining_space]
                collected_articles.extend(articles_to_add)

                # Update URL tracker with newly added articles
                new_urls = {
                    article.get("url")
                    for article in articles_to_add
                    if article.get("url")
                }
                collected_urls.update(new_urls)

                logger.info(
                    f"Added {len(articles_to_add)} articles to collection "
                    f"(total: {len(collected_articles)}/{n_articles})"
                )
            elif cross_iteration_filtered:
                logger.info("Collection is full, skipping remaining articles")

            # Check if we've reached target
            if len(collected_articles) >= n_articles:
                logger.info("Target number of articles reached")
                break

            # Delay between iterations (except last one)
            if iteration < max_iterations - 1:
                logger.info(f"Waiting {delay_seconds} seconds before next iteration...")
                time.sleep(delay_seconds)

        except Exception as e:
            logger.error(f"Error in iteration {iteration + 1}: {e}")
            continue

    if not collected_articles:
        logger.error("No articles collected after all iterations")
        return None

    logger.info(f"Collected {len(collected_articles)} articles total")

    # Generate embeddings for all articles
    try:
        embedded_articles = get_embedded_articles(collected_articles)
        if not embedded_articles:
            logger.error("Failed to generate embeddings for collected articles")
            return None

        logger.info(f"Generated embeddings for {len(embedded_articles)} articles")

        # Save to file
        return save_test_news_file(embedded_articles, consistency_dir)

    except Exception as e:
        logger.error(f"Failed to process collected articles: {e}")
        return None


# File I/O Functions for Test Scores
def save_test_scores_file(
    articles: list[dict[str, Any]], run_dir: str
) -> str | None:
    """Save test scores to timestamped JSON file in run directory.

    Args:
        articles: List of articles with consistency scores
        run_dir: Directory path for this specific run

    Returns:
        Path to saved file or None if failed
    """
    try:
        # Generate timestamped filename
        filename = make_timestamped_filename(output_name="test_scores")
        if not filename:
            logger.error("Failed to generate filename for test scores")
            return None

        filepath = Path(run_dir) / filename

        # Save articles with scores to file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(articles, f, indent=4, ensure_ascii=False)

        logger.info(f"Saved test scores for {len(articles)} articles to {filepath}")
        return str(filepath)

    except Exception as e:
        logger.error(f"Failed to save test scores file: {e}")
        return None


def load_test_scores_file(filepath: str) -> list[dict[str, Any]]:
    """Load test scores from JSON file.

    Args:
        filepath: Path to the test scores file

    Returns:
        List of articles with consistency scores or empty list if failed
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            articles = cast(list[dict[str, Any]], json.load(f))

        logger.info(f"Loaded test scores for {len(articles)} articles from {filepath}")
        return articles

    except Exception as e:
        logger.error(f"Failed to load test scores file {filepath}: {e}")
        return []


# Pipeline Copy for Testing
def run_pipeline_copy(
    articles: list[dict[str, Any]],
    similarity_mode: bool = False,
) -> list[dict[str, Any]]:
    """Run sentiment analysis pipeline and return results without saving to DB.

    This is a modified copy of run_pipeline that returns analyzed articles
    instead of saving them to the database.

    Args:
        articles: List of articles with embeddings
        similarity_mode: Whether to enrich with similar articles before analysis

    Returns:
        List of analyzed articles with sentiment scores
    """
    try:
        logger.info(
            f"Running pipeline copy in {'similarity' if similarity_mode else 'standard'} mode"
        )
        pipeline_start_time = time.perf_counter()

        if not articles:
            logger.warning("No articles provided to pipeline copy")
            return []

        # Enrich with similar articles if similarity mode is enabled
        if similarity_mode:
            enriched_articles = get_enriched_articles(articles)
            if not enriched_articles:
                logger.warning(
                    "Failed to enrich articles with similar articles, using original"
                )
                enriched_articles = articles
        else:
            enriched_articles = articles

        # Analyze sentiment
        analyzed_articles = get_analyzed_articles(enriched_articles)
        if not analyzed_articles:
            logger.error("Failed to analyze articles in pipeline copy")
            return []

        pipeline_duration = time.perf_counter() - pipeline_start_time
        logger.info(
            f"Pipeline copy completed in {pipeline_duration:.2f}s: "
            f"analyzed {len(analyzed_articles)} articles"
        )

        return analyzed_articles

    except Exception as e:
        logger.error(f"Pipeline copy failed: {e}")
        return []


# Main Consistency Testing Function
def run_consistency_test(
    articles: list[dict[str, Any]],
    n_runs: int = 10,
    similarity_mode: bool = False,
    consistency_dir: str = "consistency",
) -> str | None:
    """Run consistency test on articles across multiple runs.

    For each article, runs sentiment analysis N times and collects all scores
    to measure consistency of the LLM analyzer.

    Args:
        articles: List of articles with embeddings
        n_runs: Number of times to analyze each article
        similarity_mode: Whether to use similarity-based scoring
        consistency_dir: Base directory for consistency tests

    Returns:
        Path to saved test scores file or None if failed
    """
    logger.info(
        f"Starting consistency test: {len(articles)} articles, "
        f"{n_runs} runs per article, similarity_mode={similarity_mode}"
    )

    if not articles:
        logger.error("No articles provided for consistency test")
        return None

    # Create run directory with timestamp
    try:
        base_path = Path(__file__).parent / consistency_dir
        base_path.mkdir(exist_ok=True)

        # Generate timestamped run directory
        now = datetime.now()
        sortable_timestamp = f"{99999999999999 - int(now.timestamp())}"
        readable_timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        run_dir_name = f"run_{sortable_timestamp}_{readable_timestamp}"
        run_dir = base_path / run_dir_name
        run_dir.mkdir(exist_ok=True)

        logger.info(f"Created run directory: {run_dir}")

    except Exception as e:
        logger.error(f"Failed to create run directory: {e}")
        return None

    # Initialize articles with consistency scores array
    articles_with_scores = []
    for article in articles:
        article_copy = dict(article)  # Create copy to avoid modifying original
        article_copy["consistency_scores"] = []
        articles_with_scores.append(article_copy)

    # Run analysis multiple times
    for run_num in range(n_runs):
        logger.info(f"Running consistency test - Run {run_num + 1}/{n_runs}")

        try:
            # Run pipeline copy
            analyzed_articles = run_pipeline_copy(articles, similarity_mode)

            if not analyzed_articles:
                logger.warning(f"No analyzed articles returned in run {run_num + 1}")
                continue

            # Extract scores and add to consistency arrays
            for i, analyzed_article in enumerate(analyzed_articles):
                if i < len(articles_with_scores):
                    score = analyzed_article.get("sentiment_score")
                    if score is not None:
                        articles_with_scores[i]["consistency_scores"].append(score)
                    else:
                        logger.warning(
                            f"No sentiment score found for article {i} in run {run_num + 1}"
                        )
                else:
                    logger.warning(
                        f"More analyzed articles than original articles in run {run_num + 1}"
                    )

            logger.info(
                f"Completed run {run_num + 1}: processed {len(analyzed_articles)} articles"
            )

        except Exception as e:
            logger.error(f"Error in run {run_num + 1}: {e}")
            continue

    # Validate results
    valid_articles = []
    for i, article in enumerate(articles_with_scores):
        scores = article.get("consistency_scores", [])
        if scores:
            logger.debug(f"Article {i}: {len(scores)} scores collected")
            valid_articles.append(article)
        else:
            logger.warning(f"Article {i} has no valid scores")

    if not valid_articles:
        logger.error("No articles have valid consistency scores")
        return None

    logger.info(
        f"Consistency test completed: {len(valid_articles)}/{len(articles)} "
        f"articles have valid scores"
    )

    # Save results
    return save_test_scores_file(valid_articles, str(run_dir))


# Statistical Analysis Functions
def calculate_consistency_metrics(scores: list[float]) -> dict[str, float]:
    """Calculate consistency metrics for a list of scores.

    Args:
        scores: List of sentiment scores for a single article

    Returns:
        Dictionary containing consistency metrics
    """
    if not scores:
        return {
            "count": 0,
            "mean": 0.0,
            "std_dev": 0.0,
            "mode": 0.0,
            "mean_abs_deviation": 0.0,
            "deviation_fraction": 0.0,
            "consistency_score": 0.0,
        }

    scores_array = np.array(scores)
    count = len(scores)
    mean_score = np.mean(scores_array)
    std_dev = np.std(scores_array)

    # Calculate mode (most frequent score)
    mode_result = stats.mode(scores_array, keepdims=True)
    mode_score = float(mode_result.mode[0]) if mode_result.mode.size > 0 else mean_score

    # Calculate mean absolute deviation from mode
    abs_deviations = np.abs(scores_array - mode_score)
    mean_abs_deviation = np.mean(abs_deviations)

    # Calculate deviation fraction (percentage of scores that differ from mode)
    deviations = abs_deviations > 0.001  # Small tolerance for floating point
    deviation_fraction = np.sum(deviations) / count

    # Calculate consistency score (higher = more consistent)
    # Simple formula: 1 / (1 + mean_abs_deviation + deviation_fraction)
    consistency_score = 1.0 / (1.0 + mean_abs_deviation + deviation_fraction)

    return {
        "count": count,
        "mean": float(mean_score),
        "std_dev": float(std_dev),
        "mode": float(mode_score),
        "mean_abs_deviation": float(mean_abs_deviation),
        "deviation_fraction": float(deviation_fraction),
        "consistency_score": float(consistency_score),
    }


def create_consistency_visualizations(
    articles: list[dict[str, Any]], output_dir: str
) -> list[str]:
    """Create visualization plots for consistency analysis.

    Args:
        articles: List of articles with consistency scores
        output_dir: Directory to save plots

    Returns:
        List of paths to generated plot files
    """
    plot_files: list[str] = []

    try:
        # Extract consistency metrics for all articles
        metrics_list = []
        score_distributions = []

        for article in articles:
            scores = article.get("consistency_scores", [])
            if scores:
                metrics = calculate_consistency_metrics(scores)
                metrics_list.append(metrics)
                score_distributions.append(scores)

        if not metrics_list:
            logger.warning("No valid metrics for visualization")
            return plot_files

        # Create plots
        plt.style.use("default")
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Consistency Analysis Results", fontsize=16, fontweight="bold")

        # Plot 1: Consistency Score Distribution
        consistency_scores = [m["consistency_score"] for m in metrics_list]
        axes[0, 0].hist(
            consistency_scores, bins=20, alpha=0.7, color="skyblue", edgecolor="black"
        )
        axes[0, 0].set_title("Consistency Score Distribution")
        axes[0, 0].set_xlabel("Consistency Score")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].axvline(
            np.mean(consistency_scores),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(consistency_scores):.3f}",
        )
        axes[0, 0].legend()

        # Plot 2: Mean Absolute Deviation Distribution
        mad_scores = [m["mean_abs_deviation"] for m in metrics_list]
        axes[0, 1].hist(
            mad_scores, bins=20, alpha=0.7, color="lightcoral", edgecolor="black"
        )
        axes[0, 1].set_title("Mean Absolute Deviation Distribution")
        axes[0, 1].set_xlabel("Mean Absolute Deviation")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].axvline(
            np.mean(mad_scores),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(mad_scores):.3f}",
        )
        axes[0, 1].legend()

        # Plot 3: Deviation Fraction Distribution
        dev_fractions = [m["deviation_fraction"] for m in metrics_list]
        axes[1, 0].hist(
            dev_fractions, bins=20, alpha=0.7, color="lightgreen", edgecolor="black"
        )
        axes[1, 0].set_title("Deviation Fraction Distribution")
        axes[1, 0].set_xlabel("Deviation Fraction")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].axvline(
            np.mean(dev_fractions),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(dev_fractions):.3f}",
        )
        axes[1, 0].legend()

        # Plot 4: Consistency vs Standard Deviation Scatter
        std_devs = [m["std_dev"] for m in metrics_list]
        axes[1, 1].scatter(std_devs, consistency_scores, alpha=0.6, color="purple")
        axes[1, 1].set_title("Consistency Score vs Standard Deviation")
        axes[1, 1].set_xlabel("Standard Deviation")
        axes[1, 1].set_ylabel("Consistency Score")

        # Add correlation coefficient
        if len(std_devs) > 1 and len(consistency_scores) > 1:
            correlation = np.corrcoef(std_devs, consistency_scores)[0, 1]
            axes[1, 1].text(
                0.05,
                0.95,
                f"Correlation: {correlation:.3f}",
                transform=axes[1, 1].transAxes,
                verticalalignment="top",
            )

        plt.tight_layout()

        # Save the plot
        plot_path = Path(output_dir) / "consistency_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        plot_files.append(str(plot_path))

        logger.info(f"Created consistency analysis plot: {plot_path}")

        # Create sample score distribution plots for first few articles
        n_sample_plots = min(6, len(score_distributions))
        if n_sample_plots > 0:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle("Sample Score Distributions", fontsize=16, fontweight="bold")
            axes = axes.flatten()

            for i in range(n_sample_plots):
                scores = score_distributions[i]
                metrics = metrics_list[i]

                axes[i].hist(
                    scores,
                    bins=min(10, len(set(scores))),
                    alpha=0.7,
                    color="steelblue",
                    edgecolor="black",
                )
                axes[i].set_title(
                    f'Article {i+1} (Consistency: {metrics["consistency_score"]:.3f})'
                )
                axes[i].set_xlabel("Sentiment Score")
                axes[i].set_ylabel("Frequency")
                axes[i].axvline(
                    metrics["mode"],
                    color="red",
                    linestyle="--",
                    label=f'Mode: {metrics["mode"]:.1f}',
                )
                axes[i].legend()

            # Hide unused subplots
            for i in range(n_sample_plots, len(axes)):
                axes[i].set_visible(False)

            plt.tight_layout()

            # Save sample plots
            sample_plot_path = Path(output_dir) / "sample_distributions.png"
            plt.savefig(sample_plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            plot_files.append(str(sample_plot_path))

            logger.info(f"Created sample distributions plot: {sample_plot_path}")

    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")

    return plot_files


def analyze_consistency_results(test_scores_file: str) -> str | None:
    """Analyze consistency test results and generate report.

    Args:
        test_scores_file: Path to test scores JSON file
        consistency_dir: Base directory for consistency tests

    Returns:
        Path to analysis report file or None if failed
    """
    logger.info(f"Analyzing consistency results from: {test_scores_file}")

    try:
        # Load test scores
        articles = load_test_scores_file(test_scores_file)
        if not articles:
            logger.error("Failed to load test scores file")
            return None

        # Calculate metrics for all articles
        all_metrics = []
        detailed_results = []

        for i, article in enumerate(articles):
            scores = article.get("consistency_scores", [])
            if not scores:
                logger.warning(f"Article {i} has no consistency scores")
                continue

            metrics = calculate_consistency_metrics(scores)
            all_metrics.append(metrics)

            detailed_results.append(
                {
                    "article_index": i,
                    "title": article.get("title", "Unknown")[:80] + "...",
                    "scores_count": len(scores),
                    "scores": scores,
                    "metrics": metrics,
                }
            )

        if not all_metrics:
            logger.error("No valid metrics calculated")
            return None

        # Calculate overall statistics
        consistency_scores = [m["consistency_score"] for m in all_metrics]
        mad_scores = [m["mean_abs_deviation"] for m in all_metrics]
        dev_fractions = [m["deviation_fraction"] for m in all_metrics]
        std_devs = [m["std_dev"] for m in all_metrics]

        overall_stats = {
            "total_articles": len(all_metrics),
            "avg_consistency_score": np.mean(consistency_scores),
            "median_consistency_score": np.median(consistency_scores),
            "std_consistency_score": np.std(consistency_scores),
            "avg_mean_abs_deviation": np.mean(mad_scores),
            "avg_deviation_fraction": np.mean(dev_fractions),
            "avg_std_deviation": np.mean(std_devs),
            "min_consistency_score": np.min(consistency_scores),
            "max_consistency_score": np.max(consistency_scores),
        }

        # Determine output directory (same as test scores file)
        test_scores_path = Path(test_scores_file)
        output_dir = test_scores_path.parent

        # Create visualizations
        plot_files = create_consistency_visualizations(articles, str(output_dir))

        # Generate analysis report
        report_lines = [
            "Consistency Analysis Report",
            "=" * 40,
            f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Source file: {test_scores_file}",
            "",
            "Overall Statistics:",
            f"- Total articles analyzed: {overall_stats['total_articles']}",
            f"- Average consistency score: {overall_stats['avg_consistency_score']:.3f}",
            f"- Median consistency score: {overall_stats['median_consistency_score']:.3f}",
            f"- Consistency score std dev: {overall_stats['std_consistency_score']:.3f}",
            f"- Average mean absolute deviation: {overall_stats['avg_mean_abs_deviation']:.3f}",
            f"- Average deviation fraction: {overall_stats['avg_deviation_fraction']:.3f}",
            f"- Average standard deviation: {overall_stats['avg_std_deviation']:.3f}",
            f"- Consistency score range: {overall_stats['min_consistency_score']:.3f} - {overall_stats['max_consistency_score']:.3f}",
            "",
            "Interpretation:",
            "- Consistency Score: Higher values (closer to 1.0) indicate more consistent scoring",
            "- Mean Absolute Deviation: Average distance of scores from the most common score",
            "- Deviation Fraction: Percentage of runs that produced different scores from the mode",
            "",
            "Article Details:",
            "-" * 60,
        ]

        # Add detailed results for each article
        for result in detailed_results:
            metrics = result["metrics"]
            report_lines.extend(
                [
                    f"Article {result['article_index']}: {result['title']}",
                    f"  Scores ({result['scores_count']}): {[round(s, 1) for s in result['scores']]}",
                    f"  Consistency Score: {metrics['consistency_score']:.3f}",
                    f"  Mode: {metrics['mode']:.1f}, Mean: {metrics['mean']:.1f}, Std Dev: {metrics['std_dev']:.3f}",
                    f"  Mean Abs Deviation: {metrics['mean_abs_deviation']:.3f}",
                    f"  Deviation Fraction: {metrics['deviation_fraction']:.3f}",
                    "",
                ]
            )

        # Add plot file references
        if plot_files:
            report_lines.extend(
                [
                    "",
                    "Generated Visualizations:",
                    *[f"- {Path(f).name}" for f in plot_files],
                ]
            )

        # Save analysis report
        report_filename = make_timestamped_filename(
            output_name="analysis", output_filetype="txt"
        )
        if not report_filename:
            logger.error("Failed to generate report filename")
            return None

        report_path = Path(output_dir) / report_filename
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

        logger.info(f"Analysis report saved to: {report_path}")
        return str(report_path)

    except Exception as e:
        logger.error(f"Failed to analyze consistency results: {e}")
        return None


# Main Orchestration Function
def run_full_consistency_test(
    n_articles: int = 100,
    n_runs: int = 10,
    max_iterations: int = 10,
    delay_seconds: int = 1,
    similarity_mode: bool = False,
    consistency_dir: str = "consistency",
    force_recreate: bool = False,
    use_smart_search: bool = True,
) -> dict[str, str | None]:
    """Run the complete consistency testing pipeline.

    Orchestrates the entire consistency testing process:
    1. Checks for existing test dataset or creates new one
    2. Loads test articles and runs consistency tests across multiple runs
    3. Analyzes results and generates visualizations

    Args:
        n_articles: Number of articles to collect for testing
        n_runs: Number of times to analyze each article
        max_iterations: Maximum attempts to collect articles
        delay_seconds: Delay between article collection iterations
        similarity_mode: Whether to use similarity-based scoring
        consistency_dir: Directory for all consistency test files
        force_recreate: If True, always creates new test dataset instead of reusing existing ones
        use_smart_search: Use smart search for body content.

    Returns:
        Dictionary with paths to generated files or None values if failed
        {
            "test_news_file": path or None,
            "test_scores_file": path or None,
            "analysis_report": path or None
        }
    """
    logger.info("Starting full consistency test pipeline")
    logger.info(
        f"Parameters: {n_articles} articles, {n_runs} runs, similarity_mode={similarity_mode}, force_recreate={force_recreate}"
    )

    results: dict[str, str | None] = {
        "test_news_file": None,
        "test_scores_file": None,
        "analysis_report": None,
    }

    try:
        # Step 1: Check for existing test dataset or create new one
        test_news_file = None

        if not force_recreate:
            logger.info("Step 1: Looking for existing test dataset")
            base_path = Path(__file__).parent / consistency_dir
            existing_file = find_latest_file(str(base_path), "test_news", "json")

            if existing_file:
                test_news_file = existing_file
                logger.info(f"Found existing test dataset: {test_news_file}")
            else:
                logger.info("No existing test dataset found")

        if test_news_file is None:
            logger.info("Step 1: Creating new consistency test dataset")
            test_news_file = create_consistency_dataset(
                n_articles=n_articles,
                max_iterations=max_iterations,
                delay_seconds=delay_seconds,
                consistency_dir=consistency_dir,
                use_smart_search=use_smart_search
            )

            if not test_news_file:
                logger.error("Failed to create test dataset")
                return results

            logger.info(f"New test dataset created: {test_news_file}")

        results["test_news_file"] = test_news_file

        # Step 2: Load test articles
        logger.info("Step 2: Loading test articles")
        articles = load_test_news_file(test_news_file)
        if not articles:
            logger.error("Failed to load test articles")
            return results

        logger.info(f"Loaded {len(articles)} test articles")

        # Step 3: Run consistency test
        logger.info("Step 3: Running consistency test")
        test_scores_file = run_consistency_test(
            articles=articles,
            n_runs=n_runs,
            similarity_mode=similarity_mode,
            consistency_dir=consistency_dir
        )

        if not test_scores_file:
            logger.error("Failed to run consistency test")
            return results

        results["test_scores_file"] = test_scores_file
        logger.info(f"Consistency test completed: {test_scores_file}")

        # Step 4: Analyze results
        logger.info("Step 4: Analyzing consistency results")
        analysis_report = analyze_consistency_results(test_scores_file=test_scores_file)

        results["analysis_report"] = analysis_report
        if analysis_report:
            logger.info(f"Analysis completed: {analysis_report}")
        else:
            logger.warning("Analysis failed but test was completed")

        # Summary
        logger.info("Full consistency test pipeline completed successfully!")
        logger.info(f"Results summary:")
        logger.info(f"  - Test dataset: {results['test_news_file']}")
        logger.info(f"  - Test scores: {results['test_scores_file']}")
        logger.info(f"  - Analysis report: {results['analysis_report']}")

        return results

    except Exception as e:
        logger.error(f"Full consistency test pipeline failed: {e}")
        return results


# Interactive CLI Helper Functions
def get_user_choice(prompt: str, valid_choices: list[str]) -> str | None:
    """Get user choice from a list of valid options.

    Args:
        prompt: The prompt message to display
        valid_choices: List of valid user choices

    Returns:
        User's validated choice
    """
    while True:
        try:
            choice = input(prompt).strip().lower()
            if choice in valid_choices:
                return choice
            print(f"Invalid choice. Please enter one of: {', '.join(valid_choices)}")
        except (EOFError, KeyboardInterrupt):
            print("\nOperation cancelled by user.")
            return None


def get_numeric_input(prompt: str, default: int | float, min_val: int = 1, datatype: str = "int") -> int | float | None:
    """Get numeric input from user with validation.

    Args:
        prompt: The prompt message to display
        default: Default value if user enters empty string
        min_val: Minimum valid value

    Returns:
        Validated numeric input
    """
    while True:
        try:
            user_input = input(prompt).strip()
            if not user_input:
                return default

            value = int(user_input) if datatype == "int" else float(user_input)
            if value >= min_val:
                return value
            print(f"Value must be at least {min_val}. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
        except (EOFError, KeyboardInterrupt):
            print("\nOperation cancelled by user.")
            return None


def get_boolean_input(prompt: str, default: bool) -> bool | None:
    """Get boolean input from user.

    Args:
        prompt: The prompt message to display
        default: Default value if user enters empty string

    Returns:
        Boolean value from user input
    """
    while True:
        try:
            user_input = input(prompt).strip().lower()
            if not user_input:
                return default

            if user_input in ["y", "yes", "true", "1"]:
                return True
            elif user_input in ["n", "no", "false", "0"]:
                return False
            else:
                print("Please enter 'y'/'n' or 'yes'/'no'.")
        except (EOFError, KeyboardInterrupt):
            print("\nOperation cancelled by user.")
            return None


def get_string_input(prompt: str, default: str) -> str | None:
    """Get string input from user.

    Args:
        prompt: The prompt message to display
        default: Default value if user enters empty string

    Returns:
        String input from user
    """
    try:
        user_input = input(prompt).strip()
        return user_input if user_input else default
    except (EOFError, KeyboardInterrupt):
        print("\nOperation cancelled by user.")
        return None


def display_confirmation(operation: str) -> bool:
    """Display operation confirmation and get user approval.

    Args:
        operation: Description of the operation to confirm

    Returns:
        True if user confirms, False otherwise
    """
    print(f"\nYou chose: {operation}")
    confirmation = get_boolean_input("Continue? (y/n, default: n): ", False)
    return confirmation if confirmation is not None else False


# Interactive CLI Main Function
def interactive_cli() -> None:
    """Interactive command-line interface for consistency testing.

    Provides a user-friendly menu-driven interface for running consistency tests,
    creating datasets, and patching article body content.
    """
    print("=" * 60)
    print("    Consistency Testing Interactive CLI")
    print("=" * 60)
    print()

    while True:
        print("Choose an operation:")
        print("1. Create consistency dataset only")
        print("2. Run patch process (fill missing body content)")
        print("3. Run full consistency test")
        print("q. Quit")
        print()

        choice = get_user_choice("Enter choice (1-3, q): ", ["1", "2", "3", "q"])

        if choice is None:
            return
        elif choice == "q":
            print("Goodbye!")
            return
        elif choice == "1":
            handle_dataset_creation()
        elif choice == "2":
            handle_patch_process()
        elif choice == "3":
            handle_full_consistency_test()

        print("\n" + "-" * 40 + "\n")


def handle_dataset_creation() -> None:
    """Handle dataset creation operation."""
    if not display_confirmation("Create consistency dataset only"):
        return

    print("\nEnter parameters (press Enter for defaults):")

    n_articles = get_numeric_input(
        "Number of articles to collect (default: 100): ", 100
    )
    if n_articles is None:
        return

    max_iterations = get_numeric_input(
        "Max iterations for article collection (default: 10): ", 10
    )
    if max_iterations is None:
        return

    delay_seconds = get_numeric_input("Delay between iterations (default: 60): ", 60)
    if delay_seconds is None:
        return

    consistency_dir = get_string_input(
        "Consistency directory (default: consistency): ", "consistency"
    )
    if consistency_dir is None:
        return
    
    use_smart_search = get_boolean_input("Use smart search? (y/n, default: y): ", True)
    if use_smart_search is None:
        return

    print(f"\nCreating consistency dataset with your parameters...")
    result = create_consistency_dataset(
        n_articles=int(n_articles),
        max_iterations=int(max_iterations),
        delay_seconds=int(delay_seconds),
        consistency_dir=consistency_dir,
        use_smart_search=use_smart_search
    )

    if result:
        print(f"✅ Dataset created successfully: {result}")
    else:
        print("❌ Failed to create dataset")


def handle_patch_process() -> None:
    """Handle patch process operation."""
    if not display_confirmation("Run patch process (fill missing body content)"):
        return

    # First, find the latest test news file
    print("\nLooking for latest test news file...")
    base_path = Path(__file__).parent / "consistency"
    latest_file = find_latest_file(str(base_path), "test_news", "json")

    if not latest_file:
        print("❌ No test news file found. Please create a dataset first.")
        return

    print(f"Found test news file: {latest_file}")

    use_latest = get_boolean_input("Use this file? (y/n, default: y): ", True)
    if use_latest is None:
        return

    if not use_latest:
        filepath = get_string_input("Enter path to test news file: ", "")
        if not filepath:
            print("No file path provided.")
            return
    else:
        filepath = latest_file

    print("\nEnter patch parameters (press Enter for defaults):")

    searxng_url: str | None = get_string_input("SearXNG URL (default: None): ", "")
    searxng_url = searxng_url if searxng_url else None

    skip_existing = get_boolean_input(
        "Skip articles with existing body content? (y/n, default: y): ", True
    )
    if skip_existing is None:
        return

    request_delay = get_numeric_input("Delay between requests (default: 1.0): ", 1)
    if request_delay is None:
        return

    stop_on_rate_limit = get_boolean_input(
        "Stop on rate limit detection? (y/n, default: y): ", True
    )
    if stop_on_rate_limit is None:
        return
    
    use_smart_search = get_boolean_input(
        "Use smart search? (y/n, default: y): ", True
    )
    if use_smart_search is None:
        return

    print(f"\nRunning patch process on: {filepath}")
    result = patch_test_news_bodies(
        filepath=filepath,
        searxng_url=searxng_url,
        skip_existing=skip_existing,
        request_delay=request_delay,
        stop_on_rate_limit=stop_on_rate_limit,
        use_smart_search=use_smart_search
    )

    if result:
        print(f"✅ Patch process completed: {result}")
    else:
        print("❌ Patch process failed")


def handle_full_consistency_test() -> None:
    """Handle full consistency test operation."""
    if not display_confirmation("Run full consistency test"):
        return

    print("\nEnter parameters (press Enter for defaults):")

    n_articles = get_numeric_input(
        "Number of articles to collect (default: 100): ", 100
    )
    if n_articles is None:
        return

    n_runs = get_numeric_input("Number of test runs (default: 10): ", 10)
    if n_runs is None:
        return

    max_iterations = get_numeric_input(
        "Max iterations for article collection (default: 10): ", 10
    )
    if max_iterations is None:
        return

    delay_seconds = get_numeric_input("Delay between iterations (default: 60): ", 60)
    if delay_seconds is None:
        return

    similarity_mode = get_boolean_input(
        "Use similarity-based scoring? (y/n, default: n): ", False
    )
    if similarity_mode is None:
        return

    force_recreate = get_boolean_input(
        "Force recreate dataset? (y/n, default: n): ", False
    )
    if force_recreate is None:
        return

    consistency_dir = get_string_input(
        "Consistency directory (default: consistency): ", "consistency"
    )
    if consistency_dir is None:
        return
    
    use_smart_search = get_boolean_input("Use smart search? (y/n, default: y): ", True)
    if use_smart_search is None:
        return
    
    print(f"\nRunning full consistency test with your parameters...")
    result = run_full_consistency_test(
        n_articles=int(n_articles),
        n_runs=int(n_runs),
        max_iterations=int(max_iterations),
        delay_seconds=int(delay_seconds),
        similarity_mode=similarity_mode,
        force_recreate=force_recreate,
        consistency_dir=consistency_dir,
        use_smart_search=use_smart_search,
    )

    if result and any(result.values()):
        print("✅ Full consistency test completed successfully!")
        if result.get("test_news_file"):
            print(f"  Test dataset: {result['test_news_file']}")
        if result.get("test_scores_file"):
            print(f"  Test scores: {result['test_scores_file']}")
        if result.get("analysis_report"):
            print(f"  Analysis report: {result['analysis_report']}")
    else:
        print("❌ Full consistency test failed")


# Define the public API for this module
__all__ = []

if __name__ == "__main__":
    interactive_cli()
