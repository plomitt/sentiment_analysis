import argparse
import datetime
from pathlib import Path
import tomllib
from typing import Dict
from apscheduler.schedulers.blocking import BlockingScheduler

from sentiment_analysis.config_utils import get_config
from sentiment_analysis.pipeline import run_pipeline
from sentiment_analysis.utils import setup_logging

logger = setup_logging(__name__)


def job_wrapper(
    scheduler: BlockingScheduler,
    cycle_counter: list[int],
    start_time: datetime.datetime,
    query: str,
    article_count: int,
    no_content: bool,
    end_datetime: datetime.datetime | None,
    max_cycles: int | None,
    max_duration_minutes: int | None
):
    """
    Execute a single pipeline run and check end conditions.

    This function handles individual pipeline execution with comprehensive
    error handling and logging. After each run, it evaluates all configured
    end conditions and shuts down the scheduler if any condition is met.

    Args:
        scheduler: The blocking scheduler instance to control.
        cycle_counter: Mutable list containing the current cycle count.
        start_time: When the supervisor started running.
        query: Search query string for news articles.
        article_count: Number of articles to process per run.
        no_content: Whether to skip content fetching during pipeline runs.
        end_datetime: Optional datetime when supervisor should stop.
        max_cycles: Optional maximum number of pipeline cycles to run.
        max_duration_minutes: Optional maximum runtime in minutes.

    Side effects:
        - Increments the cycle counter
        - Executes the sentiment analysis pipeline
        - Logs execution status and results
        - May shutdown the scheduler based on end conditions
    """
    cycle_counter[0] += 1
    logger.info(f"Starting pipeline run #{cycle_counter[0]}")
    try:
        run_pipeline(query=query, article_count=article_count, no_content=no_content)
        logger.info(f"Pipeline run #{cycle_counter[0]} completed")
    except Exception as e:
        logger.error(f"Pipeline run #{cycle_counter[0]} failed: {e}")

    # Check end conditions after this job
    now = datetime.datetime.now()
    if end_datetime is not None and now >= end_datetime:
        logger.info(f"Reached end datetime {end_datetime}, stopping scheduler.")
        scheduler.shutdown(wait=False)
    elif max_cycles is not None and cycle_counter[0] >= max_cycles:
        logger.info(f"Reached max cycles {max_cycles}, stopping scheduler.")
        scheduler.shutdown(wait=False)
    elif max_duration_minutes is not None and (now - start_time).total_seconds() >= max_duration_minutes * 60:
        logger.info(f"Reached max duration {max_duration_minutes} minutes, stopping scheduler.")
        scheduler.shutdown(wait=False)


def main(
    minute_interval: int,
    query: str,
    article_count: int,
    no_content: bool,
    end_datetime: datetime.datetime | None,
    max_cycles: int | None,
    max_duration_minutes: int | None
):
    """
    Run the sentiment analysis pipeline on a scheduled interval with end conditions.

    Creates a blocking scheduler that executes the pipeline every N minutes
    with configurable stop conditions based on time, cycle count, or duration.
    Provides real-time logging of scheduler status and pipeline execution results.

    Args:
        minute_interval: Minutes between pipeline runs.
        query: Search query string for news articles.
        article_count: Number of articles to process per run.
        no_content: Whether to skip content fetching during pipeline runs.
        end_datetime: Optional datetime when supervisor should stop.
        max_cycles: Optional maximum number of pipeline cycles to run.
        max_duration_minutes: Optional maximum runtime in minutes.

    Note:
        This function blocks until one of the end conditions is met
        or the process is interrupted via Ctrl+C.
    """
    scheduler = BlockingScheduler()
    start_time = datetime.datetime.now()
    cycle_counter = [0]  # Use list to allow modification in job_wrapper

    # Schedule job every N minutes aligned to the clock
    scheduler.add_job(
        job_wrapper,
        trigger='cron',
        minute=f'*/{minute_interval}',
        kwargs={
            'scheduler': scheduler,
            'cycle_counter': cycle_counter,
            'start_time': start_time,
            'query': query,
            'article_count': article_count,
            'no_content': no_content,
            'end_datetime': end_datetime,
            'max_cycles': max_cycles,
            'max_duration_minutes': max_duration_minutes
        }
    )

    logger.info(f"Supervisor started: every {minute_interval} minutes; Query={query}, Count={article_count}, NoContent={no_content}")
    if end_datetime:
        logger.info(f"Will stop at end datetime: {end_datetime}")
    elif max_cycles:
        logger.info(f"Will stop after max cycles: {max_cycles}")
    elif max_duration_minutes:
        logger.info(f"Will stop after max duration (minutes): {max_duration_minutes}")
    else:
        logger.info("Will run until interrupted manually")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped via keyboard/system exit.")

if __name__ == "__main__":
    config = get_config()

    main(
        minute_interval=config["minutes"],
        query=config["query"],
        article_count=config["article_count"],
        no_content=config["no_content"],
        end_datetime=config["end_datetime"],
        max_cycles=config["max_cycles"],
        max_duration_minutes=config["max_duration_minutes"]
    )