#!/usr/bin/env python3
"""
Pipeline scheduler and supervisor for automated sentiment analysis.

This module provides scheduled execution of the sentiment analysis pipeline
with configurable intervals, end conditions, and comprehensive error handling.
"""

from datetime import datetime, timedelta
from typing import cast

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

from sentiment_analysis.config_utils import CONFIG
from sentiment_analysis.google_news import process_google_news
from sentiment_analysis.logging_utils import setup_logging

logger = setup_logging(__name__)

MINUTE_INTERVAL=cast(int, CONFIG["interval_min"])
START_DATETIME=cast(datetime, CONFIG["start_datetime"])
END_DATETIME=cast(datetime, CONFIG["end_datetime"])
MAX_CYCLES=cast(int, CONFIG["max_cycles"])
MAX_DURATION_MINUTES=cast(int, CONFIG["max_duration_minutes"])


def compute_first_run(dt: datetime, add_minutes: int = 1, threshold_seconds: int = 50) -> datetime:
    """
    Compute a start time by adding minutes (or more if seconds exceed threshold),
    then set seconds and microseconds to zero.
    """
    if dt.second >= threshold_seconds:
        minutes_to_add = add_minutes + 1   # e.g., add_minutes=1 -> add 2
    else:
        minutes_to_add = add_minutes

    new_dt = dt + timedelta(minutes=minutes_to_add)
    # Set seconds and microseconds to zero
    new_dt = new_dt.replace(second=0, microsecond=0)
    return new_dt


def job_wrapper(
    scheduler: BlockingScheduler,
    cycle_counter: list[int],
    start_time: datetime,
) -> None:
    """
    Execute a single pipeline run and check end conditions.

    This function handles individual pipeline execution with comprehensive
    error handling and logging. After each run, it evaluates all configured
    end conditions and shuts down the scheduler if any condition is met.

    Args:
        scheduler: The blocking scheduler instance to control.
        cycle_counter: Mutable list containing the current cycle count.
        start_time: When the supervisor started running.

    Side effects:
        - Increments the cycle counter
        - Executes the sentiment analysis pipeline
        - Logs execution status and results
        - May shutdown the scheduler based on end conditions
    """
    cycle_counter[0] += 1
    logger.info(f"Starting pipeline run #{cycle_counter[0]}")
    try:
        process_google_news()
        logger.info(f"Pipeline run #{cycle_counter[0]} completed")
    except Exception as e:
        logger.error(f"Pipeline run #{cycle_counter[0]} failed: {e}")

    # Check end conditions after this job
    now = datetime.now()
    if END_DATETIME is not None and now >= END_DATETIME:
        logger.info(f"Reached end datetime {END_DATETIME}, stopping scheduler.")
        scheduler.shutdown(wait=False)
    elif MAX_CYCLES is not None and cycle_counter[0] >= MAX_CYCLES:
        logger.info(f"Reached max cycles {MAX_CYCLES}, stopping scheduler.")
        scheduler.shutdown(wait=False)
    elif MAX_DURATION_MINUTES is not None and (now - start_time).total_seconds() >= MAX_DURATION_MINUTES * 60:
        logger.info(f"Reached max duration {MAX_DURATION_MINUTES} minutes, stopping scheduler.")
        scheduler.shutdown(wait=False)


def supervise_google_news() -> None:
    """
    Run the google news pipeline on a scheduled interval with end conditions.

    Creates a blocking scheduler that executes the pipeline every N minutes
    with configurable stop conditions based on time, cycle count, or duration.
    Provides real-time logging of scheduler status and pipeline execution results.

    Note:
        This function blocks until one of the end conditions is met
        or the process is interrupted via Ctrl+C.
    """
    scheduler = BlockingScheduler()
    cycle_counter = [0]  # Use list to allow modification in job_wrapper

    # Determine start for trigger
    if START_DATETIME is None:
        now = datetime.now()
        trigger_start = compute_first_run(now)
    else:
        trigger_start = START_DATETIME

    trigger = IntervalTrigger(
        minutes=MINUTE_INTERVAL,
        start_date=trigger_start
    )

    # Schedule job
    scheduler.add_job(
        job_wrapper,
        trigger=trigger,
        max_instances=1,
        kwargs={
            'scheduler': scheduler,
            'cycle_counter': cycle_counter,
            'start_time': trigger_start,
        }
    )

    logger.info(f"Supervisor started: from {trigger_start}, every {MINUTE_INTERVAL} minutes.")
    if END_DATETIME:
        logger.info(f"Will stop at end datetime: {END_DATETIME}")
    elif MAX_CYCLES:
        logger.info(f"Will stop after max cycles: {MAX_CYCLES}")
    elif MAX_DURATION_MINUTES:
        logger.info(f"Will stop after max duration (minutes): {MAX_DURATION_MINUTES}")
    else:
        logger.info("Will run until interrupted manually")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped via keyboard/system exit.")


# Define the public API for this module
__all__ = [
    "compute_first_run",
    "job_wrapper",
    "supervise_google_news"
]

if __name__ == "__main__":
    supervise_google_news()