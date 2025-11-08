from datetime import datetime, timedelta

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

from sentiment_analysis.config_utils import get_config
from sentiment_analysis.pipeline import run_pipeline
from sentiment_analysis.utils import setup_logging

logger = setup_logging(__name__)


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
    query: str,
    article_count: int,
    no_content: bool,
    end_datetime: datetime | None,
    max_cycles: int | None,
    max_duration_minutes: int | None,
    use_smart_search: bool = False,
    use_reasoning: bool = None,
    temperature: float | None = None
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
        use_smart_search: Use smart search if True for body content fetching.

    Side effects:
        - Increments the cycle counter
        - Executes the sentiment analysis pipeline
        - Logs execution status and results
        - May shutdown the scheduler based on end conditions
    """
    cycle_counter[0] += 1
    logger.info(f"Starting pipeline run #{cycle_counter[0]}")
    try:
        run_pipeline(query=query, article_count=article_count, no_content=no_content, use_smart_search=use_smart_search, use_reasoning=use_reasoning, temperature=temperature)
        logger.info(f"Pipeline run #{cycle_counter[0]} completed")
    except Exception as e:
        logger.error(f"Pipeline run #{cycle_counter[0]} failed: {e}")

    # Check end conditions after this job
    now = datetime.now()
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
    start_datetime: datetime | None,
    end_datetime: datetime | None,
    max_cycles: int | None,
    max_duration_minutes: int | None,
    use_smart_search: bool = False,
    use_reasoning: bool = None,
    temperature: float | None = None
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
        use_smart_search: Use smart search if True for body content fetching.

    Note:
        This function blocks until one of the end conditions is met
        or the process is interrupted via Ctrl+C.
    """
    scheduler = BlockingScheduler()
    cycle_counter = [0]  # Use list to allow modification in job_wrapper

    # Determine start for trigger
    if start_datetime is None:
        now = datetime.now()
        trigger_start = compute_first_run(now)
    else:
        trigger_start = start_datetime

    trigger = IntervalTrigger(
        minutes=minute_interval,
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
            'query': query,
            'article_count': article_count,
            'no_content': no_content,
            'end_datetime': end_datetime,
            'max_cycles': max_cycles,
            'max_duration_minutes': max_duration_minutes,
            'use_smart_search': use_smart_search,
            'use_reasoning': use_reasoning,
            'temperature': temperature
        }
    )

    logger.info(f"Supervisor started: from {trigger_start}, every {minute_interval} minutes; Query={query}, Count={article_count}, NoContent={no_content}")
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
        minute_interval=config["interval_min"],
        query=config["query"],
        article_count=config["article_count"],
        no_content=config["no_content"],
        start_datetime=config["start_datetime"],
        end_datetime=config["end_datetime"],
        max_cycles=config["max_cycles"],
        max_duration_minutes=config["max_duration_minutes"],
        use_smart_search=config["use_smart_search"],
        use_reasoning=config["use_reasoning"],
        temperature=config["temperature"]
    )