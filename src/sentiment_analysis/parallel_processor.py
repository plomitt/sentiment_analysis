"""
Parallel news processing system for running multiple news sources concurrently.

This module provides a unified interface to run Google News (scheduled),
Telegram (real-time), and Alpaca (real-time) news sources in parallel,
ensuring simultaneous data retrieval and processing while maintaining
database integrity.
"""

import asyncio
import logging
import signal
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, cast, Callable

from sentiment_analysis.alpaca_news import process_realtime_alpaca_news
from sentiment_analysis.config_utils import CONFIG
from sentiment_analysis.google_supervisor import supervise_google_news
from sentiment_analysis.telegram_news import process_realtime_telegram_news

logger = logging.getLogger(__name__)


class ConfigCache:
    """Cached configuration access for performance."""

    def __init__(self) -> None:
        self.max_restarts = cast(int, CONFIG.get("max_restarts", 3))
        self.initial_backoff = cast(float, CONFIG.get("initial_backoff_seconds", 1.0))
        self.backoff_factor = cast(float, CONFIG.get("backoff_multiplier", 2.0))
        self.enable_auto_restart = cast(bool, CONFIG.get("enable_auto_restart", True))
        self.health_check_interval = cast(float, CONFIG.get("health_check_interval", 5.0))

@dataclass
class TaskMetadata:
    """Metadata for tracking task restart information."""
    name: str
    max_restarts: int
    restart_count: int = 0
    consecutive_failures: int = 0
    is_permanently_failed: bool = False
    last_restart_time: Optional[datetime] = None

    @property
    def should_restart(self) -> bool:
        return self.restart_count < self.max_restarts and not self.is_permanently_failed

    def record_restart(self) -> None:
        self.restart_count += 1
        self.last_restart_time = datetime.now()

    def record_failure(self) -> None:
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.max_restarts:
            self.is_permanently_failed = True

    def reset_failures(self) -> None:
        if self.consecutive_failures > 0:
            self.consecutive_failures = 0

@dataclass
class TaskConfig:
    """Configuration for a specific task."""
    name: str
    task_type: str  # "thread" or "async"
    start_func: Callable[[Optional[TaskMetadata]], Any]
    requires_cleanup: bool = False

@dataclass
class ManagedTask:
    """Unified task management containing both task object and metadata."""
    config: TaskConfig
    metadata: TaskMetadata = field(default_factory=lambda: TaskMetadata(name="", max_restarts=3))
    task_obj: Optional[Any] = None

    def __post_init__(self) -> None:
        if self.metadata.name == "":
            self.metadata.name = self.config.name
            # Update max_restarts from config cache
            self.metadata.max_restarts = CONFIG_CACHE.max_restarts

    @property
    def is_alive(self) -> bool:
        if self.config.task_type == "thread":
            return self.task_obj is not None and self.task_obj.is_alive()
        else:  # async
            return self.task_obj is not None and not self.task_obj.done()

# Global config cache instance
CONFIG_CACHE = ConfigCache()

# Task registry with all available tasks
TASK_REGISTRY: Dict[str, TaskConfig] = {
    "google_news": TaskConfig(
        name="google_news",
        task_type="thread",
        start_func=lambda metadata: None
    ),
    "telegram": TaskConfig(
        name="telegram",
        task_type="async",
        start_func=lambda metadata: None,
        requires_cleanup=True
    ),
    "alpaca": TaskConfig(
        name="alpaca",
        task_type="async",
        start_func=lambda metadata: None,
        requires_cleanup=True
    )
}


class ParallelProcessor:
    """
    Manages parallel execution of multiple news processing sources.

    Coordinates three data streams:
    1. Google News RSS feed (scheduled via supervisor)
    2. Telegram channel monitoring (real-time)
    3. Alpaca WebSocket news feed (real-time)
    """

    def __init__(self) -> None:
        """Initialize the parallel processor with optimized task management."""
        self.running = False
        self.tasks: Dict[str, ManagedTask] = {}
        self.start_time: Optional[datetime] = None

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        logger.info("ParallelProcessor initialized")

    def _signal_handler(self, signum: int, _frame: Any) -> None:
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        asyncio.create_task(self.stop_all())

    async def start_all(self) -> None:
        """Start all news processing sources concurrently."""
        if self.running:
            logger.warning("ParallelProcessor is already running")
            return

        self.running = True
        self.start_time = datetime.now()
        logger.info("Starting all news processing sources...")

        try:
            for task_name in ["google_news", "telegram", "alpaca"]:
                await self._start_task(task_name)

            logger.info("All news sources started successfully")
            logger.info(f"Parallel processing started at {self.start_time}")
            await self._maintain_event_loop()

        except Exception as e:
            logger.error(f"Error starting news sources: {e}")
            await self.stop_all()
            raise

    async def stop_all(self) -> None:
        """Gracefully stop all news processing sources."""
        if not self.running:
            return

        logger.info("Stopping all news processing sources...")
        self.running = False

        for task_name, managed_task in self.tasks.items():
            if managed_task.config.task_type == "async" and managed_task.task_obj and not managed_task.task_obj.done():
                logger.info(f"Cancelling task: {task_name}")
                managed_task.task_obj.cancel()
                try:
                    await managed_task.task_obj
                except asyncio.CancelledError:
                    logger.info(f"Task {task_name} cancelled successfully")
            elif managed_task.config.task_type == "thread" and managed_task.task_obj and managed_task.task_obj.is_alive():
                logger.info(f"Waiting for thread to finish: {task_name}")
                managed_task.task_obj.join(timeout=5.0)
                if managed_task.task_obj.is_alive():
                    logger.warning(f"Thread {task_name} did not finish gracefully")

        duration = datetime.now() - self.start_time if self.start_time else None
        logger.info(f"All news sources stopped. Runtime: {duration}")

    async def _start_task(self, task_name: str, metadata: Optional[TaskMetadata] = None) -> None:
        """Unified task start method handling both initial starts and restarts."""
        config = TASK_REGISTRY[task_name]
        managed_task = self.tasks.get(task_name)

        if managed_task is None:
            # Initial start - create new managed task
            managed_task = ManagedTask(config=config)
            self.tasks[task_name] = managed_task
            logger.info(f"Starting {config.name}...")
            is_restart = False
        else:
            # Restart - update metadata
            managed_task.metadata.record_restart()
            logger.info(f"Restarting {config.name} (restart #{managed_task.metadata.restart_count})")
            is_restart = True

        try:
            if config.task_type == "thread":
                managed_task.task_obj = await self._start_google_news_task()
            elif task_name == "telegram":
                managed_task.task_obj = await self._start_telegram_task()
            elif task_name == "alpaca":
                managed_task.task_obj = await self._start_alpaca_task()

            action = "restarted" if is_restart else "started"
            logger.info(f"{config.name} {action} successfully")

        except Exception as e:
            logger.error(f"Failed to {'start' if not is_restart else 'restart'} {config.name}: {e}")
            if is_restart:
                managed_task.metadata.record_failure()

    @staticmethod
    async def _start_google_news_task() -> threading.Thread:
        """Start Google News scheduler thread."""
        def run_supervisor() -> None:
            try:
                supervise_google_news()
            except Exception as e:
                logger.error(f"Google News scheduler error: {e}")

        thread = threading.Thread(target=run_supervisor, name="GoogleNewsScheduler", daemon=True)
        thread.start()
        return thread

    @staticmethod
    async def _start_telegram_task() -> asyncio.Task:
        """Start Telegram monitor task."""
        return asyncio.create_task(process_realtime_telegram_news())

    @staticmethod
    async def _start_alpaca_task() -> asyncio.Task:
        """Start Alpaca WebSocket task."""
        async def run_alpaca() -> None:
            try:
                await process_realtime_alpaca_news()
            except Exception as e:
                logger.error(f"Alpaca WebSocket error: {e}")

        return asyncio.create_task(run_alpaca())

    def _calculate_backoff_delay(self, metadata: TaskMetadata) -> float:
        """Calculate exponential backoff delay for task restart."""
        return CONFIG_CACHE.initial_backoff * (CONFIG_CACHE.backoff_factor ** metadata.restart_count)

    async def _should_restart_task(self, metadata: TaskMetadata) -> bool:
        """Determine if a task should be restarted based on policies."""
        if not metadata.should_restart:
            return False

        if metadata.last_restart_time:
            time_since_restart = datetime.now() - metadata.last_restart_time
            backoff_delay = self._calculate_backoff_delay(metadata)
            return time_since_restart.total_seconds() >= backoff_delay

        return True

    async def _check_and_restart_tasks(self, enable_restart: bool = True) -> None:
        """Unified function to check and optionally restart all tasks."""
        if all(t.metadata.is_permanently_failed for t in self.tasks.values()):
            logger.warning(f"All tasks are permanently failed, stopping all tasks.")
            await self.stop_all()
            return
    
        for task_name, managed_task in self.tasks.items():
            is_stopped = not managed_task.is_alive
            task_label = managed_task.config.task_type.title()

            if is_stopped:
                managed_task.metadata.record_failure()

                if managed_task.metadata.is_permanently_failed:
                    logger.warning(f"{task_label} {task_name} is permanently failed, skipping restart")
                    continue

                if enable_restart and await self._should_restart_task(managed_task.metadata):
                    # Log async task result if available
                    if (managed_task.config.task_type == "async" and
                        managed_task.task_obj and managed_task.task_obj.done()):
                        try:
                            result = await managed_task.task_obj
                            logger.warning(f"Task {task_name} completed unexpectedly: {result}")
                        except Exception as e:
                            logger.error(f"Task {task_name} failed: {e}")

                    logger.warning(f"{task_label} {task_name} has stopped, attempting restart")
                    await self._restart_task(task_name, managed_task.metadata, managed_task.config.task_type)
                else:
                    logger.info(f"{task_label} {task_name} stopped but {'restart disabled' if not enable_restart else 'not ready for restart'}")
            elif managed_task.metadata.consecutive_failures > 0:
                logger.info(f"{task_label} {task_name} is healthy, resetting consecutive failures")
                managed_task.metadata.reset_failures()

    async def _restart_task(self, task_name: str, metadata: TaskMetadata, task_type: str) -> None:
        """Unified restart dispatcher for both threads and async tasks."""
        managed_task = self.tasks.get(task_name)
        if not managed_task:
            logger.error(f"Task {task_name} not found")
            return

        # Clean up old async task if needed
        if task_type == "async" and managed_task.config.requires_cleanup:
            old_task = managed_task.task_obj
            if old_task and not old_task.done():
                logger.info(f"Cancelling old task: {task_name}")
                old_task.cancel()
                try:
                    await old_task
                except asyncio.CancelledError:
                    logger.info(f"Old task {task_name} cancelled successfully")

        # Dispatch to unified start method
        await self._start_task(task_name, metadata)

    async def _maintain_event_loop(self) -> None:
        """Keep the event loop running while managing the parallel processes."""
        logger.info("Entering main event loop maintenance...")

        try:
            while self.running:
                await self._check_and_restart_tasks(CONFIG_CACHE.enable_auto_restart)
                await asyncio.sleep(CONFIG_CACHE.health_check_interval)

        except asyncio.CancelledError:
            logger.info("Event loop maintenance cancelled")
        except Exception as e:
            logger.error(f"Error in event loop maintenance: {e}")

    
    
        except Exception as e:
            logger.error(f"Error in event loop maintenance: {e}")

    def get_status(self) -> dict[str, Any]:
        """Get current status of all news processing sources with restart information."""
        status: dict[str, Any] = {
            "running": self.running,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "uptime": (
                str(datetime.now() - self.start_time)
                if self.start_time else None
            ),
            "tasks": {}
        }

        # Add unified task status
        for name, managed_task in self.tasks.items():
            status["tasks"][name] = {
                "alive": managed_task.is_alive,
                "type": managed_task.config.task_type,
                "restart_count": managed_task.metadata.restart_count,
                "consecutive_failures": managed_task.metadata.consecutive_failures,
                "last_restart": managed_task.metadata.last_restart_time.isoformat() if managed_task.metadata.last_restart_time else None,
                "permanently_failed": managed_task.metadata.is_permanently_failed
            }

        return status


async def run_parallel_processor() -> None:
    """
    Convenience function to run the parallel processor.
    """
    processor = ParallelProcessor()

    try:
        await processor.start_all()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        await processor.stop_all()


__all__ = [
    "ParallelProcessor",
    "run_parallel_processor"
]


if __name__ == "__main__":
    asyncio.run(run_parallel_processor())
