"""
Parallel news processing system for running multiple news sources concurrently.

This module provides a unified interface to run Google News (scheduled),
Telegram (real-time), and Alpaca (real-time) news sources in parallel,
ensuring simultaneous data retrieval and processing while maintaining
database integrity.
"""

import asyncio
import threading
import logging
import signal
from datetime import datetime
from typing import Dict, Any

from sentiment_analysis.google_supervisor import supervise_google_news
from sentiment_analysis.telegram_news import process_realtime_telegram_news
from sentiment_analysis.alpaca_news import process_realtime_alpaca_news

logger = logging.getLogger(__name__)


class ParallelProcessor:
    """
    Manages parallel execution of multiple news processing sources.

    Coordinates three data streams:
    1. Google News RSS feed (scheduled via supervisor)
    2. Telegram channel monitoring (real-time)
    3. Alpaca WebSocket news feed (real-time)
    """

    def __init__(self):
        """Initialize the parallel processor with configuration."""
        self.running = False
        self.tasks = {}
        self.threads = {}
        self.start_time = None

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("ParallelProcessor initialized")

    def _signal_handler(self, signum: int, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        asyncio.create_task(self.stop_all())

    async def start_all(self):
        """
        Start all three news processing sources concurrently.

        Creates threads for blocking operations and async tasks for
        real-time sources. Manages lifecycle and error handling.
        """
        if self.running:
            logger.warning("ParallelProcessor is already running")
            return

        self.running = True
        self.start_time = datetime.now()
        logger.info("Starting all news processing sources...")

        try:
            # Start Google News scheduler in thread (blocking operation)
            await self._start_google_news_thread()

            # Start Telegram monitor (async-compatible)
            await self._start_telegram_monitor()

            # Start Alpaca WebSocket (async)
            await self._start_alpaca_websocket()

            logger.info("All news sources started successfully")
            logger.info(f"Parallel processing started at {self.start_time}")

            # Keep the main event loop running
            await self._maintain_event_loop()

        except Exception as e:
            logger.error(f"Error starting news sources: {e}")
            await self.stop_all()
            raise

    async def stop_all(self):
        """
        Gracefully stop all news processing sources.

        Coordinates clean shutdown across threads and async tasks.
        """
        if not self.running:
            return

        logger.info("Stopping all news processing sources...")
        self.running = False

        # Cancel async tasks
        for task_name, task in self.tasks.items():
            if task and not task.done():
                logger.info(f"Cancelling task: {task_name}")
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    logger.info(f"Task {task_name} cancelled successfully")

        # Wait for threads to finish
        for thread_name, thread in self.threads.items():
            if thread.is_alive():
                logger.info(f"Waiting for thread to finish: {thread_name}")
                thread.join(timeout=5.0)  # Wait up to 5 seconds
                if thread.is_alive():
                    logger.warning(f"Thread {thread_name} did not finish gracefully")

        end_time = datetime.now()
        duration = end_time - self.start_time if self.start_time else "unknown"
        logger.info(f"All news sources stopped. Runtime: {duration}")

    async def _start_google_news_thread(self):
        """Start Google News scheduler in a separate thread."""
        logger.info("Starting Google News scheduler thread...")

        def run_supervisor():
            try:
                supervise_google_news()
            except Exception as e:
                logger.error(f"Google News scheduler error: {e}")

        thread = threading.Thread(target=run_supervisor, name="GoogleNewsScheduler")
        thread.daemon = True  # Allow main thread to exit
        self.threads["google_news"] = thread
        thread.start()
        logger.info("Google News scheduler thread started")

    async def _start_telegram_monitor(self):
        """Start Telegram real-time monitoring."""
        logger.info("Starting Telegram real-time monitor...")

        try:
            task = asyncio.create_task(process_realtime_telegram_news())
            self.tasks["telegram"] = task
            logger.info("Telegram monitor started")
        except Exception as e:
            logger.error(f"Telegram monitor error: {e}")

    async def _start_alpaca_websocket(self):
        """Start Alpaca WebSocket news feed."""
        logger.info("Starting Alpaca WebSocket news feed...")

        async def run_alpaca():
            try:
                await process_realtime_alpaca_news()
            except Exception as e:
                logger.error(f"Alpaca WebSocket error: {e}")

        task = asyncio.create_task(run_alpaca())
        self.tasks["alpaca"] = task
        logger.info("Alpaca WebSocket started")

    async def _maintain_event_loop(self):
        """Keep the event loop running while managing the parallel processes."""
        logger.info("Entering main event loop maintenance...")

        try:
            while self.running:
                # Check thread health
                for thread_name, thread in self.threads.items():
                    if not thread.is_alive():
                        logger.warning(f"Thread {thread_name} has stopped unexpectedly")
                        # Optionally restart the thread here

                # Check task health
                for task_name, task in self.tasks.items():
                    if task.done():
                        try:
                            result = await task
                            logger.warning(f"Task {task_name} completed unexpectedly: {result}")
                        except Exception as e:
                            logger.error(f"Task {task_name} failed: {e}")

                # Sleep before next check
                await asyncio.sleep(5.0)

        except asyncio.CancelledError:
            logger.info("Event loop maintenance cancelled")
        except Exception as e:
            logger.error(f"Error in event loop maintenance: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get current status of all news processing sources."""
        return {
            "running": self.running,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "threads": {
                name: thread.is_alive()
                for name, thread in self.threads.items()
            },
            "tasks": {
                name: not task.done() if task else False
                for name, task in self.tasks.items()
            },
            "uptime": (
                str(datetime.now() - self.start_time)
                if self.start_time else None
            )
        }


async def run_parallel_processor():
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


if __name__ == "__main__":
    asyncio.run(run_parallel_processor())