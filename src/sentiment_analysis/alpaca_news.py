import asyncio
import json
import os
from typing import Any

import websockets
from dotenv import load_dotenv

from sentiment_analysis.config_utils import CONFIG
from sentiment_analysis.logging_utils import setup_logging
from sentiment_analysis.pipeline import run_pipeline
from sentiment_analysis.utils import (
    clean_up_body_text,
    convert_alpaca_to_iso,
    convert_alpaca_to_unix,
)

logger = setup_logging(__name__)

ALPACA_NEWS_WS_URL = "wss://stream.data.alpaca.markets/v1beta1/news"
PING_INTERVAL = 20
PING_TIMEOUT = 10


async def authenticate(websocket: Any) -> bool | None:
    """Authenticate with Alpaca websocket API.

    Connects to the Alpaca news websocket and authenticates using API
    credentials from environment variables.

    Args:
        websocket: The websocket connection to authenticate.

    Returns:
        bool | None: True if authentication successful, None if failed.
    """
    load_dotenv()

    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_API_SECRET")

    if not (api_key and secret_key):
        logger.error("ALPACA_API_KEY or ALPACA_API_SECRET not found in environment variables")
        return None

    # Connect
    conn_response = await websocket.recv()
    response_data = json.loads(conn_response)

    if response_data[0]["T"] == "success":
        logger.info("Connected to alpaca news websocket successfully")
    else:
        logger.error(f"Connection to alpaca failed: {response_data}")
        return None

    # Authenticate
    auth_message = {
        "action": "auth",
        "key": api_key,
        "secret": secret_key
    }
    await websocket.send(json.dumps(auth_message))
    auth_response = await websocket.recv()
    response_data = json.loads(auth_response)

    if response_data[0]["T"] == "success":
        logger.info("Authentication to alpaca successful")
        return True
    logger.error(f"Authentication to alpaca failed: {response_data}")
    return None


async def unsubscribe(websocket: Any) -> bool | None:
    """Unsubscribe from Alpaca news websocket.

    Authenticates with the websocket and unsubscribes from all news
    data streams.

    Args:
        websocket: The websocket connection to unsubscribe from.

    Returns:
        bool | None: True if unsubscription successful, None if failed.
    """
    # Authenticate
    authenticated = await authenticate(websocket)
    if not authenticated:
        logger.error("Authentication failed, cannot subscribe")
        return None

    # Unsubscribe
    subscription_message = {
        "action": "unsubscribe",
        "news": ["*"],
    }
    await websocket.send(json.dumps(subscription_message))
    subscription_response = await websocket.recv()
    response_data = json.loads(subscription_response)

    if response_data[0]["T"] == "subscription":
        logger.info("Unsubscription from alpaca successful")
        return True
    logger.error(f"Unsubscription from alpaca failed: {response_data}")
    return None


async def subscribe(websocket: Any) -> bool | None:
    """Subscribe to Alpaca news websocket.

    Authenticates with the websocket and subscribes to BTCUSD news
    data streams.

    Args:
        websocket: The websocket connection to subscribe to.

    Returns:
        bool | None: True if subscription successful, None if failed.
    """
    # Authenticate
    authenticated = await authenticate(websocket)
    if not authenticated:
        logger.error("Authentication failed, cannot subscribe")
        return None

    # Subscribe
    subscription_message = {
        "action": "subscribe",
        "news": CONFIG["news_symbols"],
    }
    await websocket.send(json.dumps(subscription_message))
    subscription_response = await websocket.recv()
    response_data = json.loads(subscription_response)

    if response_data[0]["T"] == "subscription":
        logger.info("Subscription to alpaca successful")
        return True
    logger.error(f"Subscription to alpaca failed: {response_data}")
    return None


async def unsubscribe_from_alpaca_news():
    """Connect to Alpaca websocket and unsubscribe from news streams.

    Establishes a websocket connection to Alpaca and unsubscribes from
    all news data streams. Used for cleanup purposes.
    """
    try:
        logger.info("Starting alpaca news websocket for unsubscription")

        async with websockets.connect(
            ALPACA_NEWS_WS_URL,
            ping_interval=PING_INTERVAL,
            ping_timeout=PING_TIMEOUT
        ) as websocket:
            # Unsubscribe
            unsubscribed = await unsubscribe(websocket)
            if not unsubscribed:
                logger.error("Unsubscription failed, exiting")
                return

    except websockets.exceptions.InvalidURI as e:
        logger.error(f"Invalid URI: {e}")
    except websockets.exceptions.ConnectionClosed as e:
        logger.error(f"Connection failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")


async def process_realtime_alpaca_news():
    """Process real-time news from Alpaca websocket.

    Connects to Alpaca news websocket, subscribes to BTCUSD news,
    and processes incoming news articles through the sentiment analysis pipeline.
    """
    try:
        logger.info("Starting alpaca news websocket")

        async with websockets.connect(
            ALPACA_NEWS_WS_URL,
            ping_interval=PING_INTERVAL,
            ping_timeout=PING_TIMEOUT
        ) as websocket:
            # Subscribe
            subscribed = await subscribe(websocket)
            if not subscribed:
                logger.error("Subscription failed, exiting")
                return

            # Keep connection alive and listen for messages
            try:
                while True:
                    message = await websocket.recv()
                    logger.info("News message received (alpaca)")

                    # Try to parse message
                    try:
                        message = json.loads(message)
                        news = message[0]
                        if news["T"] == "n":
                            # Process news item
                            logger.info(f"News headline: {news['headline']}")
                            article = {
                                "title": news["headline"],
                                "body": clean_up_body_text(news["content"]),
                                "source": f"Alpaca News ({news['author']})",
                                "url": news["url"],
                                "timestamp": convert_alpaca_to_iso(news["created_at"]),
                                "unix_timestamp": convert_alpaca_to_unix(news["created_at"]),
                            }

                            # Run pipeline on single article
                            run_pipeline([article])
                        else:
                            logger.warning(f"Failed to parse news: {news}")

                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse message: {message}")

            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(f"Connection closed: {e}")

    except websockets.exceptions.InvalidURI as e:
        logger.error(f"Invalid URI: {e}")
    except websockets.exceptions.ConnectionClosed as e:
        logger.error(f"Connection failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")


def get_user_choice():
    """Prompt user to choose between processing or unsubscribing from Alpaca news."""
    print("\nAlpaca News Options:")
    print("1. Process real-time Alpaca news")
    print("2. Unsubscribe from Alpaca news")
    print("3. Exit")

    while True:
        choice = input("\nSelect an option (1-3): ").strip()

        if choice == "1":
            return "process"
        if choice == "2":
            return "unsubscribe"
        if choice == "3":
            return "exit"
        print("Invalid choice. Please enter 1, 2, or 3.")


def main():
    """Main entry point for Alpaca news processing.

    Prompts user to choose between processing real-time news,
    unsubscribing from news streams, or exiting.
    """
    choice = get_user_choice()

    if choice == "process":
        print("Starting real-time Alpaca news processing...")
        asyncio.run(process_realtime_alpaca_news())
    elif choice == "unsubscribe":
        print("Unsubscribing from Alpaca news...")
        asyncio.run(unsubscribe_from_alpaca_news())
    elif choice == "exit":
        print("Exiting...")
        exit(0)


__all__ = [
    "process_realtime_alpaca_news",
    "unsubscribe_from_alpaca_news"
]


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nScript interrupted by user")
    except Exception as e:
        logger.error(f"Script error: {e}")
