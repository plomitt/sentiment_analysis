import asyncio
import json
import os
from typing import Any, cast
import websockets
from dotenv import load_dotenv

from sentiment_analysis.config_utils import get_config
from sentiment_analysis.logging_utils import setup_logging
from sentiment_analysis.pipeline import run_pipeline
from sentiment_analysis.utils import clean_up_body_text, convert_alpaca_to_iso, convert_alpaca_to_unix

logger = setup_logging(__name__)

ALPACA_NEWS_WS_URL = "wss://stream.data.alpaca.markets/v1beta1/news"
PING_INTERVAL = 20
PING_TIMEOUT = 10


async def authenticate(websocket: Any) -> bool | None:
    load_dotenv()

    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_API_SECRET")

    if not (api_key and secret_key):
        logger.error("ALPACA_API_KEY or ALPACA_API_SECRET not found in environment variables")
        return

    # Connect
    conn_response = await websocket.recv()
    response_data = json.loads(conn_response)

    if response_data[0]["T"] == "success":
        logger.info("Connected to alpaca news websocket successfully")
    else:
        logger.error(f"Connection to alpaca failed: {response_data}")
        return

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
    else:
        logger.error(f"Authentication to alpaca failed: {response_data}")
        return


async def unsubscribe(websocket: Any) -> bool | None:
    # Authenticate
    authenticated = await authenticate(websocket)
    if not authenticated:
        logger.error("Authentication failed, cannot subscribe")
        return

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
    else:
        logger.error(f"Unsubscription from alpaca failed: {response_data}")
        return


async def subscribe(websocket: Any) -> bool | None:
    # Authenticate
    authenticated = await authenticate(websocket)
    if not authenticated:
        logger.error("Authentication failed, cannot subscribe")
        return

    # Subscribe
    subscription_message = {
        "action": "subscribe",
        "news": ["BTCUSD"],
        # "news": ["*"],
    }
    await websocket.send(json.dumps(subscription_message))
    subscription_response = await websocket.recv()
    response_data = json.loads(subscription_response)

    if response_data[0]["T"] == "subscription":
        logger.info("Subscription to alpaca successful")
        return True
    else:
        logger.error(f"Subscription to alpaca failed: {response_data}")
        return
    

async def unsubscribe_from_alpaca_news():
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
    try:
        logger.info("Starting alpaca news websocket")

        async with websockets.connect(
            ALPACA_NEWS_WS_URL,
            ping_interval=PING_INTERVAL,
            ping_timeout=PING_TIMEOUT
        ) as websocket:
            # Load config
            config = get_config()

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
                            run_pipeline(
                                str(config["query"]),
                                cast(int, config["article_count"]),
                                bool(config["no_content"]),
                                bool(config["use_similarity_scoring"]),
                                bool(config["use_smart_search"]),
                                bool(config["use_reasoning"]),
                                cast(float, config["temperature"]),
                                news_articles=[article]
                            )
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
        elif choice == "2":
            return "unsubscribe"
        elif choice == "3":
            return "exit"
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


def main():
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