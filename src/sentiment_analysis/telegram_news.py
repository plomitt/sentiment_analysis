import os
import asyncio
from typing import Any, cast
from telethon import TelegramClient, events
from dotenv import load_dotenv

from sentiment_analysis.config_utils import get_config
from sentiment_analysis.logging_utils import setup_logging
from sentiment_analysis.pipeline import run_pipeline
from sentiment_analysis.utils import convert_alpaca_to_iso, convert_alpaca_to_unix, clean_up_body_text

logger = setup_logging(__name__)

load_dotenv()

api_id = int(os.getenv("TELEGRAM_API_ID"))
api_hash = os.getenv("TELEGRAM_API_HASH")
client = TelegramClient('session_name', api_id, api_hash)

channels_to_monitor = ['https://t.me/markettwits', 'https://t.me/crypto_hd']


async def telegram_listen():
    logger.info("Starting listening to Telegram channels...")

    # Load config
    config = get_config()

    # Resolve entities
    entities = []
    for ch in channels_to_monitor:
        ent = await client.get_entity(ch)
        entities.append(ent)

    @client.on(events.NewMessage(chats=entities))
    async def handler(event):
        m = event.message
        try:
            logger.info("New telegram message received, processing...")

            # Get message info
            channel = await event.get_chat()
            channel_name = getattr(channel, 'title', None)
            user_name = getattr(channel, 'username', None)
            msg_id = m.id
            url = f"https://t.me/{user_name or channel_name}/{msg_id}"
            text = m.text or m.raw_text

            # Process article
            article = {
                "title": '',
                "body": text,
                "source": f"Telegram ({channel_name or user_name})",
                "url": url,
                "timestamp": convert_alpaca_to_iso(m.date),
                "unix_timestamp": convert_alpaca_to_unix(m.date),
            }

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
            
            logger.info(f"New message in {channel_name} (ID: {msg_id}) processed.")
            
        except Exception as e:
            logger.error("Failed to process message:", exc_info=e)

    logger.info("Listening for telegram messages...")
    await client.run_until_disconnected()


def process_realtime_telegram_news():
    with client:
        client.loop.run_until_complete(telegram_listen())


__all__ = [
    "process_realtime_telegram_news"
]


if __name__ == '__main__':
    process_realtime_telegram_news()