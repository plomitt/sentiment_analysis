import asyncio
import os

from dotenv import load_dotenv
from telethon import TelegramClient, events

from sentiment_analysis.config_utils import CONFIG
from sentiment_analysis.logging_utils import setup_logging
from sentiment_analysis.pipeline import run_pipeline
from sentiment_analysis.utils import convert_alpaca_to_iso, convert_alpaca_to_unix

logger = setup_logging(__name__)

load_dotenv()

api_id = int(os.getenv("TELEGRAM_API_ID"))
api_hash = os.getenv("TELEGRAM_API_HASH")
client = TelegramClient("session_name", api_id, api_hash)

channels_to_monitor = CONFIG["channels_to_monitor"]


async def telegram_listen():
    """Listen to Telegram channels for new messages.

    Sets up event handlers for specified Telegram channels and processes
    new messages through the sentiment analysis pipeline.
    """
    logger.info("Starting listening to Telegram channels...")

    # Validate channels configuration
    if not channels_to_monitor:
        logger.warning("No Telegram channels configured to monitor. Please set channels_to_monitor in your config.toml file.")
        logger.info("Example configuration in config.toml:")
        logger.info("[telegram]")
        logger.info('channels_to_monitor = ["https://t.me/markettwits", "https://t.me/crypto_hd"]')
        return

    # Resolve entities
    entities = []
    for ch in channels_to_monitor:
        ent = await client.get_entity(ch)
        entities.append(ent)

    @client.on(events.NewMessage(chats=entities))
    async def handler(event):
        """Handle new Telegram message events."""
        m = event.message
        try:
            logger.info("New telegram message received, processing...")

            # Get message info
            channel = await event.get_chat()
            channel_name = getattr(channel, "title", None)
            user_name = getattr(channel, "username", None)
            msg_id = m.id
            body = m.text or m.raw_text
            source = f"Telegram ({channel_name or user_name})"
            url = f"https://t.me/{user_name or channel_name}/{msg_id}"

            # Define article
            article = {
                "title": "",
                "body": body,
                "source": source,
                "url": url,
                "timestamp": convert_alpaca_to_iso(m.date),
                "unix_timestamp": convert_alpaca_to_unix(m.date),
            }

            # Process article
            run_pipeline([article])

            logger.info(f"New message in {channel_name} (ID: {msg_id}) processed.")

        except Exception as e:
            logger.error("Failed to process message:", exc_info=e)

    logger.info("Listening for telegram messages...")
    await client.run_until_disconnected()


async def process_realtime_telegram_news():
    """
    Async version of Telegram monitor that works with asyncio event loops.
    This function is compatible with the parallel processor's async execution.
    """
    try:
        logger.info("Starting async Telegram monitor...")
        await client.start()
        try:
            await telegram_listen()
        finally:
            await client.disconnect()
    except Exception as e:
        logger.error(f"Async Telegram monitor error: {e}", exc_info=True)
        try:
            await client.disconnect()
        except:
            pass


__all__ = [
    "process_realtime_telegram_news"
]


if __name__ == "__main__":
    asyncio.run(process_realtime_telegram_news())
