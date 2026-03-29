"""
Telegram Bot: KisanSathi — answers farmer queries via text and voice on Telegram.

Wires python-telegram-bot to the KisanSathi RAG pipeline.

Usage:
    python -m src.bot       # Start the bot
    # Then message @YourBotName on Telegram

Requires:
    TELEGRAM_BOT_TOKEN in .env file (get from @BotFather)
"""

import logging
import os
import tempfile

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from src.pipeline import KisanSathiPipeline

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Initialize pipeline once at module level
pipeline = KisanSathiPipeline()


def format_response(result: dict) -> str:
    """Format pipeline result into a user-friendly Telegram message."""
    parts = [result["answer"]]

    # Add grounding warning if applicable
    if not result["grounding"]["is_grounded"]:
        parts.append(
            "\n⚠️ This answer may not be fully supported by our source documents."
        )

    # Add top source
    if result["sources"]:
        top_source = result["sources"][0]["scheme"]
        parts.append(f"\n📄 Source: {top_source}")

    return "\n".join(parts)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command."""
    welcome = (
        "🌾 *KisanSathi* — किसान साथी\n\n"
        "नमस्ते! मैं KisanSathi हूँ। मैं आपको सरकारी योजनाओं के बारे में जानकारी दे सकता हूँ।\n\n"
        "Hello! I am KisanSathi. I can help you with information about government schemes for farmers.\n\n"
        "*How to use:*\n"
        "• Send a text message in Hindi or English\n"
        "• Send a voice note in Hindi\n\n"
        "*Examples:*\n"
        "• PM-KISAN योजना के लिए कौन पात्र है?\n"
        "• What is crop insurance?\n"
        "• किसान क्रेडिट कार्ड कैसे बनवाएं?\n\n"
        "Type /help for more info."
    )
    await update.message.reply_text(welcome, parse_mode="Markdown")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command."""
    help_text = (
        "🌾 *KisanSathi Help*\n\n"
        "*Commands:*\n"
        "/start — Welcome message\n"
        "/help — This help message\n\n"
        "*Features:*\n"
        "• Ask about any government scheme for farmers\n"
        "• Supports Hindi and English text\n"
        "• Supports Hindi voice notes\n\n"
        "*Covered Schemes:*\n"
        "PM-KISAN, PMFBY (Crop Insurance), Kisan Credit Card, "
        "PM-KUSUM (Solar), Soil Health Card, eNAM, "
        "Agriculture Infrastructure Fund, and more.\n\n"
        "⚠️ *Disclaimer:* This bot provides general information only. "
        "Please verify details with your local agricultural office."
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming text messages."""
    query = update.message.text.strip()
    if not query:
        return

    if len(query) > 500:
        await update.message.reply_text(
            "Your message is too long. Please keep your question under 500 characters."
        )
        return

    logger.info("Text query from %s: %s", update.effective_user.id, query[:100])

    # Send typing indicator
    await update.message.chat.send_action("typing")

    try:
        result = pipeline.text_query(query)
        response = format_response(result)
        await update.message.reply_text(response)
    except TimeoutError:
        logger.error("LLM timeout for query: %s", query[:100])
        await update.message.reply_text(
            "The AI model is taking too long to respond. Please try again in a moment."
        )
    except RuntimeError as e:
        logger.error("LLM error: %s", e)
        await update.message.reply_text(
            "The AI model encountered an error. Please make sure the service is running and try again."
        )
    except Exception as e:
        logger.error("Error processing text query: %s", e)
        await update.message.reply_text(
            "Sorry, I encountered an error processing your question. Please try again."
        )


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming voice messages."""
    logger.info("Voice message from %s", update.effective_user.id)

    await update.message.chat.send_action("typing")

    try:
        # Download voice file
        voice = update.message.voice
        file = await context.bot.get_file(voice.file_id)

        # Save to temp .ogg file
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            ogg_path = tmp.name
            await file.download_to_drive(ogg_path)

        # Process through voice pipeline
        result = pipeline.voice_query(ogg_path)

        # Include transcription in response
        response_parts = [
            f"🎙️ Transcription: {result['transcription']}\n",
            format_response(result),
        ]
        await update.message.reply_text("\n".join(response_parts))

        # Clean up temp file
        os.unlink(ogg_path)

    except TimeoutError:
        logger.error("LLM timeout for voice message")
        await update.message.reply_text(
            "The AI model is taking too long to respond. Please try again in a moment."
        )
    except RuntimeError as e:
        logger.error("LLM error on voice: %s", e)
        await update.message.reply_text(
            "The AI model encountered an error. Please try again or send a text message."
        )
    except Exception as e:
        logger.error("Error processing voice message: %s", e)
        await update.message.reply_text(
            "Sorry, I couldn't process your voice message. "
            "Please try again or send a text message instead."
        )


def main():
    """Start the Telegram bot."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("Error: TELEGRAM_BOT_TOKEN not found in .env file.")
        print("Get a token from @BotFather on Telegram and add it to .env")
        return

    print("Starting KisanSathi Telegram bot...")

    app = Application.builder().token(token).build()

    # Register handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))

    print("Bot is running. Press Ctrl+C to stop.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
