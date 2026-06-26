import os
import sys
import datetime
import logging
import joblib
import pandas as pd
from dotenv import load_dotenv
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    ConversationHandler,
)

# Reuse the model's config (paths, feature order, level names) as the single
# source of truth, so the bot can never drift from how the model was trained.
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ML_Model"))
import config  # noqa: E402

# Load environment variables
load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

# Load the trained pipeline (preprocessing + model in one artifact).
try:
    pipeline = joblib.load(config.MODEL_PATH)
    print("AI pipeline loaded successfully into the Bot!")
except Exception as e:
    print(f"Warning: Could not load AI pipeline. Error: {e}")
    pipeline = None

# Conversation states (now 6: we also ask about necessity, deliberation, planning)
PRICE, CATEGORY, FREQUENCY, ESSENTIAL, DELIBERATION, WISHLIST = range(6)

# Valid categories (must match the training data)
CATEGORIES = ['clothing', 'food', 'electronics', 'entertainment', 'home', 'beauty']

# Per-level UX response (matches the 4 graded levels the model outputs)
LEVEL_RESPONSE = {
    0: ("✅ *PURCHASE APPROVED!*", "This looks planned and low-risk. You're good to proceed."),
    1: ("🟡 *MILDLY IMPULSIVE*", "Probably fine, but stay aware of why you're buying this."),
    2: ("⚠️ *MODERATELY IMPULSIVE*", "Worth a second thought — maybe sleep on it."),
    3: ("🚨 *STRONGLY IMPULSIVE*", "This follows an emotional, unplanned pattern. Wait 24 hours before completing it. 🛑"),
}


def _yes_no(text):
    """Parse a yes/no answer into 1/0, or None if invalid."""
    t = text.lower().strip()
    if t in ("yes", "y", "1"):
        return 1
    if t in ("no", "n", "0"):
        return 0
    return None


# /start: begin the guided purchase simulator
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        "🧠 *Welcome to ImpulseGuard!*\n\n"
        "Let's evaluate your next purchase to prevent emotional spending.\n"
        "Please enter the **Price** of the item (e.g., 45.50):",
        parse_mode="Markdown"
    )
    return PRICE


async def convert_price(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        context.user_data['price'] = float(update.message.text)
        reply_keyboard = [CATEGORIES[i:i + 2] for i in range(0, len(CATEGORIES), 2)]
        await update.message.reply_text(
            "Great! Now select or type the **Category** of this item:",
            reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True, resize_keyboard=True)
        )
        return CATEGORY
    except ValueError:
        await update.message.reply_text("Please enter a valid number for the price (e.g., 50 or 12.99):")
        return PRICE


async def convert_category(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    category = update.message.text.lower().strip()
    if category not in CATEGORIES:
        await update.message.reply_text(
            f"Invalid category. Please choose one of the following: {', '.join(CATEGORIES)}"
        )
        return CATEGORY
    context.user_data['category'] = category
    await update.message.reply_text(
        "Got it. On a scale from 1 to 20, how **frequently** do you buy items in this category? "
        "(1 = Rarely, 20 = Very often):",
        reply_markup=ReplyKeyboardRemove()
    )
    return FREQUENCY


async def convert_frequency(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        frequency = int(update.message.text)
        if not (1 <= frequency <= 20):
            await update.message.reply_text("Please enter a number between 1 and 20:")
            return FREQUENCY
        context.user_data['frequency'] = frequency
        await update.message.reply_text(
            "Is this an **essential need** (e.g. groceries, an appliance you must replace)?",
            reply_markup=ReplyKeyboardMarkup([["Yes", "No"]], one_time_keyboard=True, resize_keyboard=True)
        )
        return ESSENTIAL
    except ValueError:
        await update.message.reply_text("Please enter a valid whole number between 1 and 20:")
        return FREQUENCY


async def convert_essential(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    value = _yes_no(update.message.text)
    if value is None:
        await update.message.reply_text("Please answer Yes or No:")
        return ESSENTIAL
    context.user_data['is_essential'] = value
    await update.message.reply_text(
        "How many **minutes** did you spend thinking about this before buying? (e.g. 2, 60, 1440):",
        reply_markup=ReplyKeyboardRemove()
    )
    return DELIBERATION


async def convert_deliberation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        minutes = int(update.message.text)
        if minutes < 0:
            raise ValueError
        context.user_data['deliberation_minutes'] = minutes
        await update.message.reply_text(
            "Last one — was this **planned ahead / on a wishlist**?",
            reply_markup=ReplyKeyboardMarkup([["Yes", "No"]], one_time_keyboard=True, resize_keyboard=True)
        )
        return WISHLIST
    except ValueError:
        await update.message.reply_text("Please enter a valid number of minutes (e.g. 5 or 120):")
        return DELIBERATION


async def convert_wishlist(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    value = _yes_no(update.message.text)
    if value is None:
        await update.message.reply_text("Please answer Yes or No:")
        return WISHLIST
    context.user_data['on_wishlist'] = value

    await update.message.reply_text(
        "Analyzing behavior with ImpulseGuard AI... 🔍",
        reply_markup=ReplyKeyboardRemove()
    )

    if not pipeline:
        await update.message.reply_text("⚠️ error: AI model is offline. Could not complete prediction.")
        return ConversationHandler.END

    # Assemble the purchase. hour is captured automatically at purchase time.
    purchase = {
        "hour": datetime.datetime.now().hour,
        "price": context.user_data['price'],
        "category": context.user_data['category'],
        "frequency": context.user_data['frequency'],
        "is_essential": context.user_data['is_essential'],
        "deliberation_minutes": context.user_data['deliberation_minutes'],
        "on_wishlist": context.user_data['on_wishlist'],
    }
    purchase_df = pd.DataFrame([purchase])[config.FEATURES]

    level = int(pipeline.predict(purchase_df)[0])
    probabilities = pipeline.predict_proba(purchase_df)[0]
    confidence = probabilities[list(pipeline.classes_).index(level)]

    _save_log(purchase, level)

    title, message = LEVEL_RESPONSE[level]
    await update.message.reply_text(
        f"{title}\n\n{message}\n\n_Confidence: {confidence:.0%}_",
        parse_mode="Markdown"
    )
    return ConversationHandler.END


def _save_log(purchase, level):
    """Append the prediction to a shared CSV log for the team."""
    try:
        os.makedirs("Logs", exist_ok=True)
        log_file = "Logs/predicted_transactions.csv"
        columns = ["timestamp"] + config.FEATURES + [config.TARGET]
        if not os.path.exists(log_file):
            with open(log_file, "w") as f:
                f.write(",".join(columns) + "\n")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [timestamp] + [str(purchase[c]) for c in config.FEATURES] + [str(level)]
        with open(log_file, "a") as f:
            f.write(",".join(row) + "\n")
        print(f"Log saved successfully to {log_file}")
    except Exception as log_error:
        print(f"Error saving log: {log_error}")


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        "Evaluation canceled. Use /start whenever you want to test another purchase.",
        reply_markup=ReplyKeyboardRemove()
    )
    return ConversationHandler.END


def main() -> None:
    if not TOKEN:
        print("Error: TELEGRAM_BOT_TOKEN not found.")
        return

    application = Application.builder().token(TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            PRICE: [MessageHandler(filters.TEXT & ~filters.COMMAND, convert_price)],
            CATEGORY: [MessageHandler(filters.TEXT & ~filters.COMMAND, convert_category)],
            FREQUENCY: [MessageHandler(filters.TEXT & ~filters.COMMAND, convert_frequency)],
            ESSENTIAL: [MessageHandler(filters.TEXT & ~filters.COMMAND, convert_essential)],
            DELIBERATION: [MessageHandler(filters.TEXT & ~filters.COMMAND, convert_deliberation)],
            WISHLIST: [MessageHandler(filters.TEXT & ~filters.COMMAND, convert_wishlist)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    application.add_handler(conv_handler)

    print("ImpulseGuard Bot with Real AI is running... Press Ctrl+C to stop.")
    application.run_polling()


if __name__ == '__main__':
    main()
