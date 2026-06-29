import os
import re
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
import preprocess  # noqa: E402

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

# Conversation states (the last one collects user feedback for retraining)
PRICE, CATEGORY, FREQUENCY, ESSENTIAL, DELIBERATION, WISHLIST, FEEDBACK = range(7)

# Valid categories (must match the training data)
CATEGORIES = ['clothing', 'food', 'electronics', 'entertainment', 'home', 'beauty']

# Persistent menu button so the user never has to type /start again
NEW_CHECK_BUTTON = "🛒 Check a purchase"

# Frequency menu: 1..10 (1 = rarely, 10 = very often)
FREQUENCY_OPTIONS = [str(i) for i in range(1, 11)]

# Deliberation menu: each option maps to a representative number of minutes.
# The breakpoints respect the model's logic (the strong signals are < 10 min
# and 10-60 min), so the buckets land cleanly on either side of them.
DELIBERATION_OPTIONS = {
    "Less than 5 minutes": 3,
    "5-10 minutes": 8,
    "10-30 minutes": 20,
    "30-60 minutes": 45,
    "1-3 hours": 120,
    "4-8 hours": 360,
    "More than 8 hours": 1500,
}

# Feedback menu: consistent full level names (mapped back to the level number)
FEEDBACK_NAME_TO_LEVEL = {name: level for level, name in config.LEVEL_NAMES.items()}

# Per-level UX response (plain text, no markdown)
LEVEL_RESPONSE = {
    0: ("✅ Purchase approved", "This looks planned and low-risk. You're good to proceed."),
    1: ("🟡 Mildly impulsive", "Probably fine, but stay aware of why you're buying this."),
    2: ("⚠️ Moderately impulsive", "Worth a second thought - maybe sleep on it."),
    3: ("🚨 Strongly impulsive", "This looks like an emotional, unplanned buy. Wait 24 hours before completing it."),
}


def _yes_no(text):
    """Parse a yes/no answer into 1/0, or None if invalid."""
    t = text.lower().strip()
    if t in ("yes", "y", "1"):
        return 1
    if t in ("no", "n", "0"):
        return 0
    return None


def _menu(rows, one_time=True):
    """Build a reply keyboard."""
    return ReplyKeyboardMarkup(rows, one_time_keyboard=one_time, resize_keyboard=True)


# Begin a purchase check (triggered by /start OR by the menu button)
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.clear()
    await update.message.reply_text(
        "Welcome to ImpulseGuard! Let's check a purchase to help you avoid impulse spending.\n\n"
        "Please enter the price of the item in euros (for example: 45.50):",
        reply_markup=ReplyKeyboardRemove()
    )
    return PRICE


async def convert_price(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        context.user_data['price'] = float(update.message.text)
        reply_keyboard = [CATEGORIES[i:i + 2] for i in range(0, len(CATEGORIES), 2)]
        await update.message.reply_text(
            "Great! Now choose the category of this item:",
            reply_markup=_menu(reply_keyboard)
        )
        return CATEGORY
    except ValueError:
        await update.message.reply_text("Please enter a valid number for the price in euros (for example: 50 or 12.99):")
        return PRICE


async def convert_category(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    category = update.message.text.lower().strip()
    if category not in CATEGORIES:
        await update.message.reply_text(
            f"Please choose one of the following: {', '.join(CATEGORIES)}"
        )
        return CATEGORY
    context.user_data['category'] = category
    await update.message.reply_text(
        "How often do you buy items in this category? (1 = rarely, 10 = very often):",
        reply_markup=_menu([FREQUENCY_OPTIONS[:5], FREQUENCY_OPTIONS[5:]])
    )
    return FREQUENCY


async def convert_frequency(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text.strip()
    if text not in FREQUENCY_OPTIONS:
        await update.message.reply_text("Please tap a number from 1 to 10:")
        return FREQUENCY
    context.user_data['frequency'] = int(text)
    await update.message.reply_text(
        "Is this an essential need (like groceries or an appliance you must replace)?",
        reply_markup=_menu([["Yes", "No"]])
    )
    return ESSENTIAL


async def convert_essential(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    value = _yes_no(update.message.text)
    if value is None:
        await update.message.reply_text("Please tap Yes or No:")
        return ESSENTIAL
    context.user_data['is_essential'] = value
    deliberation_rows = [
        ["Less than 5 minutes", "5-10 minutes"],
        ["10-30 minutes", "30-60 minutes"],
        ["1-3 hours", "4-8 hours"],
        ["More than 8 hours"],
    ]
    await update.message.reply_text(
        "How much time did you spend thinking about this before buying?",
        reply_markup=_menu(deliberation_rows)
    )
    return DELIBERATION


async def convert_deliberation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    choice = update.message.text.strip()
    if choice not in DELIBERATION_OPTIONS:
        await update.message.reply_text("Please tap one of the time options:")
        return DELIBERATION
    context.user_data['deliberation_minutes'] = DELIBERATION_OPTIONS[choice]
    await update.message.reply_text(
        "Last question - was this planned ahead or on your wishlist?",
        reply_markup=_menu([["Yes", "No"]])
    )
    return WISHLIST


async def convert_wishlist(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    value = _yes_no(update.message.text)
    if value is None:
        await update.message.reply_text("Please tap Yes or No:")
        return WISHLIST
    context.user_data['on_wishlist'] = value

    await update.message.reply_text("Analyzing behavior with ImpulseGuard AI...", reply_markup=ReplyKeyboardRemove())

    if not pipeline:
        await update.message.reply_text("Error: AI model is offline. Could not complete prediction.")
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

    preprocess.log_prediction(purchase, level, channel="telegram")
    context.user_data['purchase'] = purchase

    title, message = LEVEL_RESPONSE[level]
    await update.message.reply_text(f"{title}\n\n{message}\n\nConfidence: {confidence:.0%}")

    # Ask for the TRUE level — this closes the human-in-the-loop.
    feedback_rows = [
        [config.LEVEL_NAMES[0], config.LEVEL_NAMES[1]],
        [config.LEVEL_NAMES[2], config.LEVEL_NAMES[3]],
        ["Skip"],
    ]
    await update.message.reply_text(
        "Was this right? Tap how impulsive it really was, to help me learn (or Skip):",
        reply_markup=_menu(feedback_rows)
    )
    return FEEDBACK


async def collect_feedback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    answer = update.message.text.strip()

    if answer.lower().startswith("skip"):
        await _show_menu(update, "No problem - no feedback recorded.")
        return ConversationHandler.END

    true_level = FEEDBACK_NAME_TO_LEVEL.get(answer)
    if true_level is None:
        await update.message.reply_text("Please tap one of the level buttons, or Skip:")
        return FEEDBACK

    preprocess.save_feedback(context.user_data['purchase'], true_level)
    await _show_menu(update, "Thanks! Your feedback was saved and will improve the next training run.")
    return ConversationHandler.END


async def _show_menu(update: Update, text):
    """Send a closing message plus the persistent 'check another' button."""
    await update.message.reply_text(
        f"{text}\n\nTap below whenever you want to check another purchase:",
        reply_markup=_menu([[NEW_CHECK_BUTTON]], one_time=False)
    )


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await _show_menu(update, "Evaluation canceled.")
    return ConversationHandler.END


def main() -> None:
    if not TOKEN:
        print("Error: TELEGRAM_BOT_TOKEN not found.")
        return

    application = Application.builder().token(TOKEN).build()

    conv_handler = ConversationHandler(
        # Start via /start OR by tapping the persistent menu button.
        entry_points=[
            CommandHandler("start", start),
            MessageHandler(filters.Regex(f"^{re.escape(NEW_CHECK_BUTTON)}$"), start),
        ],
        states={
            PRICE: [MessageHandler(filters.TEXT & ~filters.COMMAND, convert_price)],
            CATEGORY: [MessageHandler(filters.TEXT & ~filters.COMMAND, convert_category)],
            FREQUENCY: [MessageHandler(filters.TEXT & ~filters.COMMAND, convert_frequency)],
            ESSENTIAL: [MessageHandler(filters.TEXT & ~filters.COMMAND, convert_essential)],
            DELIBERATION: [MessageHandler(filters.TEXT & ~filters.COMMAND, convert_deliberation)],
            WISHLIST: [MessageHandler(filters.TEXT & ~filters.COMMAND, convert_wishlist)],
            FEEDBACK: [MessageHandler(filters.TEXT & ~filters.COMMAND, collect_feedback)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    application.add_handler(conv_handler)

    print("ImpulseGuard Bot with Real AI is running... Press Ctrl+C to stop.")
    application.run_polling()


if __name__ == '__main__':
    main()
