import os
import datetime
import logging
import joblib
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

# Load environment variables
load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

# Cargar el modelo e IA generados 
MODEL_PATH = "Results/Models/impulse_model.pkl"
ENCODER_PATH = "Results/Models/category_encoder.pkl"

try:
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    print("AI Model and Encoder loaded successfully into the Bot!")
except Exception as e:
    print(f"Warning: Could not load AI model or encoder. Error: {e}")
    model = None
    encoder = None

# Definir los estados de la conversación guiada
PRICE, CATEGORY, FREQUENCY = range(3)

# Categorías válidas basadas en el dataset 
CATEGORIES = ['clothing', 'food', 'electronics', 'entertainment', 'home', 'beauty']

# Comando /start: Inicia el simulador de compras
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        "🧠 *Welcome to ImpulseGuard!*\n\n"
        "Let's evaluate your next purchase to prevent emotional spending.\n"
        "Please enter the **Price** of the item (e.g., 45.50):",
        parse_mode="Markdown"
    )
    return PRICE

# Captura el precio y pide la categoría
async def convert_price(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        price = float(update.message.text)
        context.user_data['price'] = price
        
        # Crear botones rápidos para las categorías
        reply_keyboard = [CATEGORIES[i:i+2] for i in range(0, len(CATEGORIES), 2)]
        
        await update.message.reply_text(
            "Great! Now select or type the **Category** of this item:",
            reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True, resize_keyboard=True)
        )
        return CATEGORY
    except ValueError:
        await update.message.reply_text("Please enter a valid number for the price (e.g., 50 or 12.99):")
        return PRICE

# Captura la categoría y pide la frecuencia
async def convert_category(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    category = update.message.text.lower().strip()
    
    if category not in CATEGORIES:
        await update.message.reply_text(
            f"Invalid category. Please choose one of the following: {', '.join(CATEGORIES)}"
        )
        return CATEGORY
        
    context.user_data['category'] = category
    await update.message.reply_text(
        "Got it. Finally, on a scale from 1 to 20, how **frequently** do you buy items in this category? "
        "(1 = Rarely, 20 = Very often):",
        reply_markup=ReplyKeyboardRemove()
    )
    return FREQUENCY

# Captura la frecuencia, ejecuta la IA y da el veredicto
async def convert_frequency(update: Update, context: Update) -> int:
    try:
        frequency = int(update.message.text)
        if not (1 <= frequency <= 20):
            await update.message.reply_text("Please enter a number between 1 and 20:")
            return FREQUENCY
            
        context.user_data['frequency'] = frequency
        
        # Recuperar datos recolectados
        price = context.user_data['price']
        category = context.user_data['category']
        hour = datetime.datetime.now().hour # Captura la hora automática actual de la compra
        
        await update.message.reply_text("Analyzing behavior with ImpulseGuard AI... 🔍")
        
        if model and encoder:
            # 1. Transformar la categoría de texto a número con el encoder 
            category_encoded = encoder.transform([category])[0]
            
            # 2. Estructurar los datos exactos que pide el modelo: hour, price, category, frequency
            features = [[hour, price, category_encoded, frequency]]
            
            # 3. Realizar la predicción real
            prediction = int(model.predict(features)[0])
            
            # --- GUARDAR EN LOGS LOCALES PARA EL EQUIPO ---
            try:
                os.makedirs("Logs", exist_ok=True)
                log_file = "Logs/predicted_transactions.csv"
                
                # Crear cabecera si el archivo no existe
                if not os.path.exists(log_file):
                    with open(log_file, "w") as f:
                        f.write("timestamp,hour,price,category,frequency,is_impulsive\n")
                
                # Escribir los datos de la transacción actual
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(log_file, "a") as f:
                    f.write(f"{timestamp},{hour},{price},{category},{frequency},{prediction}\n")
                print(f"Log saved successfully to {log_file}")
            except Exception as log_error:
                print(f"Error saving log: {log_error}")
            # ----------------------------------------------
            
            # 4. Enviar respuesta UX personalizada al usuario
            if prediction == 1:
                await update.message.reply_text(
                    "🚨 *WARNING: IMPULSIVE PURCHASE DETECTED!*\n\n"
                    f"Our system predicts this {category} purchase at ${price:.2f} follows an emotional, unplanned pattern. "
                    "Take 10 deep breaths or wait 24 hours before completing this transaction. 🛑",
                    parse_mode="Markdown"
                )
            else:
                await update.message.reply_text(
                    "✅ *PURCHASE APPROVED!*\n\n"
                    f"This purchase looks planned and aligns with healthy financial behavior. "
                    "You are good to proceed safely!",
                    parse_mode="Markdown"
                )
        else:
            await update.message.reply_text("⚠️ error: AI model is offline. Could not complete prediction.")
            
        return ConversationHandler.END
        
    except ValueError:
        await update.message.reply_text("Please enter a valid whole number between 1 and 20:")
        return FREQUENCY

# Cancela la operación actual si el usuario lo desea
async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Evaluation canceled. Use /start whenever you want to test another purchase.", reply_markup=ReplyKeyboardRemove())
    return ConversationHandler.END

def main() -> None:
    if not TOKEN:
        print("Error: TELEGRAM_BOT_TOKEN not found.")
        return

    application = Application.builder().token(TOKEN).build()

    # Configurar el manejador de la conversación
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            PRICE: [MessageHandler(filters.TEXT & ~filters.COMMAND, convert_price)],
            CATEGORY: [MessageHandler(filters.TEXT & ~filters.COMMAND, convert_category)],
            FREQUENCY: [MessageHandler(filters.TEXT & ~filters.COMMAND, convert_frequency)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    application.add_handler(conv_handler)

    print("ImpulseGuard Bot with Real AI is running... Press Ctrl+C to stop.")
    application.run_polling()

if __name__ == '__main__':
    main()