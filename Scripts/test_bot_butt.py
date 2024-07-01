import logging
from aiogram import Bot, Dispatcher, types
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.contrib.middlewares.logging import LoggingMiddleware
import asyncio
import os
from aiogram.utils import executor
import pandas as pd
from parser_1 import parse_channels 
# from preprocessing_classificator import preprocess_data, classify_text
from preprocessing_classificator import preprocess_data, classify_text_async

# Настройки бота aiogram
bot_token = ''
bot = Bot(token=bot_token)
dp = Dispatcher(bot)

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Команды для кнопок
start_keyboard = types.ReplyKeyboardMarkup(keyboard=[[types.KeyboardButton(text="/start")]], resize_keyboard=True)

parse_keyboard = types.ReplyKeyboardMarkup(keyboard=[[types.KeyboardButton(text="Начнем!")]], resize_keyboard=True)

period_keyboard = types.ReplyKeyboardMarkup(keyboard=[[types.KeyboardButton(text="Выбрать период")],
                                                      [types.KeyboardButton(text="Ввести конкретную дату")],
                                                      [types.KeyboardButton(text="К вводу каналов")]], resize_keyboard=True)

period_options_keyboard = types.ReplyKeyboardMarkup(keyboard=[[types.KeyboardButton(text="1 день")],
                                                              [types.KeyboardButton(text="3 дня")], [types.KeyboardButton(text="Назад")]], resize_keyboard=True)

# Состояния для FSM
class ParseStates:
    WAITING_FOR_LINKS = 1
    WAITING_FOR_PERIOD = 2
    WAITING_FOR_DATE = 3

# Переменная для хранения данных пользователя
user_data = {}

# Команда /start
@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    await message.reply("Привет! Я здесь, чтобы помочь тебе быть в курсе всех новостей 🤓. Давай начинать!", reply_markup=parse_keyboard)
    logger.info("Пользователь начал работу с ботом.")

# Команда "поехали!"
@dp.message_handler(lambda message: message.text == "Начнем!")
async def parse_command(message: types.Message):
    await message.answer("Введите ссылки на Telegram-каналы, новости из которых вы хотите узнать, разделенные запятой:", reply_markup=types.ReplyKeyboardRemove())
    user_data[message.from_user.id] = {'state': ParseStates.WAITING_FOR_LINKS}
    logger.info(f"Пользователь {message.from_user.id} ввел команду /parse")

# Обработка введенных ссылок
@dp.message_handler(lambda message: user_data.get(message.from_user.id, {}).get('state') == ParseStates.WAITING_FOR_LINKS)
async def handle_links(message: types.Message):
    user_data[message.from_user.id]['links'] = message.text
    user_data[message.from_user.id]['state'] = ParseStates.WAITING_FOR_PERIOD
    await message.answer("Выберите период поиска новостей или введите конкретную дату:", reply_markup=period_keyboard)
    logger.info(f"Пользователь {message.from_user.id} ввел ссылки: {message.text}")

# Обработка выбора периода
@dp.message_handler(lambda message: message.text in ["Выбрать период", "Ввести конкретную дату", "1 день", "3 дня", "Назад", "К вводу каналов"])
async def handle_period_choice(message: types.Message):
    user_id = message.from_user.id
    logger.info(f"Пользователь {user_id} выбрал опцию: {message.text}")
    logger.info(f"Текущее состояние пользователя: {user_data.get(user_id, {}).get('state')}")

    if message.text == "Выбрать период":
        await message.answer("Выберите период для парсинга:", reply_markup=period_options_keyboard)
    elif message.text == "Ввести конкретную дату":
        await message.answer("Введите дату в формате ГГГГ-ММ-ДД:")
        user_data[user_id]['state'] = ParseStates.WAITING_FOR_DATE
    elif message.text in ["1 день", "3 дня"]:
        period_days = 3 if message.text == "3 дня" else 1
        links = user_data[user_id]['links'].split(',')
        await start_parsing(message, links, period_days=period_days)
    elif message.text == "Назад":
        await message.answer("Выберите период поиска новостей или введите конкретную дату:", reply_markup=period_keyboard)
    elif message.text == "К вводу каналов":
        user_data[user_id]['state'] = ParseStates.WAITING_FOR_LINKS
        await message.answer("Введите ссылки на Telegram-каналы, новости из которых вы хотите узнать, разделенные запятой:",
                             reply_markup=types.ReplyKeyboardRemove())

# Обработка введенной конкретной даты
@dp.message_handler(lambda message: user_data.get(message.from_user.id, {}).get('state') == ParseStates.WAITING_FOR_DATE)
async def handle_date(message: types.Message):
    user_id = message.from_user.id
    date = message.text
    try:
        specific_date = pd.to_datetime(date)  # Проверка корректности даты
    except ValueError:
        await message.answer("Неверный формат даты. Попробуйте снова.")
        return
    
    links = user_data[user_id]['links'].split(',')
    await start_parsing(message, links, specific_date=specific_date)
    # user_data[user_id]['state'] = None  # Очищаем состояние после парсинга
    logger.info(f"Парсинг завершен для пользователя {user_id}")

async def start_parsing(message: types.Message, links: list, period_days=None, specific_date=None):
    user_id = message.from_user.id

    await message.answer("Начинаем парсинг...", reply_markup=types.ReplyKeyboardRemove())
    logger.info(f"Начинаем парсинг для пользователя {user_id} с периодом {period_days} дней и ссылками: {links}")
    success = await parse_channels(links, period_days=period_days, specific_date=specific_date)
    logger.info(f"Результат парсинга для пользователя {user_id}: {'успех' if success else 'неудача'}")

#Только для API - aсинхронно
    if success:
        await message.answer("Парсинг завершен. Начинаем классификацию...")

        # Классификация
        scripts_dir = os.path.dirname(os.path.abspath(__file__))  # Получаем текущую директорию скрипта
        filepath = os.path.join(scripts_dir, 'output.csv')
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df = preprocess_data(df)
            classified_df = await classify_text_async(df)  # Используем асинхронную классификацию

            # Сохранение данных с классификацией
            classified_df.to_csv(filepath, index=False)
            await message.answer(f"Классификация завершена. Данные сохранены в файле {filepath}.")
            logger.info(f"Классификация завершена. Данные сохранены в файле {filepath}.")
        else:
            await message.answer(f"Ошибка: файл {filepath} не найден.")
            logger.error(f"Ошибка: файл {filepath} не найден.")
    else:
        await message.answer("Парсинг завершен. Нет данных для сохранения.")

        logger.info("Парсинг завершен. Нет данных для сохранения.")
    user_data.pop(message.from_user.id, None)  # Очистка данных пользователя

#Только для API - синхронно
    # if success:
    #     await message.answer("Парсинг завершен. Начинаем классификацию...")

    #     # Классификация
    #     scripts_dir = os.path.dirname(os.path.abspath(__file__))  # Получаем текущую директорию скрипта
    #     filepath = os.path.join(scripts_dir, 'output.csv')
    #     if os.path.exists(filepath):
    #         df = pd.read_csv(filepath)
    #         df = preprocess_data(df)
    #         classified_df = classify_text_async(df)

    #         # Сохранение данных с классификацией
    #         classified_df.to_csv(filepath, index=False)
    #         await message.answer("Классификация завершена. Данные сохранены в файле output.csv.")
    #     else:
    #         await message.answer("Ошибка: файл output.csv не найден.")
    # else:
    #     await message.answer("Парсинг завершен. Нет данных для сохранения.")
    
    # user_data.pop(message.from_user.id, None)  # Очистка данных пользователя

###Только для локальной модели
    # if success:
    #     await message.answer("Парсинг завершен. Начинаем классификацию...")

    #     # Классификация
    #     scripts_dir = os.path.dirname(os.path.abspath(__file__))  # Получаем текущую директорию скрипта
    #     filepath = os.path.join(scripts_dir, 'output.csv')
    #     if os.path.exists(filepath):
    #         df = pd.read_csv(filepath)
    #         df = preprocess_data(df)
    #         classifier = init_classifier()
    #         df = classify_text(df, classifier)

    #         # Сохранение данных с классификацией
    #         df.to_csv(filepath, index=False)
    #         await message.answer(f"Классификация завершена. Данные сохранены в файле {filepath}.")
    #         logger.info(f"Классификация завершена. Данные сохранены в файле {filepath}.")
    #     else:
    #         await message.answer(f"Ошибка: файл {filepath} не найден.")
    #         logger.error(f"Ошибка: файл {filepath} не найден.")
    # else:
    #     await message.answer("Парсинг завершен. Нет данных для сохранения.")
    #     logger.info("Парсинг завершен. Нет данных для сохранения.")

    # user_data.pop(user_id, None)  # Очистка данных пользователя


# Запуск бота
if __name__ == '__main__':
    logger.info("Телеграм клиент запущен")
    executor.start_polling(dp, skip_updates=True)
    logger.info("Бот запущен")
#Выбор из имеющихся категорий
# await message.answer("Выберите категорию новостей:", reply_markup=generate_category_keyboard(classified_df['category'].unique()))
#             user_data[user_id]['state'] = ParseStates.CHOOSING_CATEGORY
