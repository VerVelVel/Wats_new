import logging
from aiogram import Bot, Dispatcher, types
import asyncio
import os
from aiogram.utils import executor
import pandas as pd
from parser_1 import parse_channels  # Импортируем функцию из parser.py
from preprocessing_classificator import preprocess_data, init_classifier, classify_text

# Настройки бота aiogram
bot_token = 'токен'
bot = Bot(token=bot_token)
dp = Dispatcher(bot)

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Команда /start
@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    await message.reply("Привет! Я бот для парсинга постов из Telegram каналов. Используйте команду /parse для начала.")
    logger.info("Пользователь начал работу с ботом.")

# Состояния для FSM
class ParseStates:
    WAITING_FOR_LINKS = 1
    WAITING_FOR_PERIOD = 2

# Переменная для хранения данных пользователя
user_data = {}

# Команда /parse
@dp.message_handler(commands=['parse'])
async def parse_command(message: types.Message):
    await message.answer("Введите ссылки на Telegram-каналы, разделенные запятой:")
    user_data[message.from_user.id] = {'state': ParseStates.WAITING_FOR_LINKS}
    logger.info(f"Пользователь {message.from_user.id} ввел команду /parse")

# Обработка введенных ссылок
@dp.message_handler(lambda message: user_data.get(message.from_user.id, {}).get('state') == ParseStates.WAITING_FOR_LINKS)
async def handle_links(message: types.Message):
    user_data[message.from_user.id]['links'] = message.text
    user_data[message.from_user.id]['state'] = ParseStates.WAITING_FOR_PERIOD
    await message.answer("Введите период для парсинга сообщений (30, 14, 7, 3, 1 дней):")
    logger.info(f"Пользователь {message.from_user.id} ввел ссылки: {message.text}")

# Обработка введенного периода
@dp.message_handler(lambda message: user_data.get(message.from_user.id, {}).get('state') == ParseStates.WAITING_FOR_PERIOD)
async def handle_period(message: types.Message):
    period = message.text
    if period not in ["30", "14", "7", "3", "1"]:
        await message.answer("Неверный период. Попробуйте снова.")
        logger.warning(f"Пользователь {message.from_user.id} ввел неверный период: {period}")
        return
    
    period_days = int(period)
    links = user_data[message.from_user.id]['links'].split(',')

    await message.answer("Начинаем парсинг...")
    success = await parse_channels(links, period_days)

    if success:
        await message.answer("Парсинг завершен. Начинаем классификацию...")

        # Классификация
        if os.path.exists('output.csv'):
            df = pd.read_csv('output.csv')
            df = preprocess_data(df)
            classifier = init_classifier()
            classified_df = classify_text(df, classifier)

            # Сохранение данных с классификацией
            classified_df.to_csv('output.csv', index=False)
            await message.answer("Классификация завершена. Данные сохранены в файле output.csv.")
        else:
            await message.answer("Ошибка: файл output.csv не найден.")
    else:
        await message.answer("Парсинг завершен. Нет данных для сохранения.")
    
    user_data.pop(message.from_user.id, None)  # Очистка данных пользователя


# Возможные категории для классификации
candidate_labels = ["технологии", "политика", "спорт", "экономика", "развлечения", "здоровье", "образование", "мода", "происшествия"]

# Запуск бота
if __name__ == '__main__':
    logger.info("Телеграм клиент запущен")
    executor.start_polling(dp, skip_updates=True)
    logger.info("Бот запущен")


